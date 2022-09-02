import json
import copy
import torch
import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval
import pycocotools.mask as mask_util
from .distributed_utils import all_gather, is_main_process


def merge(img_ids, eval_results):
    """将多个进程之间的数据汇总在一起"""
    all_img_ids = all_gather(img_ids)
    all_eval_results = all_gather(eval_results)

    merged_img_ids = []
    for p in all_img_ids:
        merged_img_ids.extend(p)

    merged_eval_results = []
    for p in all_eval_results:
        merged_eval_results.extend(p)

    merged_img_ids = np.array(merged_img_ids)

    # keep only unique (and in sorted order) images
    # 去除重复的图片索引，多GPU训练时为了保证每个进程的训练图片数量相同，可能将一张图片分配给多个进程
    merged_img_ids, idx = np.unique(merged_img_ids, return_index=True)
    merged_eval_results = [merged_eval_results[i] for i in idx]

    return list(merged_img_ids), merged_eval_results


class EvalCOCOMetric:
    def __init__(self,
                 coco: COCO = None,
                 iou_type: str = None,
                 results_file_name: str = "predict_results.json",
                 classes_mapping: dict = None):
        self.coco = copy.deepcopy(coco)
        self.img_ids = []  # 记录每个进程处理图片的ids
        self.results = []
        self.aggregation_results = None
        self.classes_mapping = classes_mapping
        self.coco_evaluator = None
        assert iou_type in ["bbox", "segm", "keypoints"]
        self.iou_type = iou_type
        self.results_file_name = results_file_name

    def prepare_for_coco_detection(self,outputs):
        """将预测的结果转换成COCOeval指定的格式，针对目标检测任务"""
        
        # 遍历每张图像的预测结果
        for original_id, prediction in outputs.items():
            coco_results = []
            if len(prediction) == 0:
                continue
            img_id = original_id       
            if img_id in self.img_ids:
                # 防止出现重复的数据
                continue
            self.img_ids.append(img_id)

            boxes = prediction["boxes"]
            boxes = convert_to_xywh(boxes).tolist()
            labels = prediction["labels"].tolist()
            scores = prediction["scores"].tolist()
            coco_results.extend(
                [
                    {
                        "image_id": img_id,
                        "category_id": labels[k],
                        "bbox": box,
                        "score": scores[k],
                    }
                    for k, box in enumerate(boxes)]
            )
            
            self.results.append(coco_results)

    def prepare_for_coco_segmentation(self, targets, outputs):
        """将预测的结果转换成COCOeval指定的格式，针对实例分割任务"""
        # 遍历每张图像的预测结果
        for target, output in zip(targets, outputs):
            if len(output) == 0:
                continue

            img_id = int(target["image_id"])
            if img_id in self.img_ids:
                # 防止出现重复的数据
                continue

            self.img_ids.append(img_id)
            per_image_masks = output["masks"]
            per_image_classes = output["labels"].tolist()
            per_image_scores = output["scores"].tolist()

            masks = per_image_masks > 0.5

            res_list = []
            # 遍历每个目标的信息
            for mask, label, score in zip(masks, per_image_classes, per_image_scores):
                rle = mask_util.encode(np.array(mask[0, :, :, np.newaxis], dtype=np.uint8, order="F"))[0]
                rle["counts"] = rle["counts"].decode("utf-8")

                class_idx = int(label)
                if self.classes_mapping is not None:
                    class_idx = int(self.classes_mapping[str(class_idx)])

                res = {"image_id": img_id,
                       "category_id": class_idx,
                       "segmentation": rle,
                       "score": round(score, 3)}
                res_list.append(res)
            self.results.append(res_list)



    def prepare_for_coco_keypoint(self,outputs):
        
        for original_id, prediction in outputs.items():
            coco_results = []
            if len(prediction) == 0:
                continue

            img_id = original_id
            if img_id in self.img_ids:
                # 防止出现重复的数据
                continue

            self.img_ids.append(img_id)
            keypoints = prediction["keypoints"]
            scores = keypoints[:,2]
            mask = keypoints[:,2]>0.0
            if mask.sum() == 0:
                k_score = 0
            else:
                k_score = int(torch.mean(keypoints[:,2][mask]))
            keypoints = keypoints.flatten(start_dim=0).tolist()
            keypoints = [round(k,2) for k in keypoints]
            
            coco_results.append(
                    {
                        "image_id": original_id,
                        "category_id": 0,
                        'keypoints': keypoints,
                        "score": k_score,
                    }

            )
            self.results.append(coco_results)

     
    def update(self, outputs):
        if self.iou_type == "bbox":
            self.prepare_for_coco_detection(outputs)
        elif self.iou_type == "segm":
            self.prepare_for_coco_segmentation(outputs)
        elif self.iou_type == "keypoints":
            self.prepare_for_coco_keypoint(outputs)
        else:
            raise KeyError(f"not support iou_type: {self.iou_type}")

    def synchronize_results(self):
        # 同步所有进程中的数据
        eval_ids, eval_results = merge(self.img_ids, self.results)
        self.aggregation_results = {"img_ids": eval_ids, "results": eval_results}

        # 主进程上保存即可
        if is_main_process():
            results = []
            [results.extend(i) for i in eval_results]
            # write predict results into json file
            json_str = json.dumps(results, indent=4)
            with open(self.results_file_name, 'w') as json_file:
                json_file.write(json_str)

    def evaluate(self):
        # 只在主进程上评估即可
        if is_main_process():
            # accumulate predictions from all images
            coco_true = self.coco
            coco_pre = coco_true.loadRes(self.results_file_name)

            self.coco_evaluator = COCOeval(cocoGt=coco_true, cocoDt=coco_pre, iouType=self.iou_type)
            self.coco_evaluator.params.kpt_oks_sigmas = np.ones(coco_true.anns[1]["num_keypoints"],dtype=np.float)*0.01

            self.coco_evaluator.evaluate()
            self.coco_evaluator.accumulate()
            print(f"IoU metric: {self.iou_type}")
            self.coco_evaluator.summarize()

            coco_info = self.coco_evaluator.stats.tolist()  # numpy to list
            return coco_info
        else:
            return None

def convert_to_xywh(boxes):
    xmin, ymin, xmax, ymax = boxes.unbind(1)
    return torch.stack((xmin, ymin, xmax - xmin, ymax - ymin), dim=1)

