"""
old:images and one json(bbox:coco格式的左上角坐标和宽高,未归一化),labelme标注的也是这种格式
new:images and txts(per txt:some lines(class,归一化的cx、cy、w、h))
param: json_path, save_dir
"""
from pycocotools.coco import COCO
import os
from tqdm import tqdm
 
 
def cocotoyolo(coco_cor):       # len(coco_cor):4
    """coco格式的左上角坐标和宽高 -> yolo格式的cx、cy、w、h"""
    coco_cor[0] = coco_cor[0] + coco_cor[2] / 2
    coco_cor[1] = coco_cor[1] + coco_cor[3] / 2
    return coco_cor
 
 
def main(json_path, save_dir):
    """save_dir是指生成的txt文件的储存根目录"""
    coco = COCO(json_path)
    # [{'id': 0, 'width': 1537, 'height': 2049, 'file_name': 'batch_1/000006.jpg',....},........]
    images = coco.dataset["images"]
    # [{'id': 4783, 'image_id': 1499, 'category_id': 6,'bbox': [1125.0, 1858.0, 234.0, 510.0]},.....]
    annotations = coco.dataset['annotations']
 
    for img in tqdm(images):
        img_name = str(img["file_name"])        # 图片所在地址
 
        try:
            os.makedirs(os.path.join(save_dir, img_name.split('/')[0]))  # 创建保存路径
        except:
            pass
 
        txt_name = img_name.replace(img_name.split('.')[-1], 'txt')
        save_path = os.path.join(save_dir, txt_name)        # txt的保存路径
 
        with open(save_path, 'w') as f:
            img_id = img["id"]
            img_w = img["width"]
            img_h = img["height"]
 
            for ann in annotations:
                if int(ann["image_id"]) == int(img_id):     # 说明当前的ann属于当前的img
                    # coco格式的左上角坐标和宽高 -> yolo格式的cx、cy、w、h
                    bbox = cocotoyolo(ann["bbox"])
                    # 坐标归一化
                    bbox[0] /= img_w
                    bbox[2] /= img_w
                    bbox[1] /= img_h
                    bbox[3] /= img_h
                    temp = str(ann["category_id"]) + " "
                    for item in bbox:
                        assert 1 >= item >= 0, "bbox:ann_id is {}未归一化".format(ann["id"])
                        temp = temp + str(item) + " "
                    temp += '\n'
                    f.write(temp)
        f.close()
    print("successfully!")
 
 
if __name__ == "__main__":
    json_path = '/911G/data/coco2017/annotations/instances_train2017.json'
    save_dir = "./coco_yolo_dataset"
    main(json_path, save_dir)