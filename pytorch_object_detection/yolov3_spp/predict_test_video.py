import os
import json
import time
import torch
import cv2
import numpy as np
from matplotlib import pyplot as plt
from PIL import Image
from build_utils import img_utils, torch_utils, utils
from models import Darknet
from draw_box_utils import draw_objs


def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    weights = "/911G/EightModelOutputs/models/harhat_512_512/yolov3spp-155.pt"  # 改成自己训练好的权重文件
    json_path = "./data/coco_classes.json"  # json标签文件
    img_path = "/911G/data/hard_hat/test/96.jpg"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    input_size = (img_size, img_size)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, img_size, img_size), device=device)
        model(img)

        capture = cv2.VideoCapture("/911G/data/hard_hat/1661916391108032.mp4")
        frame_width = int(capture.get(3))
        frame_height = int(capture.get(4))
        frame_size = (frame_width,frame_height)
        fps = 30
        output = cv2.VideoWriter("/911G/data/result.mp4",cv2.VideoWriter_fourcc(*'XVID'),fps,frame_size)
        while capture.isOpened():
            ret,img_o = capture.read()
            if ret:
                t1 = time.time()
                img = img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))[0]
                # Convert
                img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
                img = np.ascontiguousarray(img)

                img = torch.from_numpy(img).to(device).float()
                img /= 255.0  # scale (0, 255) to (0, 1)
                img = img.unsqueeze(0)  # add batch dimension
                pred = model(img)[0]  # only get inference result
                pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
              

                if pred is None:
                    # print("No target detected.")
                    # exit(0)
                    continue
                # process detections
                pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
                print(pred.shape)

                bboxes = pred[:, :4].detach().cpu().numpy().astype(np.int64)
                scores = pred[:, 4].detach().cpu().numpy()
                classes = pred[:, 5].detach().cpu().numpy().astype(np.int64) + 1
                
                # for i in range(len(bboxes)):
                #     box = bboxes[i]
                #     cv2.rectangle(img_o,(box[0],box[1]),(box[2],box[3]),(255,0,255),2,8)
                # t2 = time.time()
                # print("predict use time:***",t2-t1)
                # cv2.imshow("img_o",img_o)
                # k =cv2.waitKey(1)
                # if k == 27:
                #     break


                pil_img = Image.fromarray(img_o[:, :, ::-1])
                plot_img = draw_objs(pil_img,
                                    bboxes,
                                    classes,
                                    scores,
                                    category_index=category_index,
                                    box_thresh=0.2,
                                    line_thickness=2,
                                    font='arial.ttf',
                                    font_size=10)
                res_img = np.array(plot_img)[...,::-1]
                cv2.imshow("res_img",res_img)
                # output.write(res_img)
                k = cv2.waitKey(1)
                if k == 27:
                    break
            else:
                break
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
