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
from keypoints_names import *

def main():
    img_size = 512  # 必须是32的整数倍 [416, 512, 608]
    cfg = "cfg/my_yolov3.cfg"  # 改成生成的.cfg文件
    # weights = "weights/yolov3spp-voc-512.pt"  # 改成自己训练好的权重文件
    weights = "/911G/EightModelOutputs/models/yolo_kp_736_1280_02/yolov3spp-14.pt"  # 改成自己训练好的权重文件
    json_path = "./data/pascal_voc_classes.json"  # json标签文件
    img_path = "000001.jpg"
    assert os.path.exists(cfg), "cfg file {} dose not exist.".format(cfg)
    assert os.path.exists(weights), "weights file {} dose not exist.".format(weights)
    assert os.path.exists(json_path), "json file {} dose not exist.".format(json_path)
    assert os.path.exists(img_path), "image file {} dose not exist.".format(img_path)

    with open(json_path, 'r') as f:
        class_dict = json.load(f)

    category_index = {str(v): str(k) for k, v in class_dict.items()}

    # input_size = (img_size, img_size)
    input_size = (736,1280)
    # input_size = (640,640)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    model = Darknet(cfg, img_size)
    model.load_state_dict(torch.load(weights, map_location='cpu')["model"])
    model.to(device)

    model.eval()
    with torch.no_grad():
        # init
        img = torch.zeros((1, 3, 736,1280), device=device)
        model(img)


        


        img_o = cv2.imread(img_path)  # BGR
        assert img_o is not None, "Image Not Found " + img_path

        img ,ratio, (dw, dh)= img_utils.letterbox(img_o, new_shape=input_size, auto=True, color=(0, 0, 0))
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(device).float()
        img /= 255.0  # scale (0, 255) to (0, 1)
        img = img.unsqueeze(0)  # add batch dimension

        t1 = torch_utils.time_synchronized()
        pred, p,kp_logits = model(img)  # only get inference result
        t2 = torch_utils.time_synchronized()
        print("pred use time :*******",t2 - t1)

        pred = utils.non_max_suppression(pred, conf_thres=0.1, iou_thres=0.6, multi_label=True)[0]
        t3 = time.time()
        print("pre process use time:*******",t3 - t2)

        if pred is None:
            print("No target detected.")
            exit(0)

        # process detections
        pred[:, :4] = utils.scale_coords(img.shape[2:], pred[:, :4], img_o.shape).round()
        print(pred.shape)

        bboxes = pred[:, :4].detach().cpu().numpy()
        scores = pred[:, 4].detach().cpu().numpy()
        # classes = pred[:, 5].detach().cpu().numpy().astype(np.int) + 1
        
        for i in range(len(bboxes)):
            box = bboxes[i]
            pt1_x = int(box[0])
            pt1_y = int(box[1])
            pt2_x = int(box[2])
            pt2_y = int(box[3])
            cv2.rectangle(img_o,(pt1_x,pt1_y),(pt2_x,pt2_y),(255,0,0),2,8)
            cv2.putText(img_o,"person",(pt1_x,pt1_y),cv2.FONT_HERSHEY_COMPLEX,1,(0,255,0),1,8)



        #heatmaps to keypoints
        kp_logit = kp_logits[0]
        w = kp_logit.shape[2]
        h = kp_logit.shape[1]
        num_keypoints = kp_logit.shape[0]
        pos = kp_logit.reshape(num_keypoints,-1).argmax(dim=1)
        x_int = pos % w
        y_int = torch.div(pos - x_int, w, rounding_mode="floor")
        xy_preds = torch.zeros((num_keypoints,3),dtype = torch.float32,device=kp_logit.device)
        # kp_scores = torch.zeros(num_keypoints,dtype=torch.float32,device=kp_logit.device)


        kp_scores = kp_logit[torch.arange(num_keypoints,device = kp_logit.device),y_int,x_int]
        xy_preds[:,0] = x_int
        xy_preds[:,1] = y_int
        xy_preds[:,2] = kp_scores

        xy_preds = utils.scale_kp_coords(img.shape[2:], xy_preds, img_o.shape).round()
        
        visible = {}


        
        #middle_down_wai
        kp_names = COCO_PERSON_KEYPOINT_NAMES_DOWN + COCO_PERSON_KEYPOINT_NAMES_HEAD_MIDDLE_DOWN
        keypoint_color_rules = KEYPOINT_CONNECTION_RULES_WHOLE_DOWN
        for i in range(len(xy_preds)):
            x = int(xy_preds[i][0])
            y = int(xy_preds[i][1])
            kp_name = kp_names[i]
            visible[kp_name] = (x,y)
            score = xy_preds[i][2]
            cv2.circle(img_o,(x,y),2,(0,0,255),-1,8)
            cv2.putText(img_o,str(int(score)),(x,y),cv2.FONT_HERSHEY_COMPLEX,0.5,(255,0,255),1,8)
        
        for kp0,kp1,color in keypoint_color_rules:
            if kp0 in visible and kp1 in visible:
                x0, y0 = visible[kp0]
                x1, y1 = visible[kp1]
                # color = tuple(x / 255.0 for x in color)
                cv2.line(img_o,(x0,y0),(x1,y1),color,1,8)




        cv2.imshow("kp_img",img_o)
        cv2.waitKey(0)
        cv2.destroyAllWindows()




        # pil_img = Image.fromarray(img_o[:, :, ::-1])
        # plot_img = draw_objs(pil_img,
        #                      bboxes,
        #                      classes,
        #                      scores,
        #                      category_index=category_index,
        #                      box_thresh=0.2,
        #                      line_thickness=1,
        #                      font='arial.ttf',
        #                      font_size=5)
        # t4 = time.time()
        # print("predict and draw used time: ****",t4-t1)
        # plt.imshow(plot_img)
        # plt.show()
        # # 保存预测的图片结果
        # plot_img.save("test_result.jpg")


if __name__ == "__main__":
    main()
