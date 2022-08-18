import copy
import glob
import os
import json
import time
import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms

def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    num_joints = 50 
    flip_test = False
    resize_hw = (288,384)
    # resize_hw = (384,640)
    # resize_hw = (256, 192)
    # img_path = "./000001.jpg"
    # weights_path = "./pose_hrnet_w32_256x192.pth"
    weights_path = "/911G/EightModelOutputs/models/hrnet_288_384_02/model-11.pth"
    keypoint_json_path = "person_keypoints.json"
    # assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


    model = HighResolutionNet(base_channel=32,num_joints = num_joints)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    kp_names = ['L-sanjiao-1', 'L-sanjiao-2', 'L-sanjiao-3', 'L-sanjiao-4', 'L-sanjiao-5',
                'L-sanjiao-6', 'L-sanjiao-7', 'L-sanjiao-8', 'L-sanjiao-9', 'R-sanjiao-1', 'R-sanjiao-2',
                 'R-sanjiao-3', 'R-sanjiao-4', 'R-sanjiao-5','R-sanjiao-6', 'R-sanjiao-7', 'R-sanjiao-8', 'R-sanjiao-9',
                'L-pangguang-9', 'L-pangguang-10', 'L-pangguang-11', 'L-pangguang-12', 'L-pangguang-13',
                'L-pangguang-14', 'L-pangguang-15', 'L-pangguang-16', 'L-pangguang-17', 'L-pangguang-18',
                'L-pangguang-19', 'L-pangguang-20', 'L-pangguang-21', 'L-pangguang-22', 'L-pangguang-23',
                'L-pangguang-24', 'R-pangguang-9', 'R-pangguang-10',
                'R-pangguang-11', 'R-pangguang-12', 'R-pangguang-13',
                'R-pangguang-14', 'R-pangguang-15', 'R-pangguang-16', 'R-pangguang-17', 'R-pangguang-18',
                'R-pangguang-19', 'R-pangguang-20', 'R-pangguang-21', 'R-pangguang-22', 'R-pangguang-23',
                'R-pangguang-24']

    img_path_list = glob.glob("/911G/data/semi_care_data/middle_down_wai/train/*.jpg")
    output_path = "/911G/newimage/output/hrnet_predict"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    with torch.inference_mode():
        count = 0
        for img_path in img_path_list:
            t1 = time.time()
            img = cv2.imread(img_path)
            org_image = copy.deepcopy(img)
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
            img_tensor = torch.unsqueeze(img_tensor, dim=0)
              
            outputs = model(img_tensor.to(device))
            # t2 = time.time()
            # print("predict used time: ", t2-t1)

            keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
            keypoints = np.squeeze(keypoints)
            scores = np.squeeze(scores)
            for i,key in enumerate(np.round(keypoints)):
                
                cv2.circle(org_image,(int(key[0]),int(key[1])),3,(0,0,255),-1,8,0)
                cv2.putText(org_image,kp_names[i].split("-")[-1],(int(key[0]),int(key[1])-2),cv2.FONT_HERSHEY_SIMPLEX,0.3,(255,0,0),1,8)
                # if i>0 and i < 9:
                #     cv2.circle(img,(int(key[0]),int(key[1])),3,(0,0,255),-1,8,0)
                #     cv2.line(img,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(0,255,255),1,8,0)
                # elif i>=9 and i <18:
                #     cv2.circle(img,(int(key[0]),int(key[1])),3,(255,0,0),-1,8,0)
                #     cv2.line(img,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(0,255,255),1,8,0)
                # elif i>=18 and i < 34:
                #     cv2.circle(img,(int(key[0]),int(key[1])),3,(0,255,0),-1,8,0)
                #     cv2.line(img,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(255,0,0),1,8,0)
                # elif i>=34 and i < 50:
                #     cv2.circle(img,(int(key[0]),int(key[1])),3,(255,0,255),-1,8,0)
                #     cv2.line(img,(int(last_key[0]),int(last_key[1])),(int(key[0]),int(key[1])),(0,0,255),1,8,0)
                # last_key = key
            t3 = time.time()
            print("predict and process keypoint and show used time: ", t3-t1)
            cv2.imshow("image",org_image)
            
            cv2.imwrite(os.path.join(output_path,f"{count}.jpg"),org_image)
            count+=1
            if cv2.waitKey(1)==ord("q"):
                break
            

        cv2.destroyAllWindows()    



if __name__ == '__main__':
    predict_single_person()
