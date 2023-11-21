import glob
import os
import json

import torch
import numpy as np
import cv2
import matplotlib.pyplot as plt

from model import HighResolutionNet
from draw_utils import draw_keypoints
import transforms

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(f"using device: {device}")
num_keypoints = 6
flip_test = False
resize_hw = (320, 320)
img_path = "/911G/data/新全经络数据/20230410/left_down_wai/test_crop/right_foot/l_down_wai_20230222132438048.jpg"
weights_path = "/home/zy/vision/deep-learning-for-image-processing/pytorch_keypoint/HRNet/save_weights/model-22.pth"
keypoint_json_path = "person_keypoints.json"
assert os.path.exists(img_path), f"file: {img_path} does not exist."
assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."


data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(1.25, 1.25), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def img2tensor(img_path):
    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)
    return img, img_tensor,target



def predict_single_person(file,img,img_tensor,target):
    
    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32,num_joints = num_keypoints)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # with torch.inference_mode():
    with torch.no_grad():
        outputs = model(img_tensor.to(device))
        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)

        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.1, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save(f"pred_output2/{file}")


if __name__ == '__main__':
    img_dir = "/home/zy/vision/ultralytics/yolo_crop/left_down_wai_right_foot"
    for img_path in glob.glob(os.path.join(img_dir,"*.jpg")):
        # img = cv2.imread(img_path)
        file = os.path.basename(img_path)
        img, img_tensor,target = img2tensor(img_path)

        predict_single_person(file,img,img_tensor,target)
