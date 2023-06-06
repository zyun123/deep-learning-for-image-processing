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


def predict_all_person():
    # TODO
    pass


def predict_single_person():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"using device: {device}")

    num_joints = 74
    flip_test = False
    resize_hw = (384,640)  #32的正数倍
    # resize_hw = (256, 192)
    img_path = "/911G/data/temp/20221229新加手托脚托新数据/20230311_最新修改/middle_up_nei/test/m_up_nei_20221228151547322.jpg"
    # weights_path = "./pose_hrnet_w32_256x192.pth"
    weights_path = "/911G/EightModelOutputs/models/hrnet_384_640_20230329/model-17.pth"
    keypoint_json_path = "person_keypoints.json"
    assert os.path.exists(img_path), f"file: {img_path} does not exist."
    assert os.path.exists(weights_path), f"file: {weights_path} does not exist."
    assert os.path.exists(keypoint_json_path), f"file: {keypoint_json_path} does not exist."

    data_transform = transforms.Compose([
        transforms.AffineTransform(scale=(0.4, 1.0), fixed_size=resize_hw),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.596, 0.575, 0.554], std=[0.096,0.118,0.169])
    ])

    # read json file
    with open(keypoint_json_path, "r") as f:
        person_info = json.load(f)

    # read single-person image
    img = cv2.imread(img_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_tensor, target = data_transform(img, {"box": [0, 0, img.shape[1] - 1, img.shape[0] - 1]})
    img_tensor = torch.unsqueeze(img_tensor, dim=0)

    # create model
    # HRNet-W32: base_channel=32
    # HRNet-W48: base_channel=48
    model = HighResolutionNet(base_channel=32,num_joints = num_joints)
    weights = torch.load(weights_path, map_location=device)
    weights = weights if "model" not in weights else weights["model"]
    model.load_state_dict(weights)
    model.to(device)
    model.eval()

    # with torch.inference_mode():
    with torch.no_grad():
        
        for i in range(10):
            t1 = time.time()
            outputs = model(img_tensor.to(device))
          
        if flip_test:
            flip_tensor = transforms.flip_images(img_tensor)
            flip_outputs = torch.squeeze(
                transforms.flip_back(model(flip_tensor.to(device)), person_info["flip_pairs"]),
            )
            # feature is not aligned, shift flipped heatmap for higher accuracy
            # https://github.com/leoxiaobin/deep-high-resolution-net.pytorch/issues/22
            flip_outputs[..., 1:] = flip_outputs.clone()[..., 0: -1]
            outputs = (outputs + flip_outputs) * 0.5

        keypoints, scores = transforms.get_final_preds(outputs, [target["reverse_trans"]], True)
        keypoints = np.squeeze(keypoints)
        scores = np.squeeze(scores)
        t2 = time.time()
        print("predict used time: ", t2-t1)
        plot_img = draw_keypoints(img, keypoints, scores, thresh=0.2, r=3)
        plt.imshow(plot_img)
        plt.show()
        plot_img.save("test_result.jpg")


if __name__ == '__main__':
    predict_single_person()
