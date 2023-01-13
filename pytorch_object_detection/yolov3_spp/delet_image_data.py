import json
import glob
import os

root_dir = "/home/zy/vision/deep-learning-for-image-processing/pytorch_object_detection/yolov3_spp/predict_output"
for path in glob.glob(os.path.join(root_dir,"*.json")):
    with open(path,"r") as f:
        data_dict = json.load(f)
        data_dict["imagePath"] = os.path.basename(path)
        data_dict["imageData"] = None
    
    with open(path,"w") as f:
        json.dump(data_dict,f,indent=4)
