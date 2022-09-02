from openvino.runtime import Core
from openvino.offline_transformations import serialize
import cv2
import numpy as np
import time
from deploy.load_onnx_test import scale_img
ie = Core()
print("use core engine inference")
devices = ie.available_devices
for device in devices:
    device_name = ie.get_property(device_name = device,name = "FULL_DEVICE_NAME")
    print(f"{device}:{device_name}")


# #####export onnx to bin,xml
# onnx_model_path = "onnx_export_weight/yolov3spp_kp.onnx"  
# model_onnx = ie.read_model(model = onnx_model_path)
# compiled_model_onnx = ie.compile_model(model = model_onnx,device_name = "CPU")
# serialize(model = model_onnx,
#             model_path ="onnx_export_weight/yolov3spp_kp.xml",
#             weights_path = "onnx_export_weight/yolov3spp_kp.bin")


model_xml = "onnx_export_weight/yolov3spp_kp.xml"
model_bin = "onnx_export_weight/yolov3spp_kp.bin"
model = ie.read_model(model = model_xml)
# model = ie.read_network(model = model_xml,weights = model_bin)

compiled_model = ie.compile_model(model = model,device_name = "CPU")

# #input layer info
# input_layer = model.input(0)
# print("input layer:####\n",input_layer)
# print("input layer precision:",input_layer.element_type)
# print("input layer shape:##",input_layer.shape)

# #output layer info
# output_layer = model.output(3)
# print("output layer:####\n",output_layer)
# # print("output layer precision:",output_layer.element_type)
# # print("output layer shape:##",output_layer.shape)


# image = cv2.imread("000001.jpg").transpose(2, 0, 1)[None,:].astype(np.float32)/255
img_o = cv2.imread("000001.jpg")
input_size = (736, 1280)
# preprocessing img
img, ratio, pad = scale_img(img_o, new_shape=input_size, auto=False, color=(0, 0, 0))
# Convert
img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
img = np.ascontiguousarray(img).astype(np.float32)

img /= 255.0  # scale (0, 255) to (0, 1)
img = np.expand_dims(img, axis=0)  # add batch dimension
time1 = time.time()
for i in range(10):
    result = compiled_model([img])
time2 = time.time()

print("openvino inference used time:##",time2-time1)
print(result)
# request = compiled_model.create_infer_request()
# request.infer({input_layer.any_name:image})
# kp = request.get_tensor("keypoints").data
# print(kp)
