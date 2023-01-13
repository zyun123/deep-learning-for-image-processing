import torch
import torchvision
from model import resnet34
import os
from PIL import Image
from torchvision import transforms
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
model = resnet34(num_classes =2)
weights_path = "./resNet34.pth"
assert os.path.exists(weights_path), "file: '{}' dose not exist.".format(weights_path)
model.load_state_dict(torch.load(weights_path, map_location="cpu"))
inputs = torch.rand(1,3,224,224)

# torch_script_module = torch.jit.trace(model,inputs)

data_transform = transforms.Compose([
        transforms.Resize(224),
        #  transforms.CenterCrop(224),
         transforms.ToTensor(),
        #  transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
         ])
# load image
img_path = "images/middle_down_wai_20230113133716830.jpg"
assert os.path.exists(img_path), "file: '{}' dose not exist.".format(img_path)
img = Image.open(img_path)

# [N, C, H, W]
img = data_transform(img)
# expand batch dimension
img = torch.unsqueeze(img, dim=0)
model.eval()
# model.to(device)
with torch.no_grad():
    torch_script_module = torch.jit.trace(model,inputs)

    output = torch_script_module(img)
    # output = torch_script_module(torch.ones(1,3,224,224))
    output2 = torch.softmax(torch.squeeze(output).cpu(),dim=0)
    print("output: ",output)
    print("output2: ",output2)
    print(output.shape)

    #---------------save model script
    torch_script_module.save("./deploy/traced_resnet_model.pt")
