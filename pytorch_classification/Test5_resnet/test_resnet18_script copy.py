import torch
import torchvision



model = torchvision.models.resnet18(pretrained = "./resnet18.pth")

inputs = torch.rand(1,3,224,224)

torch_script_module = torch.jit.trace(model,inputs)

output = torch_script_module(torch.ones(1,3,224,224))
print(output)
print(output.shape)

#---------------save model script
torch_script_module.save("./deploy/traced_resnet_model.pt")
