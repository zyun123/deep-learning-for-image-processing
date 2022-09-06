import struct
import sys
from models import *
#from utils.utils import *
import torch
model = Darknet('cfg/my_yolov3.cfg', (512, 512))
weights = "/911G/EightModelOutputs/models/harhat_512_512/yolov3spp-149.pt"
# weights = sys.argv[1]
# dev = '0'
# device = torch_utils.select_device(dev)
device =torch.device("cuda")
model.load_state_dict(torch.load(weights, map_location=device)['model'])


with open('/home/zy/vision/tensorrtx/yolov3-spp/yolov3-spp_ultralytics68.wts', 'w') as f:
    f.write('{}\n'.format(len(model.state_dict().keys())))
    for k, v in model.state_dict().items():
        vr = v.reshape(-1).cpu().numpy()
        f.write('{} {} '.format(k, len(vr)))
        for vv in vr:
            f.write(' ')
            f.write(struct.pack('>f',float(vv)).hex())
        f.write('\n')

