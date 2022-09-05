import struct
import sys
from models import *


model = Darknet('cfg/yolov3-spp.cfg', (512, 512))
weights = 'weights/yolov3-spp-ultralytics-512.pt'
device = torch.device("cuda:0")
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

