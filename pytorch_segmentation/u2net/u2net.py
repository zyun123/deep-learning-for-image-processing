import torch
import torch.nn as nn
import torchvision
import torch.nn.functional as F


class ConvBNReLU(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,dilation=1):
        padding = kernel_size // 2 if dilation ==1 else dilation
        self.conv = nn.Conv2d(in_ch, out_ch, kernel_size, padding = padding,bias = False)
        self.bn = nn.BatchNorm2d(out_ch)
        self.relu = nn.ReLU(inplace = True)

    def forward(self,x):
        return self.relu(self.bn(self.conv(x)))
    

class DownConvBNReLU(ConvBNReLU):
    def __init__(self,in_ch,out_ch,kernel_size=3,dilation=1,flag = True):
        super().__init__(in_ch,out_ch,kernel_size,dilation)
        self.down_flag = flag

    def forward(self,x):
        if self.down_flag:
            x = F.max_pool2d(x,kernel_size =2,stride = 2,ceil_mode = True)
        return self.relu(self.bn(self.conv(x)))
    

class UpConvBNReLU(ConvBNReLU):
    def __init__(self, in_ch, out_ch, kernel_size=3,dilation = 1,flag = True):
        super().__init__(in_ch,out_ch,kernel_size,dilation)
        self.up_flag = flag

    def forward(self,x1,x2):
        if self.up_flag:
            x1 = F.interpolate(x1,size = x2.shape[2:],mode="bilinear",align_corners = False)
        return self.relu(self.bn(self.conv(torch.cat([x1,x2]),dim = 1)))    

class RSU(nn.Module):
    def __init__(self,height,in_ch,mid_ch,out_ch,):
        super().__init__()

        self.conv_in = ConvBNReLU(in_ch,out_ch)
        encode_list = [DownConvBNReLU(out_ch,mid_ch,flag=False)]
        decode_list = [UpConvBNReLU(mid_ch *2,mid_ch,flag = False)]
        for i in range(height -2):
            encode_list.append(DownConvBNReLU(mid_ch,mid_ch))
            decode_list.append(UpConvBNReLU(mid_ch*2,mid_ch if i < height -3 else out_ch))  
        encode_list.append(ConvBNReLU(mid_ch,mid_ch,dilation=2))
        self.encode_modules = nn.ModuleList(encode_list)
        self.decode_modules = nn.ModuleList(decode_list)
    
    def forward(self,x):
        x_in = self.conv_in(x)
        x = x_in
        encode_outputs = []
        for m in self.encode_modules:
            x  = m(x)
            encode_outputs.append(x)
        x = encode_outputs.pop()
        for m in self.decode_modules:
            x2 = encode_outputs.pop()
            x = m(x,x2)
        return x +x_in  

class RSU4F(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super().__init__()
        self.conv_in = ConvBNReLU(in_ch,out_ch)
        self.encode_modules = nn.ModuleList([])
        






class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
