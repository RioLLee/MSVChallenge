import pdb
import torch
import torch.nn as nn
import math
import torch.nn.functional as F

'''
https://arxiv.org/abs/2103.02907
https://github.com/Andrew-Qibin/CoordAttention
'''

#relu6
class h_sigmoid(nn.Module):

    def __init__(self, inplace=True):
        super(h_sigmoid, self).__init__()
        self.relu = nn.ReLU6(inplace=inplace)

    def forward(self, x):
        return self.relu(x + 3) / 6

#sigmoid normalization
class h_swish(nn.Module):

    def __init__(self, inplace=True):
        super(h_swish, self).__init__()
        self.sigmoid = h_sigmoid(inplace=inplace)

    def forward(self, x):
        return x * self.sigmoid(x)

class CoordAtt(nn.Module):

    def __init__(self, inp, oup, reduction=8):
        super(CoordAtt, self).__init__()

        self.pool_h = nn.AdaptiveAvgPool2d((None, 1))

        self.pool_w = nn.AdaptiveAvgPool2d((1, None))

        # channel downsample
        mip = max(16, inp // reduction)

        # full convolution
        self.conv1 = nn.Conv2d(inp, mip, kernel_size=1, stride=1, padding=0)
        self.bn1 = nn.BatchNorm2d(mip)

        self.act = h_swish()
        
        self.conv_h = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        self.conv_w = nn.Conv2d(mip, oup, kernel_size=1, stride=1, padding=0)
        

    def forward(self, x):
        identity = x
        n,c,h,w = x.size()
        #x_h [n,c,h,1]
        x_h = self.pool_h(x)
        #x_w [n,c,1,w] ---> [n,c,w,1]
        x_w = self.pool_w(x).permute(0, 1, 3, 2)
        #y---> [n,c,w+h,1 ]
        y = torch.cat([x_h, x_w], dim=2)
        #channel downsample
        y = self.conv1(y)
        y = self.bn1(y)
        #relu6 + sigmoid
        y = self.act(y) 
        # split h+w ----> h and w
        x_h, x_w = torch.split(y, [h, w], dim=2)
        x_w = x_w.permute(0, 1, 3, 2)

        #linear connected full convolution
        a_h = self.conv_h(x_h).sigmoid()

        #linear connected full convolution
        a_w = self.conv_w(x_w).sigmoid()
        out = identity * a_w * a_h

        return out

if __name__ == "__main__":

   x = torch.Tensor(64,256,128,64)

   model = CoordAtt(256,256) 
   print('Number of model parameters: {}'.format(
      sum([p.data.nelement() for p in model.parameters()])))

   model(x) 
  
