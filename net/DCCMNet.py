import torch
import torch.nn as nn
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from einops.layers.torch import Rearrange
from einops import rearrange, repeat
import torchvision.models as models
import numpy as np
import math
import torch.nn.functional as F
import os
import time
import math
import copy
from functools import partial
from typing import Optional, Callable, Any
import numbers
from collections import OrderedDict

import torch.utils.checkpoint as checkpoint
from timm.models.layers import DropPath, trunc_normal_
from fvcore.nn import FlopCountAnalysis, flop_count_str, flop_count, parameter_count

from .ECM import ECM
from .FFM import FFM
from .Attention import Attention

groups = 1

DEV = False

def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(in_channels, out_channels, kernel_size, padding=(kernel_size // 2), bias=bias)

class DCCMNet(nn.Module):

    def __init__(self, img_ch=3, output_ch=1, side_dim=32):
        super(DCCMNet, self).__init__()
        self.H, self.W = 512, 512

        self.up = nn.Upsample(scale_factor=2, mode='bilinear')
        self.MaxPool = nn.MaxPool2d(kernel_size=2, stride=2)

        self.conv11 = nn.Conv2d(img_ch, 32, 1)
        self.conv1 = ECM(default_conv, 32, 3)
        self.conv21 = nn.Conv2d(32, 64, 1)
        self.conv2 = ECM(default_conv, 64, 3)
        self.conv31 = nn.Conv2d(64, 128, 1)
        self.conv3 = ECM(default_conv, 128, 3)
        self.conv41 = nn.Conv2d(128, 256, 1)
        self.conv4 = ECM(default_conv, 256, 3)
        self.conv51 = nn.Conv2d(256, 512, 1)
        self.conv5 = ECM(default_conv, 512, 3)

        self.attention1 = Attention(32, 8)
        self.attention2 = Attention(64, 8)
        self.attention3 = Attention(128, 16)
        self.attention4 = Attention(256, 16)
        self.attention5 = Attention(512, 16)

        self.fuse4 = FFM(512, 256)
        self.fuse3 = FFM(256, 128)
        self.fuse2 = FFM(128, 64)
        self.fuse1 = FFM(64, 32)

        self.side = nn.Conv2d(32, 1, 1, 1, 0)

    def forward(self, x):

        e1_ = self.conv11(x)
        e1 = self.conv1(e1_)

        e2_ = self.MaxPool(e1)
        e2_ = self.conv21(e2_)
        e2 = self.conv2(e2_)

        e3_ = self.MaxPool(e2)
        e3_ = self.conv31(e3_)
        e3 = self.conv3(e3_)

        e4_ = self.MaxPool(e3)
        e4_ = self.conv41(e4_)
        e4 = self.conv4(e4_)

        e5_ = self.MaxPool(e4)
        e5_ = self.conv51(e5_)
        e5 = self.conv5(e5_)

        attn_e1 = self.attention1(e1)
        attn_e2 = self.attention2(e2)
        attn_e3 = self.attention3(e3)
        attn_e4 = self.attention4(e4)
        attn_e5 = self.attention5(e5)

        d5 = self.up(attn_e5)
        d4_ = self.fuse4(d5, attn_e4)
        d4 = self.up(d4_)
        d3_ = self.fuse3(d4, attn_e3)
        d3 = self.up(d3_)
        d2_ = self.fuse2(d3, attn_e2)
        d2 = self.up(d2_)
        d1 = self.fuse1(d2, attn_e1)

        side1 = self.side(d1)

        return side1


if __name__ == '__main__':
    x = torch.randn([2, 3, 512, 512])
    model = DCCMNet()
    # atte = Attention(64, 8)
    # pool = nn.AdaptiveAvgPool2d(1)
    # out = atte(x)
    out = model(x)
    # out = pool(x)
    print(out.shape)
