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

groups = 1

DEV = False

class FFM(nn.Module):

    def __init__(self, x1_channels, x2_channels):
        super(FFM, self).__init__()

        self.W_gate = nn.Sequential(
            nn.Conv2d(x1_channels, x1_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=x2_channels),
            nn.Conv2d(x1_channels, x2_channels, kernel_size=1, stride=1, padding=0, bias=True),
            nn.BatchNorm2d(x2_channels)
        )

        self.W_x = nn.Sequential(
            nn.Conv2d(x2_channels, x2_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=x2_channels),
            nn.BatchNorm2d(x2_channels)
        )

        self.psi = nn.Sequential(
            nn.Conv2d(x2_channels, 1, kernel_size=1, stride=1, padding=0, bias=True),
            nn.Sigmoid()
        )
        self.out = nn.Sequential(
            nn.Conv2d(x2_channels, x2_channels, kernel_size=3, stride=1, padding=1, bias=True, groups=x2_channels),
            nn.BatchNorm2d(x2_channels)
        )

        self.relu = nn.ReLU(inplace=True)

    def forward(self, gate, skip_connection):
        g1 = self.W_gate(gate)
        x1 = self.W_x(skip_connection)
        psi = self.relu(g1 + x1)
        psi = self.psi(psi)
        out = self.out(skip_connection * psi)
        return out