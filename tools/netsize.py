import torchvision
from torch import nn

from testnet.day1222 import testnet11
from testnet.testmodule import xyASPP4, xyASPP5


# from model.DenseNetUNet import DenseUNet
# from model.day1215 import Net3

# model = Net3(n_channels=3, n_classes=1, kernel_size=9, number=16, imgsize=512, att='FEM')
class DeformConv(nn.Module):

    def __init__(self, in_channels, out_channel,groups, kernel_size=(3, 3), padding=1, stride=1, dilation=1, bias=True):
        super(DeformConv, self).__init__()

        self.offset_net = nn.Conv2d(in_channels=in_channels,
                                    out_channels=2 * kernel_size[0] * kernel_size[1],
                                    kernel_size=kernel_size,
                                    padding=padding,
                                    stride=stride,
                                    dilation=dilation,
                                    bias=True)

        self.deform_conv = torchvision.ops.DeformConv2d(in_channels=in_channels,
                                                        out_channels=out_channel,
                                                        kernel_size=kernel_size,
                                                        padding=padding,
                                                        groups=groups,
                                                        stride=stride,
                                                        dilation=dilation,
                                                        bias=False)

    def forward(self, x):
        offsets = self.offset_net(x)
        out = self.deform_conv(x, offsets)
        return out
model=testnet11(128)
total = sum([param.nelement() for param in model.parameters()])
# 精确地计算：1MB=1024KB=1048576字节
print('Number of parameter: % .4fM' % (total / 1e6))