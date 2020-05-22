import sys
import torch
from torch import nn
import torch.nn.init as init


def init_weights(modules):
    for m in modules:
        if isinstance(m, nn.Conv2d):
            init.xavier_uniform_(m.weight.data)
            if m.bias is not None:
                m.bias.data.zero_()
        elif isinstance(m, nn.BatchNorm2d):
            m.weight.data.fill_(1)
            m.bias.data.zero_()
        elif isinstance(m, nn.Linear):
            m.weight.data.normal_(0, 0.01)
            m.bias.data.zero_()


class res_block(nn.Module):

    def __init__(self, in_ch, out_ch):
        super(res_block, self).__init__()
        self.bn1 = nn.BatchNorm2d(in_ch)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(in_ch, in_ch, kernel_size=(3,3), padding=(1,1))

        self.bn2 = nn.BatchNorm2d(in_ch)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(in_ch, out_ch, kernel_size=(3,3), padding=(1,1))

        self.skip_conv = nn.Conv2d(in_ch, out_ch, kernel_size=1)

    def forward(self, x):
        skip_x = x
        x = self.conv1(self.relu1(self.bn1(x)))
        x = self.conv2(self.relu2(self.bn2(x)))
        x = x + self.skip_conv(skip_x)
        return x


class Model(nn.Module):

    def __init__(self, classes):
        super(Model, self).__init__()

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True),
            res_block(16, 32),
            res_block(32, 64),
            res_block(64, 128)
        )
        self.conv_cls = nn.Sequential(
            res_block(128, 64),
            res_block(64, 32),
            nn.Conv2d(32, classes + 1, kernel_size=1),
        )

        init_weights(self.conv.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x):
        feature = self.conv(x)
        y = self.conv_cls(feature)
        return y, feature

