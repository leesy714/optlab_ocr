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
        x =  x + self.skip_conv(skip_x)
        return x


class double_conv(nn.Module):
    def __init__(self, in_ch, mid_ch, out_ch):
        super(double_conv, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, mid_ch, kernel_size=(3,3), padding=(1, 1)),
            nn.BatchNorm2d(mid_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_ch, out_ch, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        x = self.conv(x)
        return x


class Model(nn.Module):
    TRAINED_MODEL = 'weights/craft_mlt_25k.pth'

    def __init__(self, classes, pretrained=False, preprocessed=True):

        assert not(pretrained and preprocessed)
        super(Model, self).__init__()


        if not preprocessed:
            self.craft = CRAFT()

            if pretrained:
                print('Loading weights from checkpoint (' + self.TRAINED_MODEL + ')')
                self.craft.load_state_dict(copyStateDict(torch.load(
                    self.TRAINED_MODEL, map_location='cpu')))
            for p in self.craft.parameters():
                p.requires_grad = False
            self.craft.eval()
        self.preprocessed = preprocessed

        self.conv = nn.Sequential(
            nn.Conv2d(4, 16, kernel_size=(3, 3), padding=(1, 1)),
            nn.BatchNorm2d(16),
            nn.ReLU(inplace=True)
        )
        # self.conv1 = double_conv(16, 32, 64)
        # self.conv2 = double_conv(64, 128, 256)
        self.conv1 = res_block(16, 32)
        self.conv2 = res_block(32, 64)
        self.conv3 = res_block(64, 128)


        self.conv_cls = nn.Sequential(
            #nn.Conv2d(128, 64, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            #nn.Conv2d(64, 32, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            res_block(128, 64),
            res_block(64, 32),
            nn.Conv2d(32, classes + 1, kernel_size=1),
        )

        init_weights(self.conv.modules())
        init_weights(self.conv1.modules())
        init_weights(self.conv2.modules())
        init_weights(self.conv3.modules())
        init_weights(self.conv_cls.modules())

    def forward(self, x, craft_y=None):
        assert not(self.preprocessed and craft_y is None)

        if not self.preprocessed:
            with torch.no_grad():
                craft_y, craft_feature = self.craft(x)
            craft_y = craft_y.permute(0, 3,1,2)
        pool_x = torch.max_pool2d(x, (2,2))
        x = torch.cat([pool_x, craft_y],dim=1)

        y = self.conv(x)
        feature = self.conv1(y)
        feature = self.conv2(feature)
        feature = self.conv3(feature)
        y = self.conv_cls(feature)

        return y, feature
