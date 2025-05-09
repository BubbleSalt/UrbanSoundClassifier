from torchvision.models.resnet import ResNet, BasicBlock
import torch.nn as nn
import torch.nn.init as init
from utils.hyper_parameters import HyperParameters

class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _ = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1)
        return x * y

class SEBasicBlock(BasicBlock):
    def __init__(self, *args, reduction=16, **kwargs):
        super(SEBasicBlock, self).__init__(*args, **kwargs)
        self.se = SEBlock(self.conv2.out_channels, reduction=reduction)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out
    
class AudioResNetSE(nn.Module):
    def __init__(self, cfg: HyperParameters):
        super().__init__()
        self.cfg = cfg

        self.resnet = ResNet(
            block=SEBasicBlock,
            layers=[2, 2, 2, 2],
            num_classes=cfg.ResNet.linear_out_features
        )

        self.resnet.conv1 = nn.Conv2d(
            in_channels=cfg.ResNet.conv_in_channels,
            out_channels=cfg.ResNet.conv_out_channels,
            kernel_size=cfg.ResNet.kernel_size,
            stride=cfg.ResNet.stride,
            padding=cfg.ResNet.padding,
            bias=cfg.ResNet.bias
        )

    def forward(self, x):
        return self.resnet(x)

