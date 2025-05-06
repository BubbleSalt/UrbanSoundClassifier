from torchvision.models import resnet18
import torch.nn as nn

from utils.hyper_parameters import HyperParameters

class AudioResNet(nn.Module):
    """
        用于音频分类的残差卷积神经网络
    """
    def __init__(self, cfg: HyperParameters):
        super(AudioResNet, self).__init__()

        self.cfg = cfg

        self.resnet = resnet18(weights=cfg.ResNet.weights)
        self.resnet.conv1 = nn.Conv2d(  in_channels=cfg.ResNet.conv_in_channels,
                                        out_channels=cfg.ResNet.conv_out_channels,
                                        kernel_size=cfg.ResNet.kernel_size,
                                        stride=cfg.ResNet.stride,
                                        padding=cfg.ResNet.padding,
                                        bias=cfg.ResNet.bias  )
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, cfg.ResNet.linear_out_features)

    def forward(self, x):
        return self.resnet(x)