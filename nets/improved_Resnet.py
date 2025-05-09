from torchvision.models import resnet18
import torch.nn as nn
import torch.nn.init as init
from utils.hyper_parameters import HyperParameters


class AudioImprovedResNet(nn.Module):
    """
        用于音频分类的残差卷积神经网络

        该模型基于预训练的 ResNet-18 架构，通过修改输入卷积层和全连接层来适应音频分类任务。
    """
    def __init__(self, cfg: HyperParameters):
        super(AudioImprovedResNet, self).__init__()

        self.cfg = cfg

        # 验证配置参数
        assert isinstance(cfg.ResNet.conv_in_channels, int) and cfg.ResNet.conv_in_channels > 0, "Invalid input channels for conv1"
        assert isinstance(cfg.ResNet.conv_out_channels, int) and cfg.ResNet.conv_out_channels > 0, "Invalid output channels for conv1"
        assert isinstance(cfg.ResNet.kernel_size, int) and cfg.ResNet.kernel_size > 0, "Invalid kernel size for conv1"
        assert isinstance(cfg.ResNet.stride, int) and cfg.ResNet.stride > 0, "Invalid stride for conv1"
        assert isinstance(cfg.ResNet.padding, int) and cfg.ResNet.padding >= 0, "Invalid padding for conv1"
        assert isinstance(cfg.ResNet.bias, bool), "Invalid bias value for conv1"
        assert isinstance(cfg.ResNet.linear_out_features, int) and cfg.ResNet.linear_out_features > 0, "Invalid output features for fc layer"

        # 加载 ResNet-18 模型
        self.resnet = resnet18(weights=cfg.ResNet.weights)

        # 修改输入卷积层以适应音频数据
        self.resnet.conv1 = nn.Conv2d(
            in_channels=cfg.ResNet.conv_in_channels,
            out_channels=cfg.ResNet.conv_out_channels,
            kernel_size=cfg.ResNet.kernel_size,
            stride=cfg.ResNet.stride,
            padding=cfg.ResNet.padding,
            bias=cfg.ResNet.bias
        )

        # 初始化修改后的卷积层
        init.kaiming_normal_(self.resnet.conv1.weight, mode='fan_out', nonlinearity='relu')
        if cfg.ResNet.bias:
            init.constant_(self.resnet.conv1.bias, 0)

        # 修改全连接层以适应分类任务
        self.resnet.fc = nn.Linear(self.resnet.fc.in_features, cfg.ResNet.linear_out_features)

        # 初始化修改后的全连接层
        init.normal_(self.resnet.fc.weight, 0, 0.01)
        init.constant_(self.resnet.fc.bias, 0)

    def forward(self, x):
        return self.resnet(x)