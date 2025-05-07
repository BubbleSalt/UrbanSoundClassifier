import torch.nn as nn
import torch.nn.functional as F

from utils.hyper_parameters import HyperParameters

class AudioCNN(nn.Module):
    """
        用于音频分类的卷积神经网络
    """
    def __init__(self, cfg: HyperParameters):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d( in_channels=cfg.cnn.conv1_in_channels,
                                out_channels=cfg.cnn.conv1_out_channels,
                                kernel_size=cfg.cnn.conv1_kernel_size,
                                padding=cfg.cnn.conv1_padding )
        
        self.bn1 = nn.BatchNorm2d(cfg.cnn.conv1_out_channels)

        self.conv2 = nn.Conv2d( in_channels=cfg.cnn.conv2_in_channels,
                                out_channels=cfg.cnn.conv2_out_channels,
                                kernel_size=cfg.cnn.conv2_kernel_size,
                                padding=cfg.cnn.conv2_padding )
        
        self.bn2 = nn.BatchNorm2d(cfg.cnn.conv2_out_channels)

        self.pool = nn.AdaptiveAvgPool2d(cfg.cnn.pool_size)

        self.fc = nn.Linear(cfg.cnn.linear_in_features, cfg.cnn.linear_out_features)


    # 重写前向传播函数
    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)