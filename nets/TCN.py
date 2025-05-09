import torch
import torch.nn as nn
import torch.nn.functional as F

class TemporalBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, dilation=1, dropout=0.2, bias=False):
        super(TemporalBlock, self).__init__()
        
        # 计算填充大小以保持序列长度不变
        padding = (kernel_size-1) * dilation // 2
        
        # 第一个卷积层
        self.conv1 = nn.Conv1d(
            in_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=bias
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)
        
        # 第二个卷积层
        self.conv2 = nn.Conv1d(
            out_channels, out_channels, kernel_size,
            padding=padding, dilation=dilation, bias=bias
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        # 残差连接
        self.downsample = None
        if in_channels != out_channels:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, bias=bias),
                nn.BatchNorm1d(out_channels)
            )
            
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)
        
    def forward(self, x):
        residual = x
        
        # 第一层卷积块
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu1(out)
        out = self.dropout1(out)
        
        # 第二层卷积块
        out = self.conv2(out)
        out = self.bn2(out)
        
        # 残差连接
        if self.downsample is not None:
            residual = self.downsample(residual)
            
        out += residual
        out = self.relu2(out)
        out = self.dropout2(out)
        
        return out

class TCNClassifier(nn.Module):
    def __init__(self, cfg, num_classes=None):
        super(TCNClassifier, self).__init__()
        
        # 使用ResNet中的参数
        in_channels = 1
        base_channels = 64
        kernel_size = 7
        bias = False
        
        # 如果num_classes未指定，则从配置中获取
        if num_classes is None:
            num_classes = 10
        
        # 处理频谱图输入的层
        self.feature_extractor = nn.Sequential(
            nn.Conv2d(in_channels, base_channels//2, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(base_channels//2),
            nn.ReLU(),
            nn.MaxPool2d(2, 2),  # 下采样频谱图
            nn.Conv2d(base_channels//2, base_channels, kernel_size=3, padding=1, bias=bias),
            nn.BatchNorm2d(base_channels),
            nn.ReLU(),
            nn.MaxPool2d(2, 2)   # 进一步下采样
        )
        
        # TCN层的嵌入层 - 将2D特征转换为1D序列
        self.embedding = nn.Sequential(
            nn.Conv1d(base_channels*10, base_channels, kernel_size=1, bias=bias),  # 这里的10是估计值，需要根据实际调整
            nn.BatchNorm1d(base_channels),
            nn.ReLU()
        )
        
        # TCN层
        layers = []
        num_levels = 4
        channel_sizes = [base_channels, base_channels*2, base_channels*4, base_channels*8]
        dilations = [1, 2, 4, 8]
        
        for i in range(num_levels):
            in_ch = channel_sizes[i-1] if i > 0 else base_channels
            out_ch = channel_sizes[i]
            dilation = dilations[i]
            
            layers.append(TemporalBlock(
                in_ch, out_ch, 
                kernel_size=kernel_size, 
                dilation=dilation,
                bias=bias
            ))
            
            # 加入第二个相同dilation的块，类似于ResNet中每层有多个块
            layers.append(TemporalBlock(
                out_ch, out_ch, 
                kernel_size=kernel_size, 
                dilation=dilation,
                bias=bias
            ))
        
        self.tcn_blocks = nn.Sequential(*layers)
        
        # 全局池化
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # 分类器
        self.fc = nn.Linear(channel_sizes[-1], num_classes)
        
    def forward(self, x):
        # 输入x形状: [batch_size, channels, height, width] = [64, 1, 40, 87]
        
        # 使用2D卷积网络提取特征
        x = self.feature_extractor(x)  # 输出形状: [batch_size, base_channels, height/4, width/4]
        
        # 调整维度以适应1D卷积
        batch_size, channels, height, width = x.shape
        x = x.reshape(batch_size, channels * height, width)  # 将高度维度合并到通道维度
        
        # TCN处理
        x = self.embedding(x)
        x = self.tcn_blocks(x)
        
        # 全局池化和分类
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        
        return x
