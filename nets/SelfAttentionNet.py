import torch
import torch.nn as nn
import torch.nn.functional as F

from utils.hyper_parameters import HyperParameters

class SelfAttentionClassifier(nn.Module):
    """
    用于音频分类的自注意力模型
    """
    def __init__(self, cfg: HyperParameters):
        super(SelfAttentionClassifier, self).__init__()
        
        self.cfg = cfg
        
        # 获取配置参数（假设cfg中有SelfAttention部分）
        input_dim = 1
        hidden_dim = 128
        num_classes = 10
        dropout_rate = 0.1
        
        # 降维层
        self.dim_reduction = nn.Linear(input_dim, hidden_dim)
        
        # 自注意力层
        self.query = nn.Linear(hidden_dim, hidden_dim)
        self.key = nn.Linear(hidden_dim, hidden_dim)
        self.value = nn.Linear(hidden_dim, hidden_dim)
        
        self.attention_scale = hidden_dim ** 0.5
        
        # 规范化层
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)
        
        # 前馈网络
        self.ffn = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2),
            nn.ReLU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
        
        # 输出层
        self.fc = nn.Linear(hidden_dim, num_classes)
        self.dropout = nn.Dropout(dropout_rate)


    def forward(self, x):
        # 处理输入 - 注意：这里假设输入x的形状为[batch_size, channels, height, width]（与ResNet相同）
        # 需要将其转换为序列形式 [batch_size, seq_len, features]
        batch_size = x.shape[0]
        
        # 调整输入形状 - 将通道维度的特征转换为序列
        x = x.permute(0, 2, 3, 1)  # [batch_size, height, width, channels]
        seq_len = x.shape[1] * x.shape[2]
        feature_dim = x.shape[3]
        x = x.reshape(batch_size, seq_len, feature_dim)  # [batch_size, seq_len, feature_dim]
        
        # 降维
        x = self.dim_reduction(x)  # [batch_size, seq_len, hidden_dim]
        
        # 自注意力机制 - 第一个子层
        residual = x
        
        q = self.query(x)  # [batch_size, seq_len, hidden_dim]
        k = self.key(x)    # [batch_size, seq_len, hidden_dim]
        v = self.value(x)  # [batch_size, seq_len, hidden_dim]
        
        # 计算注意力分数
        scores = torch.matmul(q, k.transpose(-2, -1)) / self.attention_scale
        attn_weights = F.softmax(scores, dim=-1)
        
        # 应用注意力
        context = torch.matmul(attn_weights, v)  # [batch_size, seq_len, hidden_dim]
        
        # 残差连接和层规范化
        x = self.layer_norm1(residual + context)  # [batch_size, seq_len, hidden_dim]
        
        # 前馈网络 - 第二个子层
        residual = x
        x = self.ffn(x)
        x = self.layer_norm2(residual + x)  # [batch_size, seq_len, hidden_dim]
        
        # 全局池化（取时间维度的平均）
        x = torch.mean(x, dim=1)  # [batch_size, hidden_dim]
        
        # 分类
        x = self.dropout(x)
        x = self.fc(x)  # [batch_size, num_classes]
        
        return x
