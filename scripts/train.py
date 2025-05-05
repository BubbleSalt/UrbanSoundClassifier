import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.hyper_parameters import HyperParameters
from utils.preprocess import AudioDataset
from utils.cnn import AudioCNN


"""模型训练函数"""
def train_per_epoch(model: AudioCNN, train_loader: DataLoader, optimizer: torch.optim.Adam, criterion: nn.CrossEntropyLoss, device: str):
    print("locate: train.py -> train_per_epoch()")

    # 进行一轮训练并计算损失函数值和正确率
    model.train()
    total_loss, total_correct = 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (out.argmax(1) == y).sum().item()

    return total_loss / len(train_loader), total_correct / len(train_loader.dataset)


def train_urban_sound(cfg: HyperParameters):
    print("locate: train.py -> train_urban_sound()")
    print("START")

    # 加载训练数据集
    dataset = AudioDataset(cfg.train)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.is_shuffle)

    # 加载卷积神经网络
    model = AudioCNN(cfg).to(cfg.train.device)

    # 加载优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # 加载损失函数
    criterion = nn.CrossEntropyLoss()       # 使用交叉熵损失
    
    # 进行训练
    for epoch in range(cfg.train.epochs):
        loss, acc = train_per_epoch(model, dataloader, optimizer, criterion, cfg.train.device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc*100:.2f}%")

    return model

if __name__ == "__main__":
    train_cfg = HyperParameters()
    urban_sound_cnn = train_urban_sound(train_cfg)

    model_name = f"model_lr{train_cfg.train.lr}_batch{train_cfg.train.batch_size}_epoch{train_cfg.train.epochs}.pkl"
    torch.save(urban_sound_cnn, train_cfg.train.save_dir + model_name)