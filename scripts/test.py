import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.preprocess import AudioDataset
from utils.cnn import AudioCNN
from utils.hyper_parameters import HyperParameters


def load_model(model_name: str, model_dir: str, device: str):
    # model = torch.load(model_dir + model_name, map_location=torch.device(device), weights_only=False)
    model = torch.load(model_dir + model_name, weights_only=False)
    return model


def test(model: AudioCNN, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: str):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():  # No gradient calculation needed
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            # 前向传播
            outputs = model(features)

            # 计算总损失函数值
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 输出预测结果标签
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy


def test_urban_sound(model_name: str, cfg: HyperParameters):
    # 加载训练好的神经网络模型
    model = load_model(model_name, cfg.test.load_path, cfg.test.device)

    # 加载测试数据集
    test_dataset = AudioDataset(cfg.test)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=cfg.test.is_shuffle)

    

    # 加载损失函数
    criterion = nn.CrossEntropyLoss()

    # 进行测试
    test_loss, test_acc = test(model, test_dataloader, criterion, cfg.test.device)
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%")


if __name__ == "__main__":
    test_cfg = HyperParameters()
    model_name = 'model_lr0.001_batch10_epoch500.pkl'
    test_urban_sound(model_name, test_cfg)