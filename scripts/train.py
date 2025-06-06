import sys, os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

from torch.utils.tensorboard import SummaryWriter

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.hyper_parameters import HyperParameters
from utils.preprocess import AudioDataset

from nets.cnn import AudioCNN
from nets.improved_cnn import AudioImprovedCNN
from nets.ResNet import AudioResNet
from nets.improved_Resnet import AudioImprovedResNet
from nets.SelfAttentionNet import SelfAttentionClassifier
from nets.TCN import TCNClassifier
from nets.ResNetSE import AudioResNetSE

"""模型训练函数"""
def train_per_epoch(model, train_loader: DataLoader, optimizer: torch.optim.Adam, criterion: nn.CrossEntropyLoss, device: str):
    print("locate: train.py -> train_per_epoch()")
    print('START')

    # 进行一轮训练并计算损失函数值和正确率
    model.train()
    total_loss, total_correct = 0, 0
    for X, y in train_loader:
        X, y = X.to(device), y.to(device)

        """
        print('type of X is:', type(X))
        print('shape of X is:', X.shape)
        
        print('type of y is:', type(y))
        print('shape of y is:', y.shape)
        """

        optimizer.zero_grad()

        outputs = model(X)

        """
        print('type of outputs is:', type(outputs))
        print('shape of outputs is:', outputs.shape)
        """

        loss = criterion(outputs, y)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item()
        total_correct += (outputs.argmax(1) == y).sum().item()

    print('END')

    return total_loss / len(train_loader), total_correct / len(train_loader.dataset)


def train_urban_sound(cfg: HyperParameters):
    print("locate: train.py -> train_urban_sound()")
    print("START")

    # tensorboard追踪训练情况
    global writer

    # 加载训练数据集
    dataset = AudioDataset(cfg.train)
    dataloader = DataLoader(dataset, batch_size=cfg.train.batch_size, shuffle=cfg.train.is_shuffle)
    print(cfg.train.current_net)
    # 加载神经网络
    if cfg.train.current_net == 'CNN':
        model = AudioCNN(cfg).to(cfg.train.device)
    elif cfg.train.current_net == 'Improved_CNN':
        model = AudioImprovedCNN().to(cfg.train.device)
    elif cfg.train.current_net == 'ResNet':
        model = AudioResNet(cfg).to(cfg.train.device) 
    elif cfg.train.current_net == 'improved_ResNet':
        model = AudioImprovedResNet(cfg).to(cfg.train.device)
    elif cfg.train.current_net == 'SelfAttentionNet':
        model = SelfAttentionClassifier(cfg).to(cfg.train.device)
    elif cfg.train.current_net == 'tcn':
        model = TCNClassifier(cfg).to(cfg.train.device)
    elif cfg.train.current_net == 'ResNetSE':
        model = AudioResNetSE(cfg).to(cfg.train.device) 
    else:
        print('Error! No such Network!')
        exit(0)

    # 使用DataParallel包装模型，实现数据并行
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs for training!")
        model = nn.DataParallel(model)
    # 加载优化器
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.train.lr)

    # 加载损失函数
    criterion = nn.CrossEntropyLoss()       # 使用交叉熵损失
    
    # 进行训练
    for epoch in range(cfg.train.epochs):
        try:
            loss, acc = train_per_epoch(model, dataloader, optimizer, criterion, cfg.train.device)

            writer.add_scalar('Loss/train', loss, epoch)
            writer.add_scalar('Accuracy/train', acc, epoch)
            print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc*100:.2f}%")
        except KeyboardInterrupt:
            break

    print('END')

    return model

if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1"  # 使用GPU 0和1
    train_cfg = HyperParameters()

    writer = SummaryWriter('/mnt/data1/data_shared/UrbanSoundClassifier/logs/' + train_cfg.train.current_net)

    urban_sound_model = train_urban_sound(train_cfg)

    # 保存神经网络模型
    model_name = f"{train_cfg.train.current_net}_model_lr{train_cfg.train.lr}_batch{train_cfg.train.batch_size}_epoch{train_cfg.train.epochs}.pkl"
    print(f"Model is saved at: {train_cfg.train.model_save_dir + model_name}")
    torch.save(urban_sound_model, train_cfg.train.model_save_dir + model_name)