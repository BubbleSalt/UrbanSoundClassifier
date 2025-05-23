import sys , os
base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(base_dir)

import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.preprocess import AudioDataset
from nets.cnn import AudioCNN
from utils.hyper_parameters import HyperParameters

from utils.visualization import urbansounds_confusion_matrix

def load_model(model_name: str, model_dir: str, device: str):
    # model = torch.load(model_dir + model_name, map_location=torch.device(device), weights_only=False)
    model = torch.load(model_dir + model_name, weights_only=False)
    return model


def test(model, test_loader: DataLoader, criterion: nn.CrossEntropyLoss, device: str):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0
    counter = 0

    all_predicted_classes = []
    all_real_classes = []

    with torch.no_grad():  # No gradient calculation needed
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            print('type of features is:', type(features))
            print('shape of features is:', features.shape)
            print('type of labels is:', type(labels))
            print('shape of labels is:', labels.shape)

            counter += 1

            # 得到神经网络输出
            outputs = model(features)
            print('type of outputs is:', type(outputs))
            print('shape of outputs is:', outputs.shape)
            print('outputs = ', outputs)

            print(counter)

            # 计算总损失函数值
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # 输出预测结果标签
            _, predicted = torch.max(outputs, 1)
            print('type of predicted is:', type(predicted))
            print('shape of predicted is:', predicted.shape)
            # print(predicted)

            print('shape of labels is:', labels.shape)
            # print(labels)

            all_predicted_classes.extend(predicted.tolist())
            all_real_classes.extend(labels.tolist())

            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    # print('length of predicted is:', len(all_predicted_classes))    
    # print(all_predicted_classes)
    # print('length of labels is:', len(all_real_classes)) 
    # print(all_real_classes)

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy, all_predicted_classes, all_real_classes


def test_urban_sound(model_name: str, cfg: HyperParameters):
    # 加载训练好的神经网络模型
    model = load_model(model_name, cfg.test.model_load_path, cfg.test.device)

    # 加载测试数据集
    test_dataset = AudioDataset(cfg.test)
    test_dataloader = DataLoader(test_dataset, batch_size=cfg.test.batch_size, shuffle=cfg.test.is_shuffle)

    # 加载损失函数
    criterion = nn.CrossEntropyLoss()

    # 进行测试
    test_loss, test_acc, real_classes, predicted_classes = test(model, test_dataloader, criterion, cfg.test.device)
    print(f"Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%")

    urbansounds_confusion_matrix(real_classes, predicted_classes)


if __name__ == "__main__":
    test_cfg = HyperParameters()
    model_name = test_cfg.test.model_name
    print('Loaded model name:',model_name)
    test_urban_sound(model_name, test_cfg)