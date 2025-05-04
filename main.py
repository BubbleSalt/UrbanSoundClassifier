import numpy as np
import librosa
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import os
import glob

# 示例数据集（文件名列表）
train_dist = '.\\UrbanSound8K\\audio\\fold1' 
# train_dist = './testfolder'
test_dist = './UrbanSound8K/audio/fold2'

def extract_feature(wav_path):
    y, sr = librosa.load(wav_path, sr=None)

    # Separate harmonics and percussives into two waveforms
    y_harmonic, y_percussive = librosa.effects.hpss(y)
    # Beat track on the percussive signal
    tempo, beat_frames = librosa.beat.beat_track(y=y_percussive,
                                                sr=sr)
    stft = librosa.stft(y, n_fft=256)
    # Compute MFCC features from the raw signal
    mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=256, n_mfcc=13, n_fft=256)

    # And the first-order differences (delta features)
    mfcc_delta = librosa.feature.delta(mfcc)

    # Stack and synchronize between beat events
    # This time, we'll use the mean value (default) instead of median
    beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]),
                                    beat_frames)

    # Compute chroma features from the harmonic signal
    chromagram = librosa.feature.chroma_cqt(y=y_harmonic,
                                            sr=sr)

    # Aggregate chroma features between beat events
    # We'll use the median value of each feature between beat frames
    beat_chroma = librosa.util.sync(chromagram,
                                    beat_frames,
                                    aggregate=np.median)

    final_features = np.vstack([beat_chroma, beat_mfcc_delta])
    print(f"{wav_path}: ({len(final_features)},{len(final_features[0])})")
    # Finally, stack all beat-synchronous features together
    return final_features # shape: (38, T)

# 节拍同步特征填充函数（时间长度归一化）
def pad_or_crop(feat, target_len=32):
    cur_len = feat.shape[1]
    if cur_len > target_len:
        return feat[:, :target_len]
    elif cur_len < target_len:
        return np.pad(feat, ((0, 0), (0, target_len - cur_len)), mode='constant')
    return feat

# 构造数据集
class AudioDataset(Dataset):
    def __init__(self, file_list, label_func, target_len=32):
        self.X = []
        self.y = []
        for f in file_list:
            feat = extract_feature(f)
            feat = pad_or_crop(feat, target_len)
            self.X.append(feat)
            self.y.append(label_func(f))  # 文件名 -> 类别数字
        self.X = np.stack(self.X)[:, np.newaxis, :, :]  # (B, 1, 38, T)
        self.y = np.array(self.y)

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

# 文件名转标签
def label_from_filename(filename: str):
    return int(filename.split("-")[1]) # 100032-3-0-0.wav -> 3

# CNN 模型
class AudioCNN(nn.Module):
    def __init__(self, num_classes):
        super(AudioCNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.bn1 = nn.BatchNorm2d(16)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm2d(32)
        self.pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        return self.fc(x)

# 模型训练函数
def train(model, dataloader, optimizer, criterion, device):
    model.train()
    total_loss, total_correct = 0, 0
    for X, y in dataloader:
        X, y = X.to(device), y.to(device)
        optimizer.zero_grad()
        out = model(X)
        loss = criterion(out, y)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        total_correct += (out.argmax(1) == y).sum().item()
    return total_loss / len(dataloader), total_correct / len(dataloader.dataset)

def test(model, test_loader, criterion, device):
    model.eval()  # Set model to evaluation mode
    correct = 0
    total = 0
    running_loss = 0.0

    with torch.no_grad():  # No gradient calculation needed
        for features, labels in test_loader:
            features, labels = features.to(device), labels.to(device)

            # Forward pass
            outputs = model(features)

            # Compute loss
            loss = criterion(outputs, labels)
            running_loss += loss.item()

            # Get predicted labels
            _, predicted = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

    avg_loss = running_loss / len(test_loader)
    accuracy = correct / total
    return avg_loss, accuracy

# 主程序
if __name__ == "__main__":
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    num_classes = 10 
    batch_size = 2

    train_audios = glob.glob(train_dist+"\\*.wav")
    test_audios = glob.glob(test_dist+"\\*.wav")

    dataset = AudioDataset(train_audios, label_from_filename, target_len=32)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = AudioCNN(num_classes=num_classes).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    criterion = nn.CrossEntropyLoss()

    epochs = 10

    for epoch in range(epochs):
        loss, acc = train(model, dataloader, optimizer, criterion, device)
        print(f"Epoch {epoch+1}: Loss = {loss:.4f}, Accuracy = {acc*100:.2f}%")

    test_dataset = AudioDataset(test_audios, label_from_filename, target_len=32)
    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    test_loss, test_acc = test(model, test_dataloader, criterion, device)
    print(f"Epoch {epoch+1}: Test Loss = {test_loss:.4f}, Test Accuracy = {test_acc*100:.2f}%")

