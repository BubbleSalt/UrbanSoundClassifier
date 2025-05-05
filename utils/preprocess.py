import numpy as np
import librosa
import torch
from torch.utils.data import Dataset

import glob
from typing import Union

from utils.hyper_parameters import HyperParameters

class AudioDataset(Dataset):
    """
        音频数据集以及预处理函数
    """
    def __init__(self, cfg: Union[HyperParameters.train, HyperParameters.test]):
        print('locate: class AudioDataset -> __init__()')
        print('START')

        self.cfg = cfg
        self.database_dir = self.cfg.database_dir

        self.X = []
        self.y = []

        audios = glob.glob(self.cfg.database_dir + "*.wav")     # 获取目录下所有的.wav音频文件
        for f in audios:
            feat = self.extract_feature(f)      # 对每条音频进行特征提取
            feat = self.pad_or_crop(feat)       # 统一数据集中音频的节拍数量：0填充或裁剪
            self.X.append(feat)

            audio_label = self.label_from_filename(f)     # 获取此条音频的类别标签
            self.y.append(audio_label)

        self.X = np.stack(self.X)[:, np.newaxis, :, :]  # (B, 1, 38, T)
        self.y = np.array(self.y)

        print('END')

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        return torch.tensor(self.X[idx], dtype=torch.float32), torch.tensor(self.y[idx], dtype=torch.long)

    def label_from_filename(self, filename: str):
        # formate of filename: 100032-3-0-0.wav -> 3
        return int(filename.split("-")[1])


    def extract_feature(self, wav_file):
        # print('locate: class AudioDataset -> extract_feature()')
        # print('START')

        # 读取音频文件，获取其原始采样率
        y, sr = librosa.load(wav_file, sr=None)

        # 对长度不足的音频进行零填充
        # 音频长度 = 采样率 × 秒数
        target_len = sr * self.cfg.target_audio_len
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')
        # print(f"length of audio is: {(len(y) / sr)}")

        # 将音频信号y分离为harmonic和percussive两个波形
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # 对percussive信号进行节拍跟踪，返回估计的节拍速度和节拍帧索引
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # 进行短时傅里叶变换，获取音频的频域特征
        # stft = librosa.stft(y, n_fft=256)

        # 计算梅尔频率倒谱系数(MFCC)及其一阶差分
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=256, n_mfcc=13, n_fft=256)
        mfcc_delta = librosa.feature.delta(mfcc)

        # 使用均值聚合方法将MFCC及其一阶差分同步到节拍帧上
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

        # 从harmonic信号中计算色度特征，表示音频信号在不同音高上的能量分布
        # chromagram = librosa.feature.chroma_cqt(y=y_harmonic, sr=sr)

        chromagram = librosa.feature.chroma_stft(   y=y_harmonic,
                                                    sr=sr,
                                                    n_fft=256,          # 配合前面的mfcc
                                                    hop_length=256  )

        # 使用中位数聚合方法将色度特征同步到节拍帧上
        beat_chroma = librosa.util.sync(chromagram,
                                        beat_frames,
                                        aggregate=np.median)

        # 将色度特征、MFCC、MFCC差分特征垂直堆叠，得到最终的特征矩阵
        final_features = np.vstack([beat_chroma, beat_mfcc_delta])

        # print(f"{wav_file}: ({len(final_features)},{len(final_features[0])})")
        # print('END')

        return final_features   # shape: (38, beats)


    # 节拍同步特征填充函数(时间长度归一化)
    def pad_or_crop(self, feat):
        cur_len = feat.shape[1]
        if cur_len > self.cfg.target_beats_num:
            return feat[ : , : self.cfg.target_beats_num]
        
        elif cur_len < self.cfg.target_beats_num:
            return np.pad(feat, ((0, 0), (0, self.cfg.target_beats_num - cur_len)), mode='constant')

        return feat