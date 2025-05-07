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

        self.audio_features = []    # 初始化音频特征列表
        self.audio_labels = []      # 初始化音频标签列表

        self.counter = 0

        if self.cfg.is_origin_data:
            self.audio_preprocess()
        else:
            self.load_audio_info()

        print('END')

    # 返回数据集的大小
    def __len__(self):
        return len(self.audio_labels)

    # 根据索引获取音频文件并提取特征
    def __getitem__(self, idx):
        return torch.tensor(self.audio_features[idx], dtype=torch.float32), torch.tensor(self.audio_labels[idx], dtype=torch.long)

    def label_from_filename(self, filename: str):
        # formate of filename: 100032-3-0-0.wav -> 3
        return int(filename.split("-")[1])


    """对一条音频提取其特征"""
    def extract_feature(self, wav_file):
        
        # print('locate: class AudioDataset -> extract_feature()')
        # print('START')

        # 读取音频文件，获取音频时间序列和原始采样率
        X, sr = librosa.load(wav_file)

        # print('type of X is:', type(X))
        # print('shape of X is:', X.shape)
        # print(X)

        # 对长度不足的音频进行循环
        # 音频时间序列长度 = 采样率 × 秒数
        target_len = self.cfg.target_audio_len * sr
        print('target_len ==', target_len)
        first_pad = True
        while X.shape[0] < target_len:
            if first_pad:
                X = np.pad(X, (0, X.shape[0]), mode='constant')
                first_pad = False
            else:
                X = np.hstack((X, X))
                # print('After hstack...')
                # print('shape of X is:', X.shape)

        if X.shape[0] > target_len:
            X = X[ : target_len]

        # print('After all clips::::::')
        # print('shape of X is:', X.shape)

        # 对音频时间序列进行归一化(振幅上的数据增强)
        X_normal = librosa.util.normalize(X)

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=X_normal, sr=sr, n_mfcc=13, hop_length=1024)

        # 计算MFCC特征的一阶差分
        mfccs_delta = librosa.feature.delta(mfccs)

        # 计算色度特征，表示音频信号在不同音高上的能量分布
        chromagram = librosa.feature.chroma_stft(y=X_normal, sr=sr, hop_length=1024)

        # 计算FBank特征
        fbank_features = librosa.feature.melspectrogram(y=X_normal, sr=sr, n_mels=40, hop_length=1024)

        # 过零率
        zrc = librosa.feature.zero_crossing_rate(y=X_normal, hop_length=1024)

        # 频谱质心
        sc = librosa.feature.spectral_centroid(y=X_normal, hop_length=1024)

        # 将MFCC特征、MFCC一阶差分特征垂直堆叠，得到最终的特征矩阵
        final_features = np.vstack([mfccs, mfccs_delta, chromagram, zrc, sc])

        # print('END')

        return final_features


    # 节拍同步特征填充函数(时间长度归一化)
    # 目前没用上
    def pad_or_crop(self, feat):
        cur_len = feat.shape[1]
        if cur_len > self.cfg.target_beats_num:
            return feat[ : , : self.cfg.target_beats_num]
        
        elif cur_len < self.cfg.target_beats_num:
            return np.pad(feat, ((0, 0), (0, self.cfg.target_beats_num - cur_len)), mode='constant')

        return feat
    

    """保存预处理后的一组音频为.npy文件"""
    def save_audio_info(self):
        np.save(self.cfg.preproc_database_dir + '/audios_features.npy', arr=self.audio_features)
        np.save(self.cfg.preproc_database_dir + '/audios_labels.npy', arr=self.audio_labels)
        print(f"Preprocessed Audio saved at: {self.cfg.preproc_database_dir}")


    """读取预处理后的一组.npy音频文件"""
    def load_audio_info(self):
        self.audio_features = np.load(self.cfg.preproc_database_dir + '/audios_features.npy')
        self.audio_labels = np.load(self.cfg.preproc_database_dir + '/audios_labels.npy')
    

    """对一组原始音频进行预处理，并保存结果"""
    def audio_preprocess(self):
        audios = glob.glob(self.cfg.ori_database_dir + "/*.wav")    # 获取目录下所有的.wav音频文件
        for f in audios:
            feat = self.extract_feature(f)      # 对每条音频进行特征提取
            # feat = self.pad_or_crop(feat)       # 统一数据集中音频的节拍数量：0填充或裁剪
            label = self.label_from_filename(f)   # 获取此条音频的类别标签
            
            print('type of feat is:', type(feat))
            print('shape of feat is:', feat.shape)

            self.counter += 1
            print(f"已完成预处理的音频数量: {self.counter}")


            self.audio_features.append(feat)
            self.audio_labels.append(label)

            

        print('type of audio_features is:', type(self.audio_features))
        print('length of audio_features is:', len(self.audio_features))

        # np.newaxis: 增加一个维度，使其和神经网络的输入相匹配
        self.audio_features = np.stack(self.audio_features)[ : , np.newaxis, : , : ]  # (audio_nums, 1, feature_nums, time_frames)

        print('After stack...')
        print('type of audio_features is:', type(self.audio_features))
        print('shape of audio_features is:', self.audio_features.shape)

        self.audio_labels = np.array(self.audio_labels)

        # 保存预处理后得到的音频特征及标签
        self.save_audio_info()
