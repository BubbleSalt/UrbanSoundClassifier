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
        """
        # 读取音频文件，获取音频时间序列和原始采样率
        y, sr = librosa.load(wav_file, sr=None)

        
        # 对长度不足的音频进行零填充
        # 音频长度 = 采样率 × 秒数
        target_len = sr * self.cfg.target_audio_len
        if len(y) < target_len:
            y = np.pad(y, (0, target_len - len(y)), mode='constant')

        # 将音频信号y分离为harmonic和percussive两个波形
        y_harmonic, y_percussive = librosa.effects.hpss(y)

        # 对percussive信号进行节拍跟踪，返回估计的节拍速度和节拍帧索引
        tempo, beat_frames = librosa.beat.beat_track(y=y_percussive, sr=sr)

        # 计算梅尔频率倒谱系数(MFCC)及其一阶差分
        mfcc = librosa.feature.mfcc(y=y, sr=sr, hop_length=256, n_mfcc=13, n_fft=256)
        mfcc_delta = librosa.feature.delta(mfcc)

        # 使用均值聚合方法将MFCC及其一阶差分同步到节拍帧上
        beat_mfcc_delta = librosa.util.sync(np.vstack([mfcc, mfcc_delta]), beat_frames)

        # 从harmonic信号中计算色度特征，表示音频信号在不同音高上的能量分布
        chromagram = librosa.feature.chroma_stft(   y=y_harmonic,
                                                    sr=sr,
                                                    n_fft=256,          # 配合前面的mfcc
                                                    hop_length=256  )

        # 使用中位数聚合方法将色度特征同步到节拍帧上
        beat_chroma = librosa.util.sync(chromagram,
                                        beat_frames,
                                        aggregate=np.median)

        # 提取FBank特征
        fbank_features = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=40, hop_length=256, n_fft=256, window='hamming')
        # 取对数操作
        fbank_features = librosa.power_to_db(fbank_features, ref=np.max)
        beat_fbank = librosa.util.sync(fbank_features, beat_frames, aggregate=np.median)

        # 将色度特征、MFCC、MFCC差分特征、FBank特征垂直堆叠，得到最终的特征矩阵
        final_features = np.vstack([beat_chroma, beat_mfcc_delta, beat_fbank])

        # print(f"{wav_file}: ({len(final_features)},{len(final_features[0])})")
        # print('END')

        return final_features   # shape: (38, beats)
        """

        # 读取音频文件，获取音频时间序列和原始采样率
        X, sr = librosa.load(wav_file)

        # print('type of X is:', type(X))
        # print('shape of X is:', X.shape)
        # print(X)

        # 对长度不足的音频进行零填充和循环
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

        print('After all clips::::::')
        print('shape of X is:', X.shape)

        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=X, sr=sr, n_mfcc=13)

        # 计算MFCC特征的一阶差分
        mfccs_delta = librosa.feature.delta(mfccs)

        # 将MFCC特征、MFCC一阶差分特征垂直堆叠，得到最终的特征矩阵
        final_features = np.vstack([mfccs, mfccs_delta])

        # print('END')

        return final_features       # shape: (26, 173)


    # 节拍同步特征填充函数(时间长度归一化)
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
            
            # print('type of feat is:', type(feat))
            print('shape of feat is:', feat.shape)

            self.counter += 1
            print(f"已完成预处理的音频数量: {self.counter}")

            if feat.shape != (26, 173):
                print("ERROR!!!")
                break

            self.audio_features.append(feat)
            self.audio_labels.append(label)

            

        print('type of audio_features is:', type(self.audio_features))
        print('length of audio_features is:', len(self.audio_features))

        # np.newaxis: 增加一个维度，使其和神经网络的输入相匹配
        self.audio_features = np.stack(self.audio_features)[ : , np.newaxis, : , : ]  # (audio_nums, 1, feature_nums, time_frames)

        # print('After stack...')
        # print('type of audio_features is:', type(self.audio_features))
        # print('shape of audio_features is:', self.audio_features.shape)

        self.audio_labels = np.array(self.audio_labels)

        # 保存预处理后得到的音频特征及标签
        self.save_audio_info()
