import librosa
import glob
import numpy as np


def label_from_filename(filename: str):
        # formate of filename: 100032-3-0-0.wav -> 3
        return int(filename.split("-")[1])


def extract_feature(wav_file):
        
        # print('locate: class AudioDataset -> extract_feature()')
        # print('START')

        # 读取音频文件，获取音频时间序列和原始采样率
        y, sr = librosa.load(wav_file)
        print(f"type of y is: {type(y)}")
        print(f"shape of y is: {y.shape}")
        print(y)

        # y_normalized = librosa.util.normalize(y)
        # 检查音频时间序列的范围
        # min_val = y_normalized.min()
        # max_val = y_normalized.max()

        # print(f"音频时间序列的最小值: {min_val}")
        # print(f"音频时间序列的最大值: {max_val}")

        print(f"sr == {sr}")

        # 提取MFCC特征
        # mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13)
        # mfccs = np.mean(librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).T, axis=0)

        # 对长度不足的音频进行零填充
        # 音频时间序列长度 = 采样率 × 秒数
        target_len = 4 * sr
        first_pad = True
        while y.shape[0] < target_len:
            if first_pad:
                y = np.pad(y, (0, y.shape[0]), mode='constant')
                first_pad = False
            else:
                y.append(y)
            if y.shape[0] > target_len:
                y = y[ : target_len]

        print('length of audios is:', y.shape[0] / sr)


        # 提取MFCC特征
        mfccs = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13, hop_length=1024)

        # 计算MFCC特征的一阶差分
        mfccs_delta = librosa.feature.delta(mfccs)


        # 查看MFCC特征的形状
        print("MFCC特征的形状:", mfccs.shape)
        print("MFCC一阶差分特征的形状:", mfccs_delta.shape)

        # 计算色度特征，表示音频信号在不同音高上的能量分布
        chromagram = librosa.feature.chroma_stft(y=y, sr=sr, hop_length=1024)

        # 查看色度特征的形状
        print("色度特征的形状:", chromagram.shape)

        # 提取FBank特征
        fbank_features = librosa.feature.melspectrogram(y=y, sr=sr, hop_length=1024)

        # 查看FBank特征的形状
        print("FBank特征的形状:", fbank_features.shape)

        # 过零率
        zrc = librosa.feature.zero_crossing_rate(y=y, hop_length=1024)
        print("zrc特征的形状:", zrc.shape)
        print(zrc)

        # 频谱质心
        sc = librosa.feature.spectral_centroid(y=y, hop_length=1024)
        print("sc特征的形状:", sc.shape)
        print(sc)

        # 将MFCC特征、MFCC一阶差分特征垂直堆叠，得到最终的特征矩阵
        final_features = np.vstack([mfccs, mfccs_delta])

        print("最终特征矩阵的类型:", type(final_features))
        print("最终特征矩阵的形状:", final_features.shape)


        label = label_from_filename(audio)
        print("这条音频的类型是:", label)
        print("这条音频类型的类型:", type(label))

        audio_info = [final_features, label]
        print(audio_info)
        print(len(audio_info))
        print(type(audio_info[0]))
        print(type(audio_info[1]))


def audio_explict_info(wav_file):
    pass



if __name__ == "__main__":
    audios = glob.glob('/mnt/data1/data_shared/UrbanSoundClassifier/resources/UrbanSound8K/audio/small_batch/*.wav')

    print('type of audios:', type(audios))
    print('length of audios:', len(audios))

    for audio in audios:
        print('type of audio:', type(audio))
        print(audio)
        extract_feature(audio)
        
        # print('shape of audio:', audio.shape)
