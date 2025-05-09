class HyperParameters:
    """
        超参数集合
    """
    class cnn:
        conv1_in_channels = 1         # 输入通道数 
        conv1_out_channels = 16       # 输出通道数(使用卷积核的个数)
        conv1_kernel_size = 3         # 卷积核尺寸
        conv1_padding = 1             # 填充方式

        conv2_in_channels = 16
        conv2_out_channels = 32
        conv2_kernel_size = 3
        conv2_padding = 1

        pool_size = (5, 5)          # 自适应池化层的输出尺寸

        linear_in_features = 32 * 5 * 5
        linear_out_features = 10    # 数据集中的标签数量


    class improvedCnn:
        pass


    class ResNet:
        weights = None

        conv_in_channels = 1
        conv_out_channels = 64
        stride = 2
        kernel_size = 7
        padding = 3
        bias = False
        linear_out_features = 10


    class SelfAttentionNet:
        pass


    class TemporalBlock:
        pass


    class train:
        mode = 'train'
        device = 'cuda'

        is_origin_data = False
        ori_database_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/resources/UrbanSound8K/audio/train'
        # ori_database_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/resources/UrbanSound8K/audio/small_batch'
        preproc_database_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/resources/preprocessed_audios/train'

        target_audio_len = 4        # 统一音频时长[s]
        target_beats_num = 12       # 统一音频节拍数
        is_shuffle = True
        
        current_net = 'CNN'
        # current_net = 'Improved_CNN'
        # current_net = 'ResNet'
        # current_net = 'improved_ResNet'
        # current_net = 'SelfAttentionNet'
        # current_net = 'tcn'

        batch_size = 128
        lr = 0.001
        epochs = 500

        model_save_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/models/'


    class test:
        mode = 'test'
        device = 'cuda'

        is_origin_data = False
        ori_database_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/resources/UrbanSound8K/audio/test'
        # ori_database_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/resources/UrbanSound8K/audio/small_batch'
        preproc_database_dir = '/mnt/data1/data_shared/UrbanSoundClassifier/resources/preprocessed_audios/test'

        model_load_path = '/mnt/data1/data_shared/UrbanSoundClassifier/models/'
        model_name = 'CNN_model_lr0.001_batch128_epoch500.pkl'

        target_audio_len = 4
        target_beats_num = 12
        is_shuffle = True

        batch_size = 128

        