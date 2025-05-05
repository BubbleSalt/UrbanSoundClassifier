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

        pool_size = (1, 1)          # 自适应池化层的输出尺寸

        linear_in_features = 32     # 
        linear_out_features = 10    # 数据集中的标签数量


    class train:
        device = 'cpu'
        database_dir = 'D:\\UrbanSound8K\\audio\\fold4\\'       # 绝对路径
        save_dir = 'C:\\Users\\THP\\Desktop\\人工智能与机器学习方法\\UrbanSoundClassifier\\models\\'

        target_audio_len = 4        # 统一音频时长
        target_beats_num = 32       # 统一音频节拍数
        is_shuffle = True
        

        batch_size = 10
        lr = 0.005
        epochs = 500


    class test:
        device = 'cpu'
        database_dir = 'D:\\UrbanSound8K\\audio\\fold6\\'       # 绝对路径
        # database_dir = 'D:\\UrbanSound8K\\audio\\small_batch\\'       # 绝对路径
        load_path = 'C:\\Users\\THP\\Desktop\\人工智能与机器学习方法\\UrbanSoundClassifier\\models\\'

        target_audio_len = 4        # 
        target_beats_num = 32       # 
        is_shuffle = True

        batch_size = 10