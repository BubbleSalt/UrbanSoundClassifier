import numpy as np
import matplotlib.pyplot as plt


def urbansounds_confusion_matrix( y_real, y_pred):
    classes = ['air_conditioner', 'car_horn', 'children_playing', 'dog_bark', 'drilling', 'engine_idling', 'gun_shot', 'jackhammer', 'siren', 'street_music']

    # 获取类别数
    num_classes = len(classes)

    # 初始化混淆矩阵
    cm = np.zeros((num_classes, num_classes), dtype=int)

    # 计算混淆矩阵
    for true, pred in zip(y_real, y_pred):
        cm[true, pred] += 1
    # 计算百分比
    sum_per_real_class = np.sum(cm, axis=1)
    for i in range(cm.shape[0]):
            print(cm[i, :])
    
    percentage_cm = cm / sum_per_real_class[ : , None]
    print(percentage_cm)

    # 绘制混淆矩阵
    fig, ax = plt.subplots(figsize=(12, 9))
    cax = ax.matshow(cm, cmap='Blues')

    # 添加颜色条
    plt.colorbar(cax)

    # 设置x轴和y轴的标签
    ax.set_xticks(np.arange(num_classes))
    ax.set_yticks(np.arange(num_classes))
    ax.set_xticklabels(classes)
    plt.xticks (rotation=60)
    ax.set_yticklabels(classes)

    # 添加标签
    plt.xlabel('predicted_class')
    plt.ylabel('real_class')
    plt.title('confusion_matrix')

    # 在每个单元格中显示数字
    for (i, j), value in np.ndenumerate(cm):
        ax.text(j, i, f'{value}', ha='center', va='bottom', color='black')
        ax.text(j, i, format(percentage_cm[i, j], '.4f'), ha='center', va='top', color='black')

    plt.show()