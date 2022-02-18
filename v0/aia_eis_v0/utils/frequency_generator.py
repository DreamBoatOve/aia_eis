import math
"""
EIS模拟实验的频率生成器
    频率由高到低，和实验顺序保持一致
    对数坐标尺度上等间距取点
    生成频率和角度频率两个数据列表
"""
def fre_generator(f_start, f_end, pts_decade):
    """
    :param
        f_start:
            int
            High frequency
            Ex: 7, == 1e7
        f_end:
            int
            Low frequency
            Ex: 7, == 1e-7
        pts_decade:
    :return:
    """
    points_num = (f_start - f_end) * pts_decade
    fre_list = [10 ** (f_start - i * (f_start - f_end) / points_num) for i in range(points_num)]
    w_list = [2 * math.pi * f for f in fre_list]
    return fre_list, w_list