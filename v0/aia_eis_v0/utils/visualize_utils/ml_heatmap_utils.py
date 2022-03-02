import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import seaborn as sns
import pandas as pd

"""
方块的颜色统一
右侧颜色标尺的尺度统一，均从一个最小值~最大值
【手动的】在图片上标出效果最好的3~5个模型的accuracy+kappa

Seaborn heatmap
    Parameters
        vmin, vmax : floats, optional
        用来设置颜色标尺的上下限
    Refers
        seaborn.heatmap参数介绍
            https://blog.csdn.net/m0_38103546/article/details/79935671
        Seaborn Heatmap Using Sns.Heatmap() | Python Seaborn Tutorial
            https://indianaiproduction.com/seaborn-heatmap/#18-how-to-hide-color-bar-using-snsheatmap-cbar-parameter
        Python可视化matplotlib&seborn14-热图heatmap
            https://mp.weixin.qq.com/s?__biz=MzUwOTg0MjczNw==&mid=2247485375&idx=1&sn=be29296af9b63a2b6379dbe89700386d&chksm=f90d43e1ce7acaf7f1507762d40178023106afd6910e2ae3f3f7bb9f14db5c289cfbf0a81922&token=1588396102&lang=zh_CN#rd
"""
# 右侧颜色标尺的尺度统一，均从一个最小值~最大值
# 目前来看只有RF的AK在1.01左右，故vmax = 1.1
vmin, vmax = 0.0, 1.1

def lrc_heatmap(txt_file_path, axis_title_list):
    """
    unique requirement:
        x/y轴变量： 字体 Time newroman， 字号
        x/y轴标题： 字体 Time newroman， 字号
    :param
        txt_file_path:
            grid search average result in a txt file
        axis_title_list:
            list[str]
            [y_axis_title, x_axis_title, z_axis_title]
    :return:
        show heatmap plot
    """
    global vmin, vmax
    data_list = []
    with open(txt_file_path, 'r') as file:
        for line in file.readlines():
            data_list.append([float(a) for a in line.strip().split(',')])

    sns.set()
    df = pd.DataFrame(data_list, columns=axis_title_list)

    # 原始数据中的Iteration一列都是浮点数，显示难看，在此转为整数类型
    df['Iteration'] = df['Iteration'].astype('int')

    df = df.pivot(axis_title_list[0], axis_title_list[1], axis_title_list[2])
    f, ax = plt.subplots(figsize=(9, 6))

    # set title of plot. no need anymore
    # ax.set_title(axis_title_list[2])

    # annot = True: show value; fmt='.5f': keep 5 number after '.'.
    # cmap: color. more color: https://blog.csdn.net/linzhjbtx/article/details/85319554
    # official configuration of heatmap: http://seaborn.pydata.org/generated/seaborn.heatmap.html#seaborn.heatmap

    # ------------------------------- 将Learning Rate 1e-5 ~ 1e3 显示为科学计数法 -------------------------------
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))
    xtick = [u"${}$".format(tick.format_data(10**x)) for x in range(-5, 4)]
    hm = sns.heatmap(df, vmin=vmin, vmax=vmax, annot=False, fmt='.5f', cmap="RdYlGn_r", linewidths=.5,
                ax=ax, square=True, xticklabels=xtick)
    # ------------------------------- 将Learning Rate 1e-5 ~ 1e3 显示为科学计数法 -------------------------------

    # Setting of Colorbar
    cb = hm.figure.colorbar(hm.collections[0])
    # 设置colorbar刻度字体大小
    cb.ax.tick_params(labelsize=20)

    # 设置横纵坐标轴上的刻度字体大小
    plt.tick_params(labelsize=20)
    # 设置横纵坐标轴上的刻度字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """
    将Y轴的刻度上的字体垂直摆放
        refer https://blog.csdn.net/songrenqing/article/details/78927283
    """
    [tick.set_rotation(0) for tick in ax.get_yticklabels()]

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 26}
    plt.xlabel(axis_title_list[1], x_y_title_font_setting)
    plt.ylabel(axis_title_list[0], x_y_title_font_setting)

    plt.show()
# ----------------- Draw Heatmap for LRC-OvO-Linear (Iteration and C) -----------------
# txt_filename = '../../ml_sl/logistic/ovo_res/linear/plots/2020_04_22_lrc_ovo_linear_avg_res_0.txt'
# lrc_heatmap(txt_file_path = txt_filename,
#             axis_title_list = ['Iteration', 'Learning Rate', 'AK'])
# ----------------- Draw Heatmap for LRC-OvO-Linear (Iteration and C) -----------------

# ----------------- Draw Heatmap for LRC-OvR-Linear (Iteration and C) -----------------
txt_filename = '../../ml_sl/logistic/ovr_res/linear/plots/2020_05_07_lrc_ovr_linear_avg_res.txt'
lrc_heatmap(txt_file_path = txt_filename,
            axis_title_list = ['Iteration', 'Learning Rate', 'AK'])
# ----------------- Draw Heatmap for LRC-OvR-Linear (Iteration and C) -----------------

def knn_heatmap(txt_file_path, axis_title_list):
    """
    unique requirement:
        x/y轴变量： 字体 Time new roman， 字号
        x/y轴标题： 字体 Time new roman， 字号
    :param
        txt_file_path:
            grid search average result in a txt file
        axis_title_list:
            list[str]
            [y_axis_title, x_axis_title, z_axis_title]
    :return:
        show heatmap plot
    """
    global vmin, vmax
    data_list = []
    with open(txt_file_path, 'r') as file:
        # the first row is column name, should be skipped
        for line in file.readlines()[1:]:
            line_str_list = line.strip().split(',')
            distance_mode_str = line_str_list[0]
            K = int(line_str_list[1])
            acc_kappa = float(line_str_list[2])
            data_list.append([distance_mode_str, K, acc_kappa])

    sns.set()
    df = pd.DataFrame(data_list, columns=axis_title_list)
    df = df.pivot(axis_title_list[0], axis_title_list[1], axis_title_list[2])
    f, ax = plt.subplots(figsize=(16, 9))

    # ax.set_title(axis_title_list[2]) # 设置图的标题，在图片顶部显示，在此处已不需要
    # sns.heatmap(data=df, vmin=vmin, vmax=vmax, annot=True, fmt='.5f', cmap="RdYlGn_r", linewidths=.5, ax=ax, square=True)

    # annot True/False 方块中显示字体与否
    hm = sns.heatmap(data=df, vmin=vmin, vmax=vmax, annot=False, fmt='.5f', cbar=False, cmap="RdYlGn_r", linewidths=.5, ax=ax, square=True)

    # Setting of Colorbar
    cb = hm.figure.colorbar(hm.collections[0])
    # 设置colorbar刻度字体大小
    cb.ax.tick_params(labelsize=18)

    cb_font_dict = {'family': 'Times New Roman', 'color': 'darkred', 'weight' : 'normal', 'size': 56}
    # cb.set_label('', fontdict=cb_font_dict)  # 设置colorbar的标签字体及其大小

    # 设置横纵坐标轴上的刻度字体大小
    plt.tick_params(labelsize=18)
    # 设置横纵坐标轴上的刻度字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """
    将Y轴的刻度上的字体垂直摆放
        refer https://blog.csdn.net/songrenqing/article/details/78927283
    """
    [tick.set_rotation(0) for tick in ax.get_yticklabels()]

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 22}
    plt.xlabel(axis_title_list[1], x_y_title_font_setting)
    plt.ylabel(axis_title_list[0], x_y_title_font_setting)

    plt.show()
# ----------------- Draw Heatmap for KNN (Distance Measure and K) -----------------
# knn_heatmap(txt_file_path='../../ml_sl/knn/plots/2020_11_23_KNN_GS_res.txt',
#             axis_title_list=['Distance Measure', 'Nearest Neighbors Number (K)', 'AK'])
# ----------------- Draw Heatmap for KNN (Distance Measure and K) -----------------

def svm_heatmap(txt_file_path, axis_title_list, para0_index=0, para1_index=1, target_index=-1):
    """
    Function:
        SVM OvO Linear has two parameters, max_iter and C, to be adjusted.
    :param
        txt_file_path:
            grid search average result in a txt file
        axis_title_list:
            list[str]
            [y_axis_title, x_axis_title, z_axis_title]
    Note:
        Heatmap 的colorbar的尺寸只能方形的图像匹配，对矩形的图像会过长而很难看，
        这时要一边调整‘f, ax = plt.subplots(figsize=(16, 12))’中的figsize，还有手动调整 plot的控制面板
    :return:
    """
    global vmin, vmax
    data_list = []
    with open(txt_file_path, 'r') as file:
        # the first row is column name, should be skipped
        for line in file.readlines():
            line_str_list = line.strip().split(',')
            iteration = float(line_str_list[para0_index])
            C = float(line_str_list[para1_index])
            acc_kappa = float(line_str_list[target_index])
            data_list.append([iteration, C, acc_kappa])

    sns.set()
    df = pd.DataFrame(data_list, columns=axis_title_list)

    # 原始数据中的Iteration一列都是浮点数，显示难看，在此转为整数类型
    df['Iteration'] = df['Iteration'].astype('int')

    df = df.pivot(axis_title_list[0], axis_title_list[1], axis_title_list[2])
    f, ax = plt.subplots(figsize=(16, 12))

    # No title
    # ax.set_title(axis_title_list[2])

    # ------------------------------- 将Learning Rate 1e-5 ~ 1e5 显示为科学计数法 -------------------------------
    tick = ticker.ScalarFormatter(useOffset=False, useMathText=True)
    tick.set_powerlimits((0, 0))
    xtick = [u"${}$".format(tick.format_data(10 ** x)) for x in range(-5, 6)]
    hm = sns.heatmap(df, vmin=vmin, vmax=vmax, annot=False, fmt='.5f', cmap="RdYlGn_r", linewidths=.5,
                ax=ax, square=True, xticklabels=xtick)
    # ------------------------------- 将Learning Rate 1e-5 ~ 1e5 显示为科学计数法 -------------------------------

    # Setting of Colorbar
    cb = hm.figure.colorbar(hm.collections[0])
    # 设置colorbar刻度字体大小
    cb.ax.tick_params(labelsize=18)

    # 设置横纵坐标轴上的刻度字体大小
    plt.tick_params(labelsize=18)
    # 设置横纵坐标轴上的刻度字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    """
    将Y轴的刻度上的字体垂直摆放
        refer https://blog.csdn.net/songrenqing/article/details/78927283
    """
    [tick.set_rotation(0) for tick in ax.get_yticklabels()]

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 22}
    plt.xlabel(axis_title_list[1], x_y_title_font_setting)
    plt.ylabel(axis_title_list[0], x_y_title_font_setting)

    plt.show()
# ----------------- Draw Heatmap for SVM-OvO-Linear (max_iter and C) -----------------
# svm_heatmap(txt_file_path='../../ml_sl/svm/ovo_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_24_svm_ovo_linear_gs_avg_res.txt',
#             axis_title_list=['Iteration', 'C', 'AK'])
# ----------------- Draw Heatmap for SVM-OvO-Linear (max_iter and C) -----------------

# ----------------- Draw Heatmap for SVM-OvR-Linear (max_iter and C) -----------------
# svm_heatmap(txt_file_path='../../ml_sl/svm/ovr_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_24_svm_ovr_linear_gs_avg_res.txt',
#             axis_title_list=['Iteration', 'C', 'AK'])
# ----------------- Draw Heatmap for SVM-OvR-Linear (max_iter and C) -----------------

# ----------------- Draw Heatmap for SVM-OvO-Poly (max_iter and C) -----------------
# svm_heatmap(txt_file_path='../../ml_sl/svm/ovo_txt_res/trained_on_tr_tested_on_vali/avg_res_and_plots/2020_06_25_svm_ovo_poly_gs_avg_res.txt',
#             axis_title_list=['Iteration', 'C', 'AK'])
# ----------------- Draw Heatmap for SVM-OvO-Poly (max_iter and C) -----------------