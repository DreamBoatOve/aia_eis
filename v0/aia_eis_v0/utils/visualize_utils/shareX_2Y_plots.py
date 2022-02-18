import matplotlib.pyplot as plt
import numpy as np

"""
Module Function
    两张图共享一个X轴，不同的Y轴在一侧
Refer:
    Creating adjacent subplots
    https://matplotlib.org/gallery/subplots_axes_and_figures/ganged_plots.html#sphx-glr-gallery-subplots-axes-and-figures-ganged-plots-py
"""
def tutorial_plot():
    t = np.arange(0.0, 2.0, 0.01)

    s1 = np.sin(2 * np.pi * t)
    s2 = np.exp(-t)
    s3 = s1 * s2

    fig, axs = plt.subplots(3, 1, sharex=True)
    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # Plot each graph, and manually set the y tick values
    axs[0].plot(t, s1)
    axs[0].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[0].set_ylim(-1, 1)

    axs[1].plot(t, s2)
    axs[1].set_yticks(np.arange(0.1, 1.0, 0.2))
    axs[1].set_ylim(0, 1)

    axs[2].plot(t, s3)
    axs[2].set_yticks(np.arange(-0.9, 1.0, 0.4))
    axs[2].set_ylim(-1, 1)

    plt.show()
# tutorial_plot()

def my_shareX_2Y_plot_4_AB_0(x_arr, y1_arr, y2_arr):
    """
    Function
        用于AB 结果的绘图
    :param
        w_type: str
            'w', raw/original weight
            'abs', the Abs(w)
            'positive', only keep the weight with positive value
    :return:
    """
    # 计算高中低三个区Avg的Avg
    avg_avg_high = np.mean(y1_arr[:int(y1_arr.shape[0] / 3)])
    avg_avg_mid = np.mean(y1_arr[int(y1_arr.shape[0] / 3) : int(2 * y1_arr.shape[0] / 3)])
    avg_avg_low = np.mean(y1_arr[int(2 * y1_arr.shape[0] / 3) : ])
    # 计算高中低三个区Avg的Var
    avg_var_high = np.mean(y2_arr[:int(y2_arr.shape[0] / 3)])
    avg_var_mid = np.mean(y2_arr[int(y2_arr.shape[0] / 3) : int(2 * y2_arr.shape[0] / 3)])
    avg_var_low = np.mean(y2_arr[int(2 * y2_arr.shape[0] / 3) : ])

    # 3d surface plot的长宽比大概是4：3，此处放大3倍，12：9
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 设置横纵坐标轴上的刻度字体大小
    # plt.tick_params(labelsize=30)

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 18}

    # Plot each graph, and manually set the y tick values
    axs[0].plot(x_arr, y1_arr)
    y1_min, y1_max = np.min(y1_arr), np.max(y1_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[0].set_yticks(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4))
    # axs[0].set_ylim(y1_min, y1_max)
    axs[0].set_yticks(np.around(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4), decimals=2))
    axs[0].set_ylim(y1_min-0.01, y1_max+0.01)

    # axs[0].set_xlabel('distance (m)')
    axs[0].set_ylabel('Average of Abs(Weights)', fontdict=x_y_title_font_setting)

    # 调整上图中（x）和y轴上的字体
    labels = axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # plt.text添加文字 设置字体颜色 https://blog.csdn.net/m0_38139098/article/details/104739475
    # axs[0].text(25, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    # axs[0].text(75, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    # axs[0].text(125, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")
    axs[0].text(10, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    axs[0].text(65, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    axs[0].text(120, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")

    axs[0].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[0].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Avg的Avg横线
    axs[0].plot(np.arange(53), np.ones(53)*avg_avg_high, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(53, 106), np.ones(53)*avg_avg_mid, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(106, 160), np.ones(54)*avg_avg_low, linestyle='-.', color='green', linewidth=3)

    # 设置上图中坐标轴刻度字体大小
    axs[0].tick_params(labelsize=14)

    axs[1].plot(x_arr, y2_arr)
    y2_min, y2_max = np.min(y2_arr), np.max(y2_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[1].set_yticks(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4))
    # axs[1].set_ylim(y2_min, y2_max)
    axs[1].set_yticks(np.around(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4), decimals=2))
    axs[1].set_ylim(y2_min-0.01, y2_max+0.01)

    axs[1].set_ylabel('Variance of Abs(Weights)', fontdict=x_y_title_font_setting)

    axs[1].set_xlabel('EIS Points Order', fontdict=x_y_title_font_setting)

    # axs[1].text(25, 0.9*(y2_max-y2_min)+y2_min, 'Low', size=18, family = "Times New Roman")
    # axs[1].text(75, 0.9*(y2_max-y2_min)+y2_min, 'Middle', size=18, family = "Times New Roman")
    # axs[1].text(125, 0.9*(y2_max-y2_min)+y2_min, 'High', size=18, family = "Times New Roman")

    axs[1].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[1].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Var的Avg横线
    axs[1].plot(np.arange(53), np.ones(53)*avg_var_high, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(53, 106), np.ones(53)*avg_var_mid, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(106, 160), np.ones(54)*avg_var_low, linestyle='-.', color='green', linewidth=3)

    # 设置下图中坐标轴刻度字体大小
    axs[1].tick_params(labelsize=14)

    # 调整下图中（x）和y轴上的字体
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()

def my_shareX_2Y_plot_4_AB_1(x_arr, y1_arr, y2_arr):
    """
    Function
        用于AB 结果的绘图
    :param
        w_type: str
            'w', raw/original weight
            'abs', the Abs(w)
            'positive', only keep the weight with positive value
    :return:
    """
    # 计算高中低三个区Avg的Avg
    avg_avg_high = np.mean(y1_arr[:int(y1_arr.shape[0] / 3)])
    avg_avg_mid = np.mean(y1_arr[int(y1_arr.shape[0] / 3) : int(2 * y1_arr.shape[0] / 3)])
    avg_avg_low = np.mean(y1_arr[int(2 * y1_arr.shape[0] / 3) : ])
    # 计算高中低三个区Avg的Var
    avg_var_high = np.mean(y2_arr[:int(y2_arr.shape[0] / 3)])
    avg_var_mid = np.mean(y2_arr[int(y2_arr.shape[0] / 3) : int(2 * y2_arr.shape[0] / 3)])
    avg_var_low = np.mean(y2_arr[int(2 * y2_arr.shape[0] / 3) : ])

    # 3d surface plot的长宽比大概是4：3，此处放大3倍，12：9
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 设置横纵坐标轴上的刻度字体大小
    # plt.tick_params(labelsize=30)

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    # x_y_title_font_setting = {'family': 'Times New Roman',
    #                           'weight': 'normal',
    #                           'size': 18}
    x_y_title_font_setting_0 = {'family': 'Times New Roman',
                              'weight': 'bold',
                              'size': 18}

    # Plot each graph, and manually set the y tick values
    axs[0].plot(x_arr, y1_arr)
    y1_min, y1_max = np.min(y1_arr), np.max(y1_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[0].set_yticks(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4))
    # axs[0].set_ylim(y1_min, y1_max)
    axs[0].set_yticks(np.around(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4), decimals=2))
    axs[0].set_ylim(y1_min-0.01, y1_max+0.01)

    # matplotlib 字母上加上划线
    # python da
    # axs[0].set_xlabel('distance (m)')
    # axs[0].set_ylabel('Average of Abs(Weights)', fontdict=x_y_title_font_setting_0)
    # axs[0].set_ylabel('|'+r"$\overline{w}$"+'|', fontdict=x_y_title_font_setting_0)

    # 调整上图中（x）和y轴上的字体
    labels = axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # plt.text添加文字 设置字体颜色 https://blog.csdn.net/m0_38139098/article/details/104739475
    # axs[0].text(25, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    # axs[0].text(75, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    # axs[0].text(125, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")
    # axs[0].text(10, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    # axs[0].text(65, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    # axs[0].text(120, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")
    # axs[0].text(10, 0.9*(y1_max-y1_min)+y1_min, 'High', size=18, family = "Times New Roman")
    # axs[0].text(65, 0.9*(y1_max-y1_min)+y1_min, 'Middle', size=18, family = "Times New Roman")
    # axs[0].text(120, 0.9*(y1_max-y1_min)+y1_min, 'Low', size=18, family = "Times New Roman")

    axs[0].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[0].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Avg的Avg横线
    axs[0].plot(np.arange(53), np.ones(53)*avg_avg_high, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(53, 106), np.ones(53)*avg_avg_mid, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(106, 160), np.ones(54)*avg_avg_low, linestyle='-.', color='green', linewidth=3)

    # 设置上图中坐标轴刻度字体大小
    axs[0].tick_params(labelsize=14)

    axs[1].plot(x_arr, y2_arr)
    y2_min, y2_max = np.min(y2_arr), np.max(y2_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[1].set_yticks(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4))
    # axs[1].set_ylim(y2_min, y2_max)
    axs[1].set_yticks(np.around(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4), decimals=2))
    axs[1].set_ylim(y2_min-0.01, y2_max+0.01)

    x_y_title_font_setting_1 = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 18}
    # axs[1].set_ylabel('Variance of Abs(Weights)', fontdict=x_y_title_font_setting_1)

    # axs[1].set_xlabel('EIS Points Order', fontdict=x_y_title_font_setting_1)

    # axs[1].text(25, 0.9*(y2_max-y2_min)+y2_min, 'Low', size=18, family = "Times New Roman")
    # axs[1].text(75, 0.9*(y2_max-y2_min)+y2_min, 'Middle', size=18, family = "Times New Roman")
    # axs[1].text(125, 0.9*(y2_max-y2_min)+y2_min, 'High', size=18, family = "Times New Roman")

    axs[1].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[1].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Var的Avg横线
    axs[1].plot(np.arange(53), np.ones(53)*avg_var_high, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(53, 106), np.ones(53)*avg_var_mid, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(106, 160), np.ones(54)*avg_var_low, linestyle='-.', color='green', linewidth=3)

    # 设置下图中坐标轴刻度字体大小
    axs[1].tick_params(labelsize=14)

    # 调整下图中（x）和y轴上的字体
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()

def my_shareX_2Y_plot_4_RF_0(x_arr, y1_arr, y2_arr, w_type):
    """
    Function
        用于RF Lime 结果的绘图
    :param
        w_type: str
            'w', raw/original weight
            'abs', the Abs(w)
            'positive', only keep the weight with positive value
    :return:
    """
    # 计算高中低三个区Avg的Avg
    avg_avg_high = np.mean(y1_arr[:int(y1_arr.shape[0] / 3)])
    avg_avg_mid = np.mean(y1_arr[int(y1_arr.shape[0] / 3) : int(2 * y1_arr.shape[0] / 3)])
    avg_avg_low = np.mean(y1_arr[int(2 * y1_arr.shape[0] / 3) : ])
    # 计算高中低三个区Avg的Var
    avg_var_high = np.mean(y2_arr[:int(y2_arr.shape[0] / 3)])
    avg_var_mid = np.mean(y2_arr[int(y2_arr.shape[0] / 3) : int(2 * y2_arr.shape[0] / 3)])
    avg_var_low = np.mean(y2_arr[int(2 * y2_arr.shape[0] / 3) : ])

    # 3d surface plot的长宽比大概是4：3，此处放大3倍，12：9
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 设置横纵坐标轴上的刻度字体大小
    # plt.tick_params(labelsize=30)

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 18}

    # Plot each graph, and manually set the y tick values
    axs[0].plot(x_arr, y1_arr)
    y1_min, y1_max = np.min(y1_arr), np.max(y1_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[0].set_yticks(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4))
    # axs[0].set_ylim(y1_min, y1_max)
    axs[0].set_yticks(np.around(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4), decimals=4))
    y1_range = y1_max - y1_min
    axs[0].set_ylim(y1_min - 0.1 * y1_range, y1_max + 0.1 * y1_range)
    # axs[0].set_xlabel('distance (m)')

    if w_type == 'abs':
        axs[0].set_ylabel('Average of Abs(Weights)', fontdict=x_y_title_font_setting)
    elif w_type == 'positive':
        axs[0].set_ylabel('Average of Positive Weights', fontdict=x_y_title_font_setting)
    elif w_type == 'w':
        axs[0].set_ylabel('Average of Weights', fontdict=x_y_title_font_setting)

    # 调整上图中（x）和y轴上的字体
    labels = axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # plt.text添加文字 设置字体颜色 https://blog.csdn.net/m0_38139098/article/details/104739475
    # axs[0].text(25, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    # axs[0].text(75, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    # axs[0].text(125, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")
    axs[0].text(10, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    axs[0].text(65, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    axs[0].text(120, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")

    axs[0].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[0].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Avg的Avg横线
    axs[0].plot(np.arange(53), np.ones(53)*avg_avg_high, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(53, 106), np.ones(53)*avg_avg_mid, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(106, 160), np.ones(54)*avg_avg_low, linestyle='-.', color='green', linewidth=3)

    # 设置上图中坐标轴刻度字体大小
    axs[0].tick_params(labelsize=14)

    axs[1].plot(x_arr, y2_arr)
    y2_min, y2_max = np.min(y2_arr), np.max(y2_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[1].set_yticks(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4))
    # axs[1].set_ylim(y2_min, y2_max)
    axs[1].set_yticks(np.around(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4), decimals=6))
    y2_range = y2_max - y2_min
    axs[1].set_ylim(y2_min - 0.1 * y2_range, y2_max + 0.1 * y2_range)

    if w_type == 'abs':
        axs[1].set_ylabel('Variance of Abs(Weights)', fontdict=x_y_title_font_setting)
    elif w_type == 'positive':
        axs[1].set_ylabel('Variance of Positive Weights', fontdict=x_y_title_font_setting)
    elif w_type == 'w':
        axs[1].set_ylabel('Variance of Weights', fontdict=x_y_title_font_setting)

    axs[1].set_xlabel('EIS Points Order', fontdict=x_y_title_font_setting)

    # axs[1].text(25, 0.9*(y2_max-y2_min)+y2_min, 'Low', size=18, family = "Times New Roman")
    # axs[1].text(75, 0.9*(y2_max-y2_min)+y2_min, 'Middle', size=18, family = "Times New Roman")
    # axs[1].text(125, 0.9*(y2_max-y2_min)+y2_min, 'High', size=18, family = "Times New Roman")

    axs[1].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[1].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Var的Avg横线
    axs[1].plot(np.arange(53), np.ones(53)*avg_var_high, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(53, 106), np.ones(53)*avg_var_mid, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(106, 160), np.ones(54)*avg_var_low, linestyle='-.', color='green', linewidth=3)

    # 设置下图中坐标轴刻度字体大小
    axs[1].tick_params(labelsize=14)

    # 调整下图中（x）和y轴上的字体
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()

def my_shareX_2Y_plot_4_RF_1(x_arr, y1_arr, y2_arr, w_type):
    """
    Function
        用于RF Lime 结果的绘图
    :param
        w_type: str
            'w', raw/original weight
            'abs', the Abs(w)
            'positive', only keep the weight with positive value
    :return:
    """
    # 计算高中低三个区Avg的Avg
    avg_avg_high = np.mean(y1_arr[:int(y1_arr.shape[0] / 3)])
    avg_avg_mid = np.mean(y1_arr[int(y1_arr.shape[0] / 3) : int(2 * y1_arr.shape[0] / 3)])
    avg_avg_low = np.mean(y1_arr[int(2 * y1_arr.shape[0] / 3) : ])
    # 计算高中低三个区Avg的Var
    avg_var_high = np.mean(y2_arr[:int(y2_arr.shape[0] / 3)])
    avg_var_mid = np.mean(y2_arr[int(y2_arr.shape[0] / 3) : int(2 * y2_arr.shape[0] / 3)])
    avg_var_low = np.mean(y2_arr[int(2 * y2_arr.shape[0] / 3) : ])

    # 3d surface plot的长宽比大概是4：3，此处放大3倍，12：9
    fig, axs = plt.subplots(2, 1, figsize=(12, 9), sharex=True)

    # 设置横纵坐标轴上的刻度字体大小
    # plt.tick_params(labelsize=30)

    # Remove horizontal space between axes
    fig.subplots_adjust(hspace=0)

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 18}

    # Plot each graph, and manually set the y tick values
    axs[0].plot(x_arr, y1_arr)
    y1_min, y1_max = np.min(y1_arr), np.max(y1_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[0].set_yticks(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4))
    # axs[0].set_ylim(y1_min, y1_max)
    axs[0].set_yticks(np.around(np.arange(y1_min, y1_max, (y1_max - y1_min) / 4), decimals=4))
    y1_range = y1_max - y1_min
    axs[0].set_ylim(y1_min - 0.1 * y1_range, y1_max + 0.1 * y1_range)
    # axs[0].set_xlabel('distance (m)')

    # if w_type == 'abs':
    #     axs[0].set_ylabel('Average of Abs(Weights)', fontdict=x_y_title_font_setting)
    # elif w_type == 'positive':
    #     axs[0].set_ylabel('Average of Positive Weights', fontdict=x_y_title_font_setting)
    # elif w_type == 'w':
    #     axs[0].set_ylabel('Average of Weights', fontdict=x_y_title_font_setting)

    # 调整上图中（x）和y轴上的字体
    labels = axs[0].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # plt.text添加文字 设置字体颜色 https://blog.csdn.net/m0_38139098/article/details/104739475
    # axs[0].text(25, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    # axs[0].text(75, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    # axs[0].text(125, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")
    # axs[0].text(10, 0.9*(y1_max-y1_min)+y1_min, 'High Frequency', size=18, family = "Times New Roman")
    # axs[0].text(65, 0.9*(y1_max-y1_min)+y1_min, 'Middle Frequency', size=18, family = "Times New Roman")
    # axs[0].text(120, 0.9*(y1_max-y1_min)+y1_min, 'Low Frequency', size=18, family = "Times New Roman")

    axs[0].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[0].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Avg的Avg横线
    axs[0].plot(np.arange(53), np.ones(53)*avg_avg_high, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(53, 106), np.ones(53)*avg_avg_mid, linestyle='-.', color='green', linewidth=3)
    axs[0].plot(np.arange(106, 160), np.ones(54)*avg_avg_low, linestyle='-.', color='green', linewidth=3)

    # 设置上图中坐标轴刻度字体大小
    axs[0].tick_params(labelsize=14)

    axs[1].plot(x_arr, y2_arr)
    y2_min, y2_max = np.min(y2_arr), np.max(y2_arr)

    # 这样设置太贴近上下边界，而且y轴数值有小数点后四位（太多，两位即可）
    # axs[1].set_yticks(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4))
    # axs[1].set_ylim(y2_min, y2_max)
    axs[1].set_yticks(np.around(np.arange(y2_min, y2_max, (y2_max - y2_min) / 4), decimals=6))
    y2_range = y2_max - y2_min
    axs[1].set_ylim(y2_min - 0.1 * y2_range, y2_max + 0.1 * y2_range)

    # if w_type == 'abs':
    #     axs[1].set_ylabel('Variance of Abs(Weights)', fontdict=x_y_title_font_setting)
    # elif w_type == 'positive':
    #     axs[1].set_ylabel('Variance of Positive Weights', fontdict=x_y_title_font_setting)
    # elif w_type == 'w':
    #     axs[1].set_ylabel('Variance of Weights', fontdict=x_y_title_font_setting)

    # axs[1].set_xlabel('EIS Points Order', fontdict=x_y_title_font_setting)

    # axs[1].text(25, 0.9*(y2_max-y2_min)+y2_min, 'Low', size=18, family = "Times New Roman")
    # axs[1].text(75, 0.9*(y2_max-y2_min)+y2_min, 'Middle', size=18, family = "Times New Roman")
    # axs[1].text(125, 0.9*(y2_max-y2_min)+y2_min, 'High', size=18, family = "Times New Roman")

    axs[1].axvline(53, linestyle='--', color='red')  # 红色虚线
    axs[1].axvline(106, linestyle='--', color='red')  # 红色虚线

    # plot 高中低三个区Var的Avg横线
    axs[1].plot(np.arange(53), np.ones(53)*avg_var_high, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(53, 106), np.ones(53)*avg_var_mid, linestyle='-.', color='green', linewidth=3)
    axs[1].plot(np.arange(106, 160), np.ones(54)*avg_var_low, linestyle='-.', color='green', linewidth=3)

    # 设置下图中坐标轴刻度字体大小
    axs[1].tick_params(labelsize=14)

    # 调整下图中（x）和y轴上的字体
    labels = axs[1].get_xticklabels() + axs[1].get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    plt.show()

def test_0():
    # 希腊字母
    plt.plot([0, 1, 2], [0, 1, 2], label=r"$\alpha$")

    # 给字母上 加 横线
    plt.plot([0, 1, 2], [0, 1, 2], label=r"$\overline{a}$")  # This is the offending line
    plt.plot([0, 1, 2], [0, 1, 2], label=r"$\overline{$\overline{a}$}$")  # This is the offending line

    # 加粗字体
    # plt.plot([0, 1, 2], [0, 1, 2], label=r"$\alpha$", fontdict={'weight': 'bold'})
    plt.legend(loc='best')
    plt.show()
# test_0()