import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

from utils.post_process.ml_post_process import single_para_boxplot_data_wrapper
"""
In some MLs, there is only one variable to be adjusted in its model
To get average performance of the model with a certain variable, the experiment is repeated for 10 times
Draw boxplot to visualize Performance Vs variable

Requirement:
    1-尝试用直线连接相邻箱型图上的中点或（均值点），能直接画就直接画，不能直接画，就再加一条直线
    2-统一纵轴的范围
    3-将LRC-OvR/OvO-Stable两组结果画到一张图上

Main refers:
    Boxplot Demo 
        https://matplotlib.org/3.2.1/gallery/pyplots/boxplot_demo_pyplot.html#sphx-glr-gallery-pyplots-boxplot-demo-pyplot-py
    Matplotlib - 箱线图、箱型图 boxplot () 所有用法详解
        https://blog.csdn.net/weixin_40683253/article/details/87857194
"""

# 右侧颜色标尺的尺度统一，均从一个最小值~最大值
# 目前来看只有RF的AK在1.01左右，故vmax=1.1
vmin, vmax = 0.0, 1.1

def lrc_boxplot(table_start_row, table_start_col, para_margin, sheet_name, excel_abs_path, plot_para_dict=None):
    """
    function:
        LRC在使用stable alpha 的时候只有一个参数（梯度下降的迭代次数）可调，
        此函数用来画 max_iteration Vs AK 的boxplot
        此函数应该会被两处用到，分别为LRC-OVO-Stable-Alpha 和 LRC-OVR-Stable-Alpha
    :param:
        table_start_row
            the first filename's row
        table_start_col
            the first filename's col
        para_margin
            int，以表格第一行第一列数据点为起点，横着数计算欲求参数的间隔距离
        sheet_name
        excel_name
        plot_para_dict
            'title': str
            'x_label': str
            'y_label': str
            'figsize': tuple(width, height)
            对图像上的参数设置
            标题
            字符大小等
    :return:
    """
    # 1- get data
    data_list, label_list = single_para_boxplot_data_wrapper(table_start_row, table_start_col, para_margin, sheet_name, excel_abs_path)
    # 2- plot
    plt.figure(figsize=plot_para_dict['figsize'])

    # 标题，并设定字号大小
    plt.xlabel(xlabel=plot_para_dict['x_label'])
    plt.ylabel(ylabel=plot_para_dict['y_label'])
    plt.title(plot_para_dict['title'], fontsize=20)

    # grid=False：代表不显示背景中的网格线
    plt.boxplot(data_list, labels=label_list)
    plt.show()
# --------------------------- Draw boxplot for LRC-OVR-Stable-Alpha ---------------------------
# table_start_row = 3
# table_start_col = 21
# # para: AK
# para_margin = 5
# sheet_name = 'LRC'
# excel_abs_path = '../../ml_sl/ml_training_records.xlsx'
# plot_para_dict = {'title':'boxplot for LRC-OVR-Stable-Alpha','x_label':'Max_iteration','y_label':'Averaged AK','figsize':(16,9)}
# lrc_boxplot(table_start_row, table_start_col, para_margin, sheet_name, excel_abs_path, plot_para_dict)
# --------------------------- Draw boxplot for LRC-OVR-Stable-Alpha ---------------------------

# --------------------------- Draw boxplot for LRC-OVO-Stable-Alpha ---------------------------
# table_start_row = 3
# table_start_col = 0
# # para: AK
# para_margin = 5
# sheet_name = 'LRC'
# excel_abs_path = '../../ml_sl/ml_training_records.xlsx'
# plot_para_dict = {'title':'boxplot for LRC-OVO-Stable-Alpha','x_label':'Max_iteration','y_label':'Averaged AK','figsize':(16,9)}
# lrc_boxplot(table_start_row, table_start_col, para_margin, sheet_name, excel_abs_path, plot_para_dict)
# --------------------------- Draw boxplot for LRC-OVO-Stable-Alpha ---------------------------

def rf_boxplot(table_start_row, table_start_col, acc_kappa_margin, sheet_name, excel_abs_path, plot_para_dict=None):
    """
    :param:
        table_start_row
            the first filename's row
        table_start_col
            the first filename's col
        sheet_name
        excel_name
        txt_filename
        plot_para_dict
            'title': str
            'x_label': str
            'y_label': str
            'figsize': tuple(width, height)
            对图像上的参数设置
            标题
            字符大小等
    :return:
    """
    # 1- get data
    data_list, label_list = single_para_boxplot_data_wrapper(table_start_row, table_start_col, acc_kappa_margin, sheet_name, excel_abs_path)
    # 2- plot
    plt.figure(figsize=plot_para_dict['figsize'])

    # 标题，并设定字号大小
    plt.xlabel(xlabel=plot_para_dict['x_label'])
    plt.ylabel(ylabel=plot_para_dict['y_label'])
    plt.title(plot_para_dict['title'], fontsize=20)

    # grid=False：代表不显示背景中的网格线
    plt.boxplot(data_list, labels=label_list)
    plt.show()
# ------------------------- RF Boxplot -------------------------
# plot_para_dict = {'title':'Random Forest',
#                     'x_label': 'Tree Number',
#                     'y_label': 'Accuracy + Kappa',
#                     'figsize': (16, 9)
#                   }
# rf_boxplot(table_start_row=3, table_start_col=0, sheet_name='RF',\
#            excel_abs_path='../ml_training_records.xlsx', plot_para_dict=plot_para_dict)
# ------------------------- RF Boxplot -------------------------

def adaBoost_boxplot(table_start_row, table_start_col, acc_kappa_margin, sheet_name, excel_abs_path, plot_para_dict=None):
    """
    :param:
        table_start_row
            the first filename's row
        table_start_col
            the first filename's col
        sheet_name
        excel_name
        txt_filename
        plot_para_dict
            'title': str
            'x_label': str
            'y_label': str
            'figsize': tuple(width, height)
            对图像上的参数设置
            标题
            字符大小等
    :return:
    """
    # Set the range of y-axis (AK)
    global vmin, vmax

    # 1- get data
    data_list, label_list = single_para_boxplot_data_wrapper(table_start_row, table_start_col,
                                                             acc_kappa_margin, sheet_name, excel_abs_path)
    # 2- plot
    plt.figure(figsize=plot_para_dict['figsize'])

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 22}
    # 标题，并设定字号大小
    plt.xlabel(xlabel=plot_para_dict['x_label'], fontdict=x_y_title_font_setting)
    plt.ylabel(ylabel=plot_para_dict['y_label'], fontdict=x_y_title_font_setting)

    # No need title
    # plt.title(plot_para_dict['title'], fontsize=20)

    # grid=False：代表不显示背景中的网格线
    plt.boxplot(data_list, labels=label_list)
    plt.show()
# ------------------------- AdaBoost Boxplot -------------------------
# plot_para_dict = {'title':'AdaBoost Grid Search Results', 'figsize':(16,9), 'x_label': 'Base Learner Number', 'y_label': 'AK'}
# plot_para_dict = {'figsize':(16,9), 'x_label': 'Base Learner Number', 'y_label': 'AK'}
# adaBoost_boxplot(table_start_row=3, table_start_col=0, acc_kappa_margin=5, sheet_name='AdaBoost',
#                  excel_abs_path='../../ml_sl/ml_training_records.xlsx', plot_para_dict=plot_para_dict)
# ------------------------- AdaBoost Boxplot -------------------------

def binary_boxplot_excel(excel_abs_path, boxplot_dict, sheet_name1, para_dict1, sheet_name2, para_dict2):
    """
    Read data from excel
    :param:
        table_start_row
            the first filename's row
        table_start_col
            the first filename's col
        sheet_name
        excel_name
        txt_filename
        plot_para_dict
            'title': str
            'x_label': str
            'y_label': str
            'figsize': tuple(width, height)
            对图像上的参数设置
            标题
            字符大小等
    :return:
    """
    # Set the range of y-axis (AK)
    global vmin, vmax

    # Get the first group data
    table_start_row1 = para_dict1['table_start_row']
    table_start_col1 = para_dict1['table_start_col']
    acc_kappa_margin1 = para_dict1['acc_kappa_margin']
    data1_list, label1_list = single_para_boxplot_data_wrapper(table_start_row=table_start_row1,
                                                               table_start_col=table_start_col1,
                                                               acc_kappa_margin=acc_kappa_margin1,
                                                               sheet_name=sheet_name1,
                                                               excel_abs_path=excel_abs_path)

    # Get the second group data
    table_start_row2 = para_dict2['table_start_row']
    table_start_col2 = para_dict2['table_start_col']
    acc_kappa_margin2 = para_dict2['acc_kappa_margin']
    data2_list, label2_list = single_para_boxplot_data_wrapper(table_start_row=table_start_row2,
                                                               table_start_col=table_start_col2,
                                                               acc_kappa_margin=acc_kappa_margin2,
                                                               sheet_name=sheet_name2,
                                                               excel_abs_path=excel_abs_path)

    df1 = pd.DataFrame(columns=['AK', 'Base Learner Number', 'ML'])
    ml1_name = para_dict1['ml']
    i1 = 0
    for data1, label1 in zip(data1_list, label1_list):
        for d in data1:
            dict1 = {'AK': d, 'Base Learner Number': label1, 'ML': ml1_name}
            df1.loc[i1] = dict1
            i1 += 1

    df2 = pd.DataFrame(columns=['AK', 'Base Learner Number', 'ML'])
    ml2_name = para_dict2['ml']
    i2 = 0
    for data2, label2 in zip(data2_list, label2_list):
        for d in data2:
            dict2 = {'AK':d, 'Base Learner Number':label2, 'ML':ml2_name}
            df2.loc[i2] = dict2
            i2 += 1

    # CONCATENATE dataframe
    cdf = pd.concat([df1, df2])

    plt.figure(figsize=boxplot_dict['figsize'])
    # hue--散点图中的分类字段
    ax = sns.boxplot(x="Base Learner Number", y="AK", hue="ML", data=cdf)

    # 设置横纵坐标轴上的刻度字体大小
    plt.tick_params(labelsize=18)
    # plt.tick_params(labelsize=28)

    # 设置横纵坐标轴上的刻度字体
    labels = ax.get_xticklabels() + ax.get_yticklabels()
    [label.set_fontname('Times New Roman') for label in labels]

    # 设置横纵坐标的[名称/标题]以及对应字体格式
    x_y_title_font_setting = {'family': 'Times New Roman',
                              'weight': 'normal',
                              'size': 16}
                              # 'size': 32}
    # 标题，并设定字号大小
    plt.xlabel(xlabel=boxplot_dict['x_label'], fontdict=x_y_title_font_setting)
    plt.ylabel(ylabel=boxplot_dict['y_label'], fontdict=x_y_title_font_setting)

    plt.ylim(vmin, vmax)
    plt.show()
# ---------------------------------- RF & AB Boxplot ----------------------------------
# 最新版本的xlrd不支持excel.xlsx, 只支持excel.xls，所以重新保存一份xls格式的文件
# excel_abs_path = '../../ml/ml_training_records.xlsx'
# boxplot_dict = {
#     'figsize': (16, 9),
#     'x_label': 'Base Learner Number',
#     'y_label': 'AK',
# }
# rf_sheet_name1 = 'RF'
# rf_para_dict1 = {
#     'table_start_row': 3,
#     'table_start_col': 0,
#     'acc_kappa_margin': 4,
#     'ml':'RF'
# }
# ab_sheet_name2 = 'AdaBoost'
# ab_para_dict2 = {
#     'table_start_row': 3,
#     'table_start_col': 0,
#     'acc_kappa_margin':5,
#     'ml':'AB'
# }
# binary_boxplot_excel(excel_abs_path=excel_abs_path, boxplot_dict=boxplot_dict,
#                sheet_name1=rf_sheet_name1, para_dict1=rf_para_dict1,
#                sheet_name2=ab_sheet_name2, para_dict2=ab_para_dict2)
# ---------------------------------- RF & AB Boxplot ----------------------------------

# ---------------------------------- LRC-OvO-Stable & LRC-OvR-Stable Boxplot ----------------------------------
# 最新版本的xlrd不支持excel.xlsx, 只支持excel.xls，所以重新保存一份xls格式的文件 再去操作
# excel_abs_path = '../../ml/ml_training_records.xlsx'
excel_abs_path = '../../ml/ml_training_records.xls'
boxplot_dict = {
    'figsize': (16, 9),
    'x_label': 'Iteration',
    'y_label': 'AK',
}
lrc_ovo_stable_sheet_name1 = 'LRC'
lrc_ovo_stable_para_dict1 = {
    'table_start_row': 3,
    'table_start_col': 0,
    'acc_kappa_margin': 5,
    'ml':'LRC-OvO-Stable'
}
lrc_ovr_stable_sheet_name2 = 'LRC'
lrc_ovr_stable_para_dict2 = {
    'table_start_row': 3,
    'table_start_col': 21,
    'acc_kappa_margin': 5,
    'ml':'LRC-OvR-Stable'
}
binary_boxplot_excel(excel_abs_path=excel_abs_path, boxplot_dict=boxplot_dict,
               sheet_name1=lrc_ovo_stable_sheet_name1, para_dict1=lrc_ovo_stable_para_dict1,
               sheet_name2=lrc_ovr_stable_sheet_name2, para_dict2=lrc_ovr_stable_para_dict2)
# ---------------------------------- LRC-OvO-Stable & LRC-OvR-Stable Boxplot ----------------------------------