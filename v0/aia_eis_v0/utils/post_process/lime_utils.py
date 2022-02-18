import numpy as np
import sys
import os
import collections
import matplotlib.pyplot as plt

from utils.file_utils.filename_utils import get_date_prefix

def feature_str2int(feature_str):
    """
    Function
        从feature_str中提取数字（EIS point order）
    :param
        feature_str:
            maybe:
                'point-1 <= 0.00',
                '0.21 <= point-54',
                '0.17 < point-128 <= 0.34',
    :return:
        int(point order), 1 ~ 160
    """
    """
    Routine
        1- 按照【空格】分割字符串后，观察得到list的长度
            1.1- 如长度=3，则为【A <= Point-N】或【Point-N <= B】
            1.2- 如长度=5，则为[A <= Point-N <= B]
            1.3- 如长度出现其它情况，报错+打印字符串+终止程序
    """
    str_list = feature_str.strip().split()

    # 1.2- 如长度=5，则为[A <= Point-N <= B]
    if len(str_list) == 5:
        point_order = int(str_list[2].split('-')[1])

    # 1.1- 如长度=3，则为【A <= Point-N】或【Point-N <= B】
    elif len(str_list) == 3:
        # 判断是否为【Point-N <= B】
        try:
            point_order = int(str_list[0].split('-')[1])
        except ValueError as e:
            print('len=3', feature_str)
            point_order = int(str_list[2].split('-')[1])

    # 1.3 - 如长度出现其它情况，打印字符串 + 终止程序
    else:
        print(feature_str)
        sys.exit(0)

    return point_order

def read_lime_explain_RF_res(fp='../../ml/rf/rf_res/lime_res', fn='2021_01_24_lime_explain_res.txt'):
    """
    Function
        读取Lime对 RF 的结果文件
    :param
        fp: dpfc_src\ml_sl\rf
        fn: 2021_01_24_lime_explain_res.txt
        file format:
            ----------Start----------
            ecm=4
            point-1 <= 0.00,0.009118543048854835
            point-89 > 0.59,0.006520717675691324
            point-26 <= 0.28,-0.0004287661951428752
            ...
            0.23 < point-139 <= 0.59,4.581205726336095e-05
            point-71 > 0.52,-4.013844195403033e-05
            point-100 <= 0.22,2.6580348141840243e-05
            ----------End----------
            ----------Start----------
            ecm=5
            0.11 < point-23 <= 0.18,0.004791066417417636
            point-148 <= 0.09,0.00477684971170501
            point-89 > 0.59,0.0047605699459234325
            ...
            0.18 < point-35 <= 0.27,0.004494778172658124
            0.17 < point-33 <= 0.26,0.00441815115254862
            point-91 > 0.60,0.0044054511347660025
            ----------End----------
            ...
    :return:
        lime_res_dict = {
            }
    """
    lime_res_dict = collections.OrderedDict()
    # 1- 内容按照【----------Start----------】和【----------End----------】分块
    with open(os.path.join(fp, fn), 'r') as f:
        for line in f.readlines():
            if line.strip() == '----------Start----------':
                continue
            elif line.strip().startswith('ecm'):
                # 'ecm=4'
                ecm_num = line.strip().split('=')[1]
                ecm_num = int(ecm_num)
                if ecm_num not in lime_res_dict.keys():
                    lime_res_dict[ecm_num] = [[]]
                else:
                    lime_res_dict[ecm_num].append([])
            elif line.strip() == '----------End----------':
                continue
            else:
                feature_str, weight_str = line.strip().split(',')
                weight = float(weight_str)
                point_order = feature_str2int(feature_str)
                lime_res_dict[ecm_num][-1].append([feature_str, weight, point_order])
    return lime_res_dict
# lime_res_dict = read_lime_explain_RF_res()

def lime_explain_RF_instance_by_fig_0(rf_explan_list, order_type, reverse=False, fig_config_dict=None):
    """
    Refer
        lime.explanation
            class Explanation
                function as_pyplot_figure
    Function
        Returns the explanation as a pyplot figure.
    :param
        rf_explan_list:
            【
                【feature_str， weight（float）， point order（int）】
                。。。
            】
        order_type: str
            'weight'：
                按照权重的大小
            'point_order'：
                按照点数的顺序
        reverse: bool
            调整数据的排序顺序
    :return:
    """
    if order_type == 'weight':
        rf_explan_list.sort(key=lambda d:d[1], reverse=reverse)
        fig = plt.figure(figsize=(4,30))
    elif order_type == 'point_order':
        rf_explan_list.sort(key=lambda d:d[2], reverse=reverse)
        fig = plt.figure(figsize=(15, 4))

    w_list = [d[1] for d in rf_explan_list]
    point_order_list = [d[2] for d in rf_explan_list]
    point_order_str_list = [str(point_order) for point_order in point_order_list]

    colors = ['green' if x > 0 else 'red' for x in w_list]

    if order_type == 'weight':
        position_list = [i + 0.5 for i in range(len(w_list))]
        plt.barh(position_list, w_list, align='center', color=colors)
        plt.yticks(position_list, point_order_list)
    elif order_type == 'point_order':
        plt.bar(point_order_list, w_list, color=colors)

    if isinstance(fig_config_dict, dict):
        ecm_num = fig_config_dict['ecm_num']
        index = fig_config_dict['index']
        title = 'ECM-{0}, No-{1}'.format(ecm_num, index)
        plt.title(title)

    plt.show()

def lime_explain_RF_instance_by_fig_1(rf_explan_list, order_type='point_order', reverse=False, fig_config_dict=None):
    """
    Refer
        lime.explanation
            class Explanation
                function as_pyplot_figure
    Version
        1: 在lime_explain_RF_instance_by_fig_0的基础上，对画图进行一些改进：
            1- 将 负数的weight 取绝对值，用红色显示
            2- 纵轴标题改为 Abs(Weight)
            3- 加两条竖线区分 高中低 三个频段
            4- 加两组横线 标明数据的Avg和Var，用不同颜色
            5- 横轴标题 EIS Points Order
    Function
        Returns the explanation as a pyplot figure.
    :param
        rf_explan_list:
            【
                【feature_str， weight（float）， point order（int）】
                。。。
            】
        order_type: str
            'weight'：
                按照权重的大小
            'point_order'：
                按照点数的顺序
        reverse: bool
            调整数据的排序顺序
    :return:
    """
    if order_type == 'weight':
        rf_explan_list.sort(key=lambda d:d[1], reverse=reverse)
        fig = plt.figure(figsize=(4,30))
    elif order_type == 'point_order':
        rf_explan_list.sort(key=lambda d:d[2], reverse=reverse)
        fig = plt.figure(figsize=(15, 4))

    w_list = [d[1] for d in rf_explan_list]
    w_len = len(w_list)
    point_order_list = [d[2] for d in rf_explan_list]
    point_order_str_list = [str(point_order) for point_order in point_order_list]

    # 1- 将 负数的weight 取绝对值，用红色显示
    colors = ['green' if x > 0 else 'red' for x in w_list]
    abs_w_list = [abs(w) for w in w_list]

    if order_type == 'weight':
        position_list = [i + 0.5 for i in range(len(w_list))]
        plt.barh(position_list, w_list, align='center', color=colors)
        plt.yticks(position_list, point_order_list)
    elif order_type == 'point_order':
        plt.bar(point_order_list, abs_w_list, color=colors)

        # 3- 加两条竖线区分 高中低 三个频段
        plt.axvline(53, linestyle='--', color='red')  # 红色虚线
        plt.axvline(106, linestyle='--', color='red')  # 红色虚线

        # 计算高中低三个区Weight的Avg
        avg_high = sum(abs_w_list[: int(w_len / 3.0)]) / int(w_len / 3.0)
        avg_mid = sum(abs_w_list[int(w_len / 3.0) : int(2 * w_len / 3.0)]) / int(2 * w_len / 3.0 - w_len / 3.0)
        avg_low = sum(abs_w_list[int(2 * w_len / 3.0) : ]) / int(w_len - int(2 * w_len / 3.0))

        # 计算高中低三个区Weight的Var
        var_high = sum([(w - avg_high)**2 for w in abs_w_list[: int(w_len / 3.0)]]) / int(w_len / 3.0)
        var_mid = sum([(w - avg_mid)**2 for w in abs_w_list[int(w_len / 3.0) : int(2 * w_len / 3.0)]]) / int(2 * w_len / 3.0 - w_len / 3.0)
        var_low = sum([(w - avg_low)**2 for w in abs_w_list[int(2 * w_len / 3.0) : ]]) / int(w_len - int(2 * w_len / 3.0))

        # 4- 加两组横线 标明数据的Avg和Var，用不同颜色
        # 4.1- 高中低三个区Weight的Avg横线
        plt.plot([i for i in range(53)], [avg_high for i in range(53)], linestyle='-.', color='yellow', linewidth=3)
        plt.plot([i for i in range(53, 106)], [avg_mid for i in range(53, 106)], linestyle='-.', color='yellow', linewidth=3)
        plt.plot([i for i in range(106, 160)], [avg_low for i in range(106, 160)], linestyle='-.', color='yellow', linewidth=3)
        # 4.2- 高中低三个区Weight的Var横线
        plt.plot([i for i in range(53)], [var_high for i in range(53)], linestyle='-.', color='red', linewidth=3)
        plt.plot([i for i in range(53, 106)], [var_mid for i in range(53, 106)], linestyle='-.', color='red', linewidth=3)
        plt.plot([i for i in range(106, 160)], [var_low for i in range(106, 160)], linestyle='-.', color='red', linewidth=3)

    if isinstance(fig_config_dict, dict):
        ecm_num = fig_config_dict['ecm_num']
        index = fig_config_dict['index']

        # 5- 横轴标题 EIS Points Order
        plt.xlabel('EIS Points Order')
        # 2- 纵轴标题改为 Abs(Weight)
        plt.ylabel('Abs(Weight)')

        title = 'ECM-{0}, No-{1}'.format(ecm_num, index)
        plt.title(title)

    plt.show()

def plot_lime_explain_RF_instance(ecm_num, order_type, reverse=False):
    lime_res_dict = read_lime_explain_RF_res()
    rf_explan_list = lime_res_dict[ecm_num]
    for i, res in enumerate(rf_explan_list):
        fig_config_dict = {'ecm_num': ecm_num, 'index': i}
        # lime_explain_RF_instance_by_fig_0(rf_explan_list=R(RC)_IS_lin-kk_res.txt, order_type=order_type, reverse=reverse,
        #                                 fig_config_dict=fig_config_dict)
        lime_explain_RF_instance_by_fig_1(rf_explan_list=res, order_type=order_type, reverse=reverse,
                                          fig_config_dict=fig_config_dict)
# plot_lime_explain_RF_instance(ecm_num=2, order_type='weight',reverse=False)
# plot_lime_explain_RF_instance(ecm_num=2, order_type='point_order',reverse=False)
# plot_lime_explain_RF_instance(ecm_num=4, order_type='point_order',reverse=False)

def lime_explain_1Kind_RF_instances(fp, fn, ecm_num, w_type=''):
    """
    :param
        fp:
        fn:
        ecm_num:
        w_type: str
            'w', raw/original weight
            'abs', the Abs(w)
            'positive', only keep the weight with positive value
    :return:
    """
    lime_res_dict = read_lime_explain_RF_res(fp, fn)
    data_list = lime_res_dict[ecm_num]
    data_len = len(data_list)

    ordered_data_list = []
    for d_list in data_list:
        # Feature_str, Weight, point order
        d_list.sort(key= lambda x: x[2], reverse=False)
        ordered_data_list.append([[d[2], d[1]] for d in d_list])

    if w_type == 'abs':
        abs_avg_list = []
        abs_var_list = []
        # EIS 160 points
        for o in range(len(data_list[0])):
            abs_w_list = []
            # Instance numbers
            for i in range(data_len):
                abs_w_list.append(abs(ordered_data_list[i][o][1]))
                # avg += ordered_data_list[i][1]
            avg_abs_w = sum(abs_w_list) / len(abs_w_list)
            var_abs_w = sum([(abs_w - avg_abs_w) ** 2 for abs_w in abs_w_list]) / len(abs_w_list)
            abs_avg_list.append(avg_abs_w)
            abs_var_list.append(var_abs_w)
        return abs_avg_list, abs_var_list
    elif w_type == 'positive':
        p_avg_list = []
        p_var_list = []
        # EIS 160 points
        for o in range(len(data_list[0])):
            p_w_list = []
            # Instance numbers
            for i in range(data_len):
                w = ordered_data_list[i][o][1]
                if w >= 0:
                    p_w_list.append(w)
                else:
                    p_w_list.append(0)
                # avg += ordered_data_list[i][1]
            avg_p_w = sum(p_w_list) / len(p_w_list)
            var_p_w = sum([(p_w - avg_p_w) ** 2 for p_w in p_w_list]) / len(p_w_list)
            p_avg_list.append(avg_p_w)
            p_var_list.append(var_p_w)
        return p_avg_list, p_var_list
    elif w_type == 'w':
        avg_list = []
        var_list = []
        # EIS 160 points
        for o in range(len(data_list[0])):
            w_list = []
            # Instance numbers
            for i in range(data_len):
                w_list.append(ordered_data_list[i][o][1])
                # avg += ordered_data_list[i][1]
            avg = sum(w_list) / len(w_list)
            var = sum([(w - avg) ** 2 for w in w_list]) / len(w_list)
            avg_list.append(avg)
            var_list.append(var)
        return avg_list, var_list
# avg_list, var_list = lime_explain_1Kind_RF_instances(fp='../../ml/rf/rf_res/lime_res', fn='2021_01_24_lime_explain_res.txt',
#                                                      ecm_num=4, w_type='abs')

def lime_res_2_XYY(ecm_num, fp='../../ml/rf/rf_res/lime_res', fn='2021_01_24_lime_explain_res.txt', w_type='abs'):
    """
    Function
        将LIME对所有样本的分析结果【dpfc_src\ml\rf\rf_res\lime_res\2021_01_24_lime_explain_res.txt】转成XYY个TXT数据用于画3D Color Surface
    :param
        fp:
        fn:
        ecm_num:
        w_type: str
            'abs', the Abs(w)
    :return:
        XYY个TXT数据用于画3D Color Surface
    """
    lime_res_dict = read_lime_explain_RF_res(fp, fn)
    data_list = lime_res_dict[ecm_num]
    data_len = len(data_list)

    ordered_data_list = []
    for d_list in data_list:
        # Feature_str, Weight, point order
        d_list.sort(key= lambda x: x[2], reverse=False)
        ordered_data_list.append([[d[2], d[1]] for d in d_list])

    if w_type == 'abs':
        res_fn = get_date_prefix()+'rf_abs(w)_ecm{0}_XYY.txt'.format(ecm_num)
        with open(res_fn, 'a+') as f:
            for point_i in range(len(ordered_data_list[0])):
                row_abs_d_list = []
                for d_i, d in enumerate(ordered_data_list):
                    row_abs_d_list.append(abs(ordered_data_list[d_i][point_i][1]))
                line_str = ','.join([str(point_i+1)] + [str(d) for d in row_abs_d_list])+'\n'
                f.write(line_str)
# lime_res_2_XYY(ecm_num=9)
"""
Generated Files are stored at:
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm2_XYY.txt
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm4_XYY.txt
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm5_XYY.txt
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm6_XYY.txt
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm7_XYY.txt
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm8_XYY.txt
    dpfc_src\ml\rf\rf_res\lime_res\abs(w)_XYY\2021_03_21_rf_abs(w)_ecm9_XYY.txt
"""