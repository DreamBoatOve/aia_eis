import numpy as np
import math
import os
import pickle
import multiprocessing

from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from utils.visualize_utils.fuzzy_curve_plots import fuzzy_curve_3D_plot

"""
Function
    Using Fuzzy-Surface to measure the importance of EIS spectra
    Finding out which part of the spectra matters
Routine:
    A EIS spectra, assume it has 30 points, is divided into three parts: the high frequency range (the first third of EIS, EIS-0~10),
    the middle frequency range (EIS-11~20), and the low frequency range (EIS-21~30)
    Make the Fuzzy-Curve of each frequency range, and compare their value range
Used Techs:
    Fuzzy-Curve:
        paper:
            A Fuzzy Approach to Input Variable Identification
"""

"""
Load EIS data
    数据要求：
        1- Each EIS has the same dimension because of 需要计算每个属性（每个点）的平均值
        2- Need normalization, otherwise, EIS comes from different sources has a wide impedance range
    Plan 1
        Load raw EIS data == Lai's data has 30 points and ZhuShan's data has 80 points,
        Wrong
    Plan 2
        Load ML-normed EIS data
        Right
"""
training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
dataset = training_dataset + validation_dataset + test_dataset

def cal_fuzzy_curve_on_2D(x, y, b, label):
    res_list = []
    res_with_label_list = []
    for x_c in [i / 100 for i in range(100)]:
        for y_c in [j / 100 for j in range(100)]:
            u = math.exp(- ((x-x_c)**2 + (y-y_c)**2)/(b ** 2))
            res_list.append([x_c, y_c, u])
            res_with_label_list.append(u * label)
    return res_list, res_with_label_list

def get_distance_of_point_list(index, point_list, que):
    # que == multiprocessing.Queue()
    distance_list = []
    for index, point_a in enumerate(point_list):
        for point_b in point_list[index + 1:]:
            # calculate distance between point_a and point_b
            distance_list.append(math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2))
    # return index, distance_list
    return que.put([index, distance_list])

def get_fuzzy_curve_data_mulP():
    # Use multi processes to accelerate
    global dataset
    data_num = len(dataset)
    # get the length of a EIS
    eis_length = len(dataset[0][1])
    high_fre_range = (0, int(eis_length / 3.0))
    mid_fre_range = (int(eis_length / 3.0), int(eis_length * 2 / 3.0))
    low_fre_range = (int(eis_length * 2 / 3.0), eis_length)

    # get averaged points on each coordinates
    # avg_point_list = []
    # for i in range(eis_length):
    #     x_coor_sum = 0.0
    #     y_coor_sum = 0.0
    #     for d in dataset:
    #         x_coor = d[1][i][0]
    #         y_coor = d[1][i][1]
    #         x_coor_sum += x_coor
    #         y_coor_sum += y_coor
    #     avg_point_list.append((x_coor_sum / data_num, y_coor_sum / data_num))

    # get the variable range of each column
    # for i in range(eis_length):
    #     col0_num_list = []
    #     col1_num_list = []
    #     for d in dataset:
    #         x, y = d[1][i]
    #         col0_num_list.append(x)
    #         col1_num_list.append(y)
    #     col0_min, col0_max = min(col0_num_list), max(col0_num_list)
    #     col1_min, col1_max = min(col1_num_list), max(col1_num_list)

    # get the maximum distance between two points in each column
    max_interval_list = []

    que = multiprocessing.Queue()
    job_list = []

    for i in range(eis_length):
        col_point_list = [d[1][i] for d in dataset]

        # col_distance_list = []
        # for index, point_a in enumerate(col_point_list):
        #     for point_b in col_point_list[index + 1 :]:
        #         # calculate distance between point_a and point_b
        #         col_distance_list.append(math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2))
        # max_interval_list.append(max(col_distance_list))

        p = multiprocessing.Process(target=get_distance_of_point_list, args=(i, col_point_list, que))
        job_list.append(p)
        p.start()
        # get_distance_of_point_list(index=i, point_list=col_point_list, que=que)
    for p in job_list:
        p.join()
    disordered_interval_list = [que.get() for j in job_list]
    disordered_interval_list.sort(key=lambda a: a[0])
    max_interval_list = [a[1] for a in disordered_interval_list]

    label_list = [d[0] for d in dataset]
    x_coor_list = []
    y_coor_list = []
    all_avg_curve_dict = {}
    for i, fre_range in enumerate([high_fre_range, mid_fre_range, low_fre_range]):
        curve_num = (fre_range[1] - fre_range[0]) * data_num
        curve_list = []
        curve_with_label_list = []
        for col_i in range(fre_range[0], fre_range[1]):
            for d_i, d in enumerate(dataset):
                x, y = d[1][col_i]
                b = max_interval_list[col_i]
                res_list, res_with_label_list = cal_fuzzy_curve_on_2D(x, y, b, label=label_list[d_i])
                if len(x_coor_list) == 0:
                    x_coor_list = [res[0] for res in res_list]
                if len(y_coor_list) == 0:
                    y_coor_list = [res[1] for res in res_list]
                curve_list.append([res[2] for res in res_list])
                curve_with_label_list.append(res_with_label_list)

        # Merge all the curves into one
        curve_len = len(curve_list[0])
        avg_curve_list = []
        for c_i in range(curve_len):
            # 此处多条U(x, y)曲线叠加后 无需再去除以 曲线数量：c = sum(U * label) / sum(U)
            # 曲线数量越多，sum(U * label)和sum(U)应该都是相应增大的，就可以不考虑曲线的条数
            # avg_c = sum([curve[c_i] for curve in curve_list]) / curve_num
            u_sum = sum([curve[c_i] for curve in curve_list])
            u_with_label_sum = sum([c_l[c_i] for c_l in curve_with_label_list])
            c = u_with_label_sum / u_sum
            avg_curve_list.append(c)

        if i == 0:
            key = 'high'
        elif i == 1:
            key = 'mid'
        else:
            key = 'low'
        all_avg_curve_dict[key] = avg_curve_list
    return x_coor_list, y_coor_list, all_avg_curve_dict

if __name__ == '__main__':
    x_coor_list, y_coor_list, all_avg_curve_dict = get_fuzzy_curve_data_mulP()
# ---------------- Draw Fuzzy Curve-3D of High frequency range of EIS ----------------
    high_avg_curve_list = all_avg_curve_dict['high']
    fuzzy_curve_3D_plot(x_list=x_coor_list, y_list=y_coor_list, z_list=high_avg_curve_list)
# ---------------- Draw Fuzzy Curve-3D of Hight frequency range of EIS ----------------