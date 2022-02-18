import copy
import math
import time
import pickle

from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from utils.visualize_utils.fuzzy_curve_plots import fuzzy_curve_3D_plot

"""
Function
    Using Fuzzy-Surface to measure the importance of EIS spectra
    Finding out which part of the spectra matters
    
Routine:
    A EIS spectra, assume it has 30 points, is divided into three parts: 
        High frequency range (the first third of EIS, EIS-0~10),
        Middle frequency range (EIS-11~20)
        Low frequency range (EIS-21~30)
    Make the Fuzzy-Curve of each frequency range, and compare their value range
    
Used Techs:
    Fuzzy-Curve:
        paper:
            0- A Fuzzy Approach to Input Variable Identification, author Yinghua Lin
            1- Input variable identification - Fuzzy curves and fuzzy surfaces, author Yinghua Lin
                这篇论文中的确有介绍Fuzzy Surface来介绍两个输入在一起的 对输出影响的 衡量方式，但是我没使用论文中的方法
                这是我的观点及做法：
                    1- 论文中的Fuzzy Surface是用来应对input之间无明显关联的情景
                    2- 我的EIS在数据合理（KKT）的情况下，已知EIS的实部和虚部是有强关联的，每个数据点（实部和虚部）是一个不可分割的整体，
                       所以将一个阻抗点的影响用一个3维的高斯函数Gaussian(Zr, Zimg)表示
Version
    1
        Modification-1: 
            First, calculate all the U_surface(xi) and U_surface(xi) * y,
            C_numerator_sum += U_surface(xi) * y;  C_denominator_sum += U_surface(xi)
            C_surface = C_numerator_sum / C_denominator_sum
        
        Modification-2: 
            ax.plot_surface(X, Y, Z), X, Y, Z : 2d arrays
            Have to change x_list, y_list, z_list, to arrays
    0
        First, calculate all the U_surface(xi) and U_surface(xi) * y, 
        then C_surface(xi) = sum(U_surface(xi) * y) / sum(U_surface(xi)) 
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
            res_list.append(u)
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

def get_fuzzy_curve_data():
    global dataset
    # data_num = len(dataset)
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
    for i in range(eis_length):
        col_point_list = [d[1][i] for d in dataset]

        col_distance_list = []
        for index, point_a in enumerate(col_point_list):
            for point_b in col_point_list[index + 1 :]:
                # calculate distance between point_a and point_b
                col_distance_list.append(math.sqrt((point_a[0] - point_b[0]) ** 2 + (point_a[1] - point_b[1]) ** 2))
        max_interval_list.append(max(col_distance_list))

    label_list = [d[0] for d in dataset]

    # the coordinates is the same as Function cal_fuzzy_curve_on_2D(x, y, b, label)
    x_coor_list = []
    y_coor_list = []
    for x_c in [i / 100 for i in range(100)]:
        for y_c in [j / 100 for j in range(100)]:
            x_coor_list.append(x_c)
            y_coor_list.append(y_c)

    all_avg_curve_dict = {}
    for i, fre_range in enumerate([high_fre_range, mid_fre_range, low_fre_range]):
        # curve_num = (fre_range[1] - fre_range[0]) * data_num
        curve_sum_list = []
        curve_with_label_sum_list = []
        for col_i in range(fre_range[0], fre_range[1]):
            for d_i, d in enumerate(dataset):
                x, y = d[1][col_i]
                b = max_interval_list[col_i]
                res_list, res_with_label_list = cal_fuzzy_curve_on_2D(x, y, b, label=label_list[d_i])

                if len(curve_sum_list) == 0:
                    curve_sum_list = copy.deepcopy(res_list)
                if len(curve_with_label_sum_list) == 0:
                    curve_with_label_sum_list = copy.deepcopy(res_with_label_list)
                curve_sum_list = [c + r for c, r in zip(curve_sum_list, res_list)]
                curve_with_label_sum_list = [c_l + r_l for c_l, r_l in zip(curve_with_label_sum_list, res_with_label_list)]

        # Merge all the curves into one
        avg_curve_list = [c_with_label / c for c_with_label, c in zip(curve_with_label_sum_list, curve_sum_list)]

        if i == 0:
            key = 'high'
        elif i == 1:
            key = 'mid'
        else:
            key = 'low'
        all_avg_curve_dict[key] = avg_curve_list
    return x_coor_list, y_coor_list, all_avg_curve_dict

def pickle_FC_data(x_coor_list, y_coor_list, all_avg_curve_dict):
    """
    Function
        It costs too much time to generate data(x_coor_list, y_coor_list, all_avg_curve_dict), so pickle them
    :param
        x_coor_list:
        y_coor_list:
        all_avg_curve_dict:
    :return:
    """
    pkl_fn = get_date_prefix() + "fuzzyCurve.pkl"
    with open(pkl_fn, 'wb') as file:
        pickle.dump((x_coor_list, y_coor_list, all_avg_curve_dict), file)

def load_FC_data(fn):
    with open(fn, 'rb') as file:
        x_coor_list, y_coor_list, all_avg_curve_dict = pickle.load(file)
    return x_coor_list, y_coor_list, all_avg_curve_dict

def output_FC_data2Txt(all_avg_curve_dict, fre_range_str):
    """
    :param
        fre_range_str
            'high', 'mid'(middle), 'low'
    x/y: 0 ~ 0.99, step size 0.01
            y=0,    y=0.01, y=0.02, ..., y=0.97, y=0.98,    y=0.99
    x=0     z00000  z00001  z00002, ..., z00097, z00098,    z00099
    x=0.01  z00100
    x=0.02
    ...
    x=0.97
    x=0.98
    x=0.99  z09990  z09991  z09992, ..., z09997, z09998,    z09999
    :return:
    """
    fc_data = all_avg_curve_dict[fre_range_str]
    # for
    fn = get_date_prefix()+fre_range_str+'.txt'
    num_index = 0
    with open(fn, 'a+') as file:
        for i in range(100):
            line = ','.join([str(d) for d in fc_data[num_index : num_index + 100]]) + '\n'
            num_index += 100
            file.write(line)

# ---------------- Generate data for plotting Fuzzy Curve-3D of three frequency ranges of EIS ----------------
# start_time = time.perf_counter()
# x_coor_list, y_coor_list, all_avg_curve_dict = get_fuzzy_curve_data()
# pickle_FC_data(x_coor_list, y_coor_list, all_avg_curve_dict)
# end_time = time.perf_counter()
# print('costed time: {0}'.format(end_time - start_time))
# costed time: 1000.7041648 s
# ---------------- Generate data for plotting Fuzzy Curve-3D of three frequency ranges of EIS ----------------

# ---------------- Output Fuzzy Curve-3D plot data to txt ----------------
# x_coor_list, y_coor_list, all_avg_curve_dict = load_FC_data(fn='2020_10_10_fuzzyCurve.pkl')
# output_FC_data2Txt(all_avg_curve_dict, fre_range_str='high')
# output_FC_data2Txt(all_avg_curve_dict, fre_range_str='mid')
# output_FC_data2Txt(all_avg_curve_dict, fre_range_str='low')
# Output R(RC)_IS_lin-kk_res.txt: 2020_10_10_high.txt 2020_10_10_low.txt 2020_10_10_mid.txt
# ---------------- Output Fuzzy Curve-3D plot data to txt ----------------

# ---------------- Draw Fuzzy Curve-3D of High frequency range of EIS ----------------
# high_avg_curve_list = all_avg_curve_dict['high']
# fuzzy_curve_3D_plot(x_list=x_coor_list, y_list=y_coor_list, z_list=high_avg_curve_list)

# Max point coordinate: (0.0, 0.99, 6.005160444045324)
# Min point coordinate: (0.99, 0.0, 5.680355547183241)
# C = 6.005160444045324 - 5.680355547183241 = 0.3248048968620827
# ---------------- Draw Fuzzy Curve-3D of Hight frequency range of EIS ----------------

# ---------------- Draw Fuzzy Curve-3D of Middle frequency range of EIS ----------------
# mid_avg_curve_list = all_avg_curve_dict['mid']
# fuzzy_curve_3D_plot(x_list=x_coor_list, y_list=y_coor_list, z_list=mid_avg_curve_list)

# Max point coordinate: (0.0, 0.99, 6.065284087108585)
# Min point coordinate: (0.99, 0.0, 5.6150993006021945)
# C = 6.065284087108585 - 5.6150993006021945 = 0.4501847865063908
# ---------------- Draw Fuzzy Curve-3D of Middle frequency range of EIS ----------------

# ---------------- Draw Fuzzy Curve-3D of Low frequency range of EIS ----------------
# low_avg_curve_list = all_avg_curve_dict['low']
# fuzzy_curve_3D_plot(x_list=x_coor_list, y_list=y_coor_list, z_list=low_avg_curve_list)

# Max point coordinate: (0.0, 0.99, 6.1328519509726735)
# Min point coordinate: (0.99, 0.0, 5.670136048299981)
# C = 6.1328519509726735 - 5.670136048299981 = 0.46271590267269236
# ---------------- Draw Fuzzy Curve-3D of Low frequency range of EIS ----------------