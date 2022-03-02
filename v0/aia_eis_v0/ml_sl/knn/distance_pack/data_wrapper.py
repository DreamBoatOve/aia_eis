import numpy as np
import math

# x_list = [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
# norm_list = [math.sqrt(x0 ** 2 + y0 ** 2)]
def single_point_list_2_norm_list(x_list):
    tmp_list = []
    for coor_pair in x_list:
        norm = math.sqrt(sum([x ** 2 for x in coor_pair]))
        tmp_list.append(norm)
    return tmp_list

def single_point_list_2_list(x_list):
    tmp_list = []
    for coor_pair in x_list:
        tmp_list.extend(coor_pair)
    return tmp_list

def single_list_2_arr(x_list):
    tmp_list = []
    for coor_pair in x_list:
        tmp_list.extend(coor_pair)
    return np.array(tmp_list)

# x_list = [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
def pack_list_2_list(dataset_data_list):
    data_list = []
    for dd_list in dataset_data_list:
        tmp_list = []
        for coor_pair in dd_list:
            # tmp_list.extend([[c] for c in coor_pair])
            tmp_list.extend(coor_pair)
        data_list.append(tmp_list)
    return data_list

# x_list = [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
def pack_list_2_arr(dataset_data_list):
    data_list = []
    for dd_list in dataset_data_list:
        tmp_list = []
        for coor_pair in dd_list:
            # tmp_list.extend([[c] for c in coor_pair])
            tmp_list.extend(coor_pair)
        data_list.append(tmp_list)

    # calculate the average of each column in a_class_data_list (By list)
    # avg_list = []
    # for i in range(len(data_list[0])):
    #     col_list = []
    #     for d_list in data_list:
    #         col_list.append(d_list[i])
    #     avg_list.append(sum(col_list) / len(col_list))
    # data_arr 2 * 6
    data_arr = np.array(data_list)
    return data_arr