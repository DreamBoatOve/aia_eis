import numpy as np

def point_2_x_y(point_list):
    """
    point_list
        [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)]
    return
        x_list = [x0, x1, x2, ...]
        y_list = [y0, y1, y2, ...]
    """
    x_list = []
    y_list = []
    for point in point_list:
        x_list.append(point[0])
        y_list.append(point[1])
    return x_list, y_list

def single_point_list_2_list(x_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
    :return:
        tmp_list
            [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
    """
    tmp_list = []
    for coor_pair in x_list:
        tmp_list.extend(coor_pair)
    return tmp_list

def pack_list_2_list(dataset_data_list):
    """
    :param
        dataset_data_list:
            [
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                ...
            ]
    :return:
        data_list
            [
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                ...
            ]
    """
    data_list = []
    for dd_list in dataset_data_list:
        tmp_list = []
        for coor_pair in dd_list:
            # tmp_list.extend([[c] for c in coor_pair])
            tmp_list.extend(list(coor_pair))
        data_list.append(tmp_list)
    return data_list

# 将完整的数据集（带标签的）分成 = 标签 list + 数据 list（不带有标签）
def split_labeled_dataset_list(labeled_dataset_list):
    """
    :param
        labeled_dataset_list:
            [
                [label number, points list]
                [1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [3, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [4, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                ...
            ]
    :return:
        label_list
            [label1, label3, label4, ...]
        points_list
            [
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                ...
            ]
    """
    label_list = []
    points_list = []
    for d_list in labeled_dataset_list:
        label_list.append(d_list[0])
        points_list.append(d_list[1])
    return label_list, points_list

def reform_labeled_dataset_list(labeled_dataset_list):
    """
    :param
        labeled_dataset_list:
            [
                [label number, points list]
                [1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [3, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [4, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                ...
            ]
    :return:
        reformed_labeled_dataset_list
            [
                [label1, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                [label3, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                [label4, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                ...
            ]
    """
    reformed_labeled_dataset_list = []
    for labeled_data_list in labeled_dataset_list:
        data_list = single_point_list_2_list(labeled_data_list[1])
        reformed_labeled_dataset_list.append([labeled_data_list[0], data_list])
    return reformed_labeled_dataset_list

"""
labeled_dataset_2_data_and_label_arr
    1-将带有标签的数据集分成数据（np-arr）和标签（np-arr）
    2-原有标签较小的，转化为-1标签，原有标签较大的，转化为+1标签：
        如：
            one original label 9 --> 1
            the other original label 8 --> -1
"""
def labeled_dataset_2_data_and_label_arr(labeled_dataset_list):
    """
    :param
        labeled_dataset_list:
            此处只会有两种标签，因为一个SVM只能做二分类的任务
            [
                [label number, points list]
                [4, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [3, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [4, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [3, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                [4, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                ...
            ]
    :return:
        data_arr
            [
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                ...
            ]
        label_arr
            (4 --> 1; 3 ---> -1)
            [1, -1, 1, -1, 1, ...]
    """
    data_list = []
    label_list = []
    for data in labeled_dataset_list:
        label_list.append(data[0])
        tmp_d_list = []
        for d_pair in data[1]:
            tmp_d_list.extend(d_pair)
        data_list.append(tmp_d_list)
    min_label = min(label_list)
    max_label = max(label_list)

    svm_label_list = []
    for label in label_list:
        if label == min_label:
            svm_label_list.append(-1)
        elif label == max_label:
            svm_label_list.append(1)
    return np.array(data_list), np.array(svm_label_list)