import math
from ml.knn.distance_pack.data_wrapper import single_point_list_2_list, pack_list_2_list

def standardized_euclidean_distance_0(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        sed_list
            measure distance between numbers
    """
    sed_list = []
    x_list = single_point_list_2_list(x_list)
    data_list = pack_list_2_list(data_list)

    col_standard_variance_list = []
    for col_index in range(len(data_list[0])):
        col_list = [d_list[col_index] for d_list in data_list]
        col_avg = sum(col_list) / len(col_list)
        col_standard_variance = math.sqrt(sum([(col - col_avg) ** 2 for col in col_list]) / len(col_list))
        col_standard_variance_list.append(col_standard_variance)

    for d_list in data_list:
        sed = math.sqrt(sum([((d - x) / col_sv) ** 2 for d, x, col_sv in zip(d_list, x_list, col_standard_variance_list)]))
        sed_list.append(sed)
    return sed_list

# if __name__ == '__main__':
#     x_list = [(1,1)]
#     data_list = [[(1, 1)],
#                  [(1, 2)],
#                  [(2, 1)]]
#     sed_list = standardized_euclidean_distance_0(x_list, data_list)
#     print('sed 0', sed_list)

def standardized_euclidean_distance_1(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        sed_list
            measure distance between points
    """
    sed_list = []
    data_num_list = pack_list_2_list(data_list)

    col_standard_variance_list = []
    for col_index in range(len(data_num_list[0])):
        col_list = [d_list[col_index] for d_list in data_num_list]
        col_avg = sum(col_list) / len(col_list)
        col_standard_variance = math.sqrt(sum([(col - col_avg) ** 2 for col in col_list]) / len(col_list))
        col_standard_variance_list.append(col_standard_variance)

    for d_list in data_list:
        col_index_count = 0
        sed = 0.0
        for x_coor, d_coor in zip(x_list, d_list):
            col_standard_variance_pair = col_standard_variance_list[col_index_count : col_index_count + 2]
            col_index_count += 2

            sed += math.sqrt(sum([((d - x) / col_sv) ** 2 for d, x, col_sv in zip(d_coor, x_coor, col_standard_variance_pair)]))
        sed_list.append(sed)
    return sed_list

if __name__ == '__main__':
    x_list = [(1, 1), (2, 2)]
    data_list = [[(1, 1), (2, 2)],
                 [(1, 2), (2, 2)],
                 [(1, 1), (3, 2)],
                 [(2, 1), (2, 2)],
                 [(2, 1), (2, 3)],
                 [(1, 1), (2, 3)]]
    sed_list = standardized_euclidean_distance_1(x_list, data_list)
    print('sed 1', sed_list)