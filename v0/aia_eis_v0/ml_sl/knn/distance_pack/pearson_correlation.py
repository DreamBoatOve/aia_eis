import math
from ml_sl.knn.distance_pack.data_wrapper import single_point_list_2_list, pack_list_2_list, single_point_list_2_norm_list

# 皮尔森相关系数 （Pearson correlation coefficient）
def pcc_distance_0(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        pccd_list
            measure distance between numbers
    """
    pccd_list = []
    x_list = single_point_list_2_list(x_list)
    data_list = pack_list_2_list(data_list)

    for d_list in data_list:
        x_avg = sum(x_list) / len(x_list)
        d_avg = sum(d_list) / len(d_list)
        numerator = sum([(x - x_avg) * (d - d_avg) for x, d in zip(x_list, d_list)])
        denominator = math.sqrt(sum([(x - x_avg) ** 2 for x in x_list])) * math.sqrt(sum([(d - d_avg) ** 2 for d in d_list]))
        pccd = numerator / denominator
        pccd_list.append(pccd)
    return pccd_list

# if __name__ == '__main__':
#     x_list = [(1,2,3)]
#     data_list = [[(1, 3, 2)],
#                  [(1, 2, 2)],
#                  [(3, 2, 1)]]
#     pccd_list = pcc_distance_0(x_list, data_list)
#     print('pccd 0',pccd_list)

def pcc_distance_1(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        pccd_list
            measure distance between norm of impedance
    """
    pccd_list = []
    x_norm_list = single_point_list_2_norm_list(x_list)

    for d_list in data_list:
        d_norm_list = single_point_list_2_norm_list(d_list)
        x_norm_avg = sum(x_norm_list) / len(x_norm_list)
        d_norm_avg = sum(d_norm_list) / len(d_norm_list)
        numerator = sum([(x - x_norm_avg) * (d - d_norm_avg) for x, d in zip(x_norm_list, d_norm_list)])
        denominator = math.sqrt(sum([(x - x_norm_avg) ** 2 for x in x_norm_list])) * math.sqrt(sum([(d - d_norm_avg) ** 2 for d in d_norm_list]))
        pccd = numerator / denominator
        pccd_list.append(pccd)
    return pccd_list

# if __name__ == '__main__':
#     # 这组数据测试不出pcc的成否
#     # x_list = [(1,1), (2,2)]         # [sqrt(2), 2]
#     # data_list = [[(1, 1), (2, 2)],  # [sqrt(2), 2]
#     #              [(1, 2), (2, 2)],  # [sqrt(5), 2]
#     #              [(1, 1), (3, 2)],  # [sqrt(2), sqrt(2)]
#     #              [(2, 1), (2, 2)],  # [sqrt(5), 2]
#     #              [(1, 1), (2, 3)]]  # [sqrt(2), sqrt(13)]
#     x_list = [(1,1), (2,2), (3,3), (4,4)]
#     data_list = [[(1, 1), (2, 2), (2, 2), (1, 1)],
#                  [(1, 1), (2, 2), (3, 3), (4, 4)],
#                  [(4, 4), (3, 3), (2, 2), (1, 1)],
#                  [(1, 2), (2, 2), (3, 2), (4, 2)]]
#     pccd_list = pcc_distance_1(x_list, data_list)
#     print('pccd 1', pccd_list)