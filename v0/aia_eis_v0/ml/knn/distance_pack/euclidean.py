import math
from ml.knn.distance_pack.data_wrapper import single_point_list_2_list

# 欧几里得距离（Euclidean Distance）
def euclidean_distance_1d(x1_list, x2_list):
    return math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(x1_list, x2_list)]))

# x_2d_list = [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
def euclidean_distance_2d(x1_2d_list, x2_2d_list):
    # x1_coor_pair = (x, y)
    def points_distance(x1_coor_pair, x2_coor_pair):
        return math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(x1_coor_pair, x2_coor_pair)]))
    return sum([points_distance(x1_2d, x2_2d) for x1_2d, x2_2d in zip(x1_2d_list, x2_2d_list)])

def euclidean_distance_0(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        ed_list
            measure distance between numbers
    """
    x_list = single_point_list_2_list(x_list)
    ed_list = []
    for d_list in data_list:
        d_list = single_point_list_2_list(d_list)
        ed = euclidean_distance_1d(x_list, d_list)
        ed_list.append(ed)
    return ed_list

# if __name__ == '__main__':
#     x_list = [(1,1), (2,2)]
#     data_list = [[(1, 1), (2, 2)],
#                  [(1, 2), (2, 2)],
#                  [(1, 1), (3, 2)],
#                  [(2, 1), (2, 2)],
#                  [(1, 1), (2, 3)]]
#     ed_list = euclidean_distance_0(x_list, data_list)
#     print('ed 0', ed_list)

def euclidean_distance_1(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        ed_list
            measure distance between points
    """
    ed_list = []
    for d_list in data_list:
        ed = 0.0
        for x_coor, d_coor in zip(x_list, d_list):
            ed += euclidean_distance_1d(x_coor, d_coor)
        ed_list.append(ed)
    return ed_list

# if __name__ == '__main__':
#     x_list = [(1,1), (2,2)]
#     data_list = [[(1, 1), (2, 2)],
#                  [(1, 2), (2, 2)],
#                  [(1, 1), (3, 2)],
#                  [(1, 1), (3, 3)],
#                  [(2, 1), (2, 2)],
#                  [(1, 1), (2, 3)]]
#     ed_list = euclidean_distance_0(x_list, data_list)
#     print('ed 1', ed_list)