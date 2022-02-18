from ml.knn.distance_pack.data_wrapper import single_point_list_2_list, pack_list_2_list

# 切比雪夫距离（Chebyshev Distance）
def chebyshev_distance_1d(x1_list, x2_list):
    return max([abs(x1 - x2) for x1, x2 in zip(x1_list, x2_list)])

def chebyshev_distance_2d(x1_2d_list, x2_2d_list):
    return sum(chebyshev_distance_1d(x1_list, x2_list) for x1_list, x2_list in zip(x1_2d_list, x2_2d_list))

def chebyshev_distance_0(x_list, data_list):
    """
    measure distance between numbers
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        cd_list
    """
    cd_list = []
    x_list = single_point_list_2_list(x_list)
    data_list = pack_list_2_list(data_list)

    for d_list in data_list:
        cd = max([abs(d - x) for d, x in zip(d_list, x_list)])
        cd_list.append(cd)
    return cd_list

# if __name__ == '__main__':
#     x_list = [(1,1), (2,2)]
#     data_list = [[(1, 1), (2, 2)],
#                  [(1, 2), (2, 2)],
#                  [(1, 1), (3, 2)],
#                  [(2, 1), (2, 2)],
#                  [(2, 1), (2, 5)],
#                  [(1, 1), (2, 3)]]
#     cd_list = chebyshev_distance_0(x_list, data_list)
#     print('cd 0', cd_list)

def chebyshev_distance_1(x_list, data_list):
    """
    measure distance between points
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        cd_list
    """
    cd_list = []
    for d_list in data_list:
        cd = 0.0
        for x_coor, d_coor in zip(x_list, d_list):
            cd += chebyshev_distance_1d(x_coor, d_coor)
        cd_list.append(cd)
    return cd_list

# if __name__ == '__main__':
#     x_list = [(1,1), (2,2)]
#     data_list = [[(1, 1), (2, 2)],
#                  [(1, 2), (2, 2)],
#                  [(1, 1), (3, 2)],
#                  [(2, 1), (2, 2)],
#                  [(2, 1), (2, 5)],
#                  [(1, 1), (2, 3)]]
#     cd_list = chebyshev_distance_1(x_list, data_list)
#     print('cd 1', cd_list)