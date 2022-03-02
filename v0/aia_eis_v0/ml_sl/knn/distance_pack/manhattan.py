from ml_sl.knn .distance_pack.data_wrapper import single_point_list_2_list

# 曼哈顿距离（Manhattan Distance）
def manhattan_distance_1d(x1_list, x2_list):
    return sum([abs(x1 - x2) for x1, x2 in zip(x1_list, x2_list)])

def manhattan_distance_2d(x1_2d_list, x2_2d_list):
    return sum([manhattan_distance_1d(x1_list, x2_list) for x1_list, x2_list in zip(x1_2d_list, x2_2d_list)])

def manhattan_distance_0(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        md_list
    """
    md_list = []
    x_list = single_point_list_2_list(x_list)
    for d_list in data_list:
        d_list = single_point_list_2_list(d_list)
        md = manhattan_distance_1d(x_list, d_list)
        md_list.append(md)
    return md_list

if __name__ == '__main__':
    x_list = [(1,1), (2,2)]
    data_list = [[(1, 1), (2, 2)],
                 [(1, 2), (2, 2)],
                 [(1, 1), (3, 2)],
                 [(2, 1), (2, 2)],
                 [(2, 1), (2, 8)],
                 [(1, 1), (2, 3)]]
    md_list = manhattan_distance_0(x_list, data_list)
    print('ma 0',md_list)

def manhattan_distance_1(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        md_list
    """
    md_list = []
    for d_list in data_list:
        md = 0.0
        for x_coor, d_coor in zip(x_list, d_list):
            md += manhattan_distance_1d(x_coor, d_coor)
        md_list.append(md)
    return md_list

# if __name__ == '__main__':
#     x_list = [(1,1), (2,2)]
#     data_list = [[(1, 1), (2, 2)],
#                  [(1, 2), (2, 2)],
#                  [(1, 1), (3, 2)],
#                  [(2, 1), (2, 2)],
#                  [(2, 1), (2, 8)],
#                  [(1, 1), (2, 3)]]
#     md_list = manhattan_distance_1(x_list, data_list)
#     print('ma 1',md_list)