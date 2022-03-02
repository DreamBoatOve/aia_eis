from ml_sl.knn.distance_pack.data_wrapper import single_point_list_2_list, pack_list_2_list

def bray_curtis_distance_0(x_list, data_list):
    """
    :param
        x_list :
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list :
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        bcd_list
    """
    bcd_list = []
    x_list = single_point_list_2_list(x_list)
    data_list = pack_list_2_list(data_list)

    for d_list in data_list:
        numerator = sum([abs(x - d) for x, d in zip(x_list, d_list)])
        denominator = sum(x_list) + sum(d_list)
        bcd = numerator / denominator
        bcd_list.append(bcd)
    return bcd_list

if __name__ == '__main__':
    x_list = [(1, 1), (2, 2)] # 6
    data_list = [[(1, 1), (2, 2)], # 6
                 [(1, 2), (2, 2)], # 7
                 [(4, 1), (3, 2)], # 10
                 [(2, 1), (2, 2)], # 7
                 [(1, 3), (2, 3)]] # 9
    bcd_list = bray_curtis_distance_0(x_list, data_list)
    print(bcd_list)