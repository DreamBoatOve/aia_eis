import math

def earth_mover_distance_0(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        emd_list
    """
    emd_list = []
    for d_list in data_list:
        numerator = 0.0
        for x_coor in x_list:
            for d_coor in d_list:
                ed = math.sqrt(sum([(x - d) ** 2 for x, d in zip(x_coor, d_coor)]))
                numerator += ed
        emd = numerator / (len(x_list) * len(d_list))
        emd_list.append(emd)
    return emd_list

if __name__ == '__main__':
    x_list = [(1,1), (2,2)]
    data_list = [[(1, 1), (2, 2)],
                 [(1, 2), (2, 2)],
                 [(1, 1), (3, 2)],
                 [(2, 1), (2, 2)],
                 [(1, 1), (2, 3)]]
    emd_list = earth_mover_distance_0(x_list, data_list)
    print(emd_list)