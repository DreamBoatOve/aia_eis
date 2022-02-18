import math

def cal_ed_mat_0(x1_list, x2_list):
    """
    :param
        x1_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        x2_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
    :return:
        dtw
            dynamic time warping betwween x1_list and x2_list
    """
    # 1-计算帧匹配距离矩阵,两点之间的距离使用euclidean（L2），其它L1，L无穷的也有
    ed_list = []
    for i in range(len(x1_list)):
        tmp_d_list = []
        for j in range(len(x2_list)):
            d = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(x1_list[i], x2_list[j])]))
            tmp_d_list.append(d)
        ed_list.append(tmp_d_list)
    return ed_list

def update_next_three_point(cur_point, ed_list):
    cur_coor = cur_point['coor_index']

    r_len = len(ed_list)
    c_len = len(ed_list[0])

    next_list = []
    if (cur_coor[0] + 1 < r_len) and (cur_coor[1] + 1 < c_len):
        next_right = {'coor_index': (cur_coor[0], cur_coor[1] + 1), 'ed': ed_list[cur_coor[0]][cur_coor[1] + 1]}
        next_up = {'coor_index': (cur_coor[0] + 1, cur_coor[0]), 'ed': ed_list[cur_coor[0] + 1][cur_coor[0]]}
        next_up_right = {'coor_index': (cur_coor[0] + 1 , cur_coor[1] + 1), 'ed': ed_list[cur_coor[0] + 1][cur_coor[1] + 1]}
        next_list = [next_right, next_up, next_up_right]
        return next_list
    elif (cur_coor[0] + 1 >= r_len) and (cur_coor[1] + 1 < c_len):
        next_right = {'coor_index': (cur_coor[0], cur_coor[1] + 1), 'ed': ed_list[cur_coor[0]][cur_coor[1] + 1]}
        next_list = [next_right]
        return next_list
    elif (cur_coor[0] + 1 < r_len) and (cur_coor[1] + 1 >= c_len):
        next_up = {'coor_index': (cur_coor[0] + 1, cur_coor[0]), 'ed': ed_list[cur_coor[0] + 1][cur_coor[0]]}
        next_list = [next_up]
        return next_list
    else:
        return next_list

def dtw_distance_0(x_list, data_list):
    dtw_list = []
    for d_list in data_list:
        ed_list = cal_ed_mat_0(x_list, d_list)
        """
        ed_list = [[ed(x0, d0), ed(x0, d1), ed(x0, d2), ... , ed(x0, dn-2), ed(x0, dn-1)],
                   [ed(x1, d0), ed(x1, d1), ed(x1, d2), ... , ed(x1, dn-2), ed(x1, dn-1)],
                   [ed(x2, d0), ed(x2, d1), ed(x2, d2), ... , ed(x2, dn-2), ed(x2, dn-1)],
                   ...
                   [ed(xn-2, d0), ed(xn-2, d1), ed(xn-2, d2), ... , ed(xn-2, dn-2), ed(xn-2, dn-1)],
                   [ed(xn-1, d0), ed(xn-1, d1), ed(xn-1, d2), ... , ed(xn-1, dn-2), ed(xn-1, dn-1)]]
        """
        dtw = ed_list[0][0]
        cur_point = {'coor_index': (0, 0), 'ed':ed_list[0][0]}

        next_list = update_next_three_point(cur_point, ed_list)

        loop_bool = True
        while loop_bool:
            # 找距离最小的下一个位置
            tmp_ed_list = [n_p['ed'] for n_p in next_list]
            minimum_ed = min(tmp_ed_list)
            min_index = tmp_ed_list.index(minimum_ed)
            cur_point = next_list[min_index]
            dtw += cur_point['ed']
            next_list = update_next_three_point(cur_point, ed_list)
            if len(next_list) == 0:
                loop_bool = False
        dtw_list.append(dtw)
    return dtw_list

if __name__ == '__main__':
    x_list = [(0,0), (1,1), (2,2), (3,1), (4,0)]
    data_list = [[(0,0), (1,1), (2,2), (3,1), (4,0)],
                 [(0, 0), (1, 1), (2, 2), (3, 1), (4, 0), (5,0), (6,0)],
                 [(0,0),(1,0),(2,1),(3,2),(4,1),(5,0),(6,0)]]
    dtwd_list = dtw_distance_0(x_list, data_list)
    print(dtwd_list)