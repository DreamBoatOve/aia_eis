import math
from ml.knn.distance_pack.data_wrapper import single_point_list_2_list, pack_list_2_list, single_point_list_2_norm_list

# Kullback-Leibler散度（Kullback-Leibler Divergence）
def kld_distance_0(x1_list, x2_list):
    kld = 0.0
    for x1, x2 in zip(x1_list, x2_list):
        try:
            # During debugging, x1 might be 0.0, then kld = 0.0
            if x1 == 0.0:
                kld += 0.0
            else:
                kld += x1 * math.log(x1 / x2, 2)
        except ValueError as e:
            print('During calculate KLD')
            print('x1={0},x2={1}'.format(x1,x2))
            print(e)
            import sys
            sys.exit(1)
    return kld

# 杰森香浓系数 （Jansen-Shanon Divergence）
def jsd_distance_0(x_list, data_list):
    """
    :param
        x:
            a list or an np.ndarray
        data:
            a list or an np.ndarray
    :return:
        jsd_list
            measure distance between numbers
    """
    jsd_list = []
    x_list = single_point_list_2_list(x_list)
    data_list = pack_list_2_list(data_list)

    for d_list in data_list:
        avg_list = [0.5 * x + 0.5 * d for x, d in zip(x_list, d_list)]
        jsd = 0.5 * kld_distance_0(x_list, avg_list) + 0.5 * kld_distance_0(d_list, avg_list)
        jsd_list.append(jsd)
    return jsd_list

# if __name__ == '__main__':
#     x_list = [(1, 1)]
#     data_list = [[(1, 1)],
#                  [(1, 0.2)],
#                  [(0.2, 1)],
#                  [(0.3, 0.4)]]
#     jsd_list = jsd_distance_0(x_list, data_list)
#     print('jsd 0', jsd_list)

# 杰森香浓系数 （Jansen-Shanon Divergence）
def jsd_distance_1(x_list, data_list):
    """
    :param
        x:
            a list or an np.ndarray
        data:
            a list or an np.ndarray
    :return:
        jsd_list
            measure distance between norm of impedance
            Why do not measure the distance between numbers or points?
                KLD calculate the difference between two distribution that all the number are above
                if use numbers or points directly, the calculation will involve negative values
    """
    jsd_list = []
    x_norm_list = single_point_list_2_norm_list(x_list)

    for d_list in data_list:
        d_norm_list = single_point_list_2_norm_list(d_list)
        avg_norm_list = [0.5 * x + 0.5 * d for x, d in zip(x_norm_list, d_norm_list)]
        jsd = 0.5 * kld_distance_0(x_norm_list, avg_norm_list) + 0.5 * kld_distance_0(d_norm_list, avg_norm_list)
        jsd_list.append(jsd)
    return jsd_list

# if __name__ == '__main__':
#     x_list = [(1, 1)]
#     data_list = [[(1, 1)],
#                  [(1, 0.2)],
#                  [(0.2, 1)],
#                  [(0.3, 0.4)]]
#     jsd_list = jsd_distance_1(x_list, data_list)
#     print('jsd 1', jsd_list)