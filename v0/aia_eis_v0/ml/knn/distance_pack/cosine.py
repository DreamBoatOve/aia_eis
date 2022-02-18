import math
from ml.knn.distance_pack.data_wrapper import single_point_list_2_list, pack_list_2_list

# 余弦距离(Cosine Distance)
def cosine_distance_0(x_list, data_list):
    """
    :param
        x_list:
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list:
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
    :return:
        cos_list
    """
    x_list = single_point_list_2_list(x_list)
    data_list = pack_list_2_list(data_list)

    cos_list = []
    for d_list in data_list:
        numerator = sum([d * x for d, x in zip(d_list, x_list)])
        denominator = math.sqrt(sum([d ** 2 for d in d_list])) * math.sqrt(sum([x ** 2 for x in x_list]))
        cos = numerator / denominator
        cos_list.append(cos)
    return cos_list

# if __name__ == '__main__':
#     x_list = [(1,1)]
#     data_list = [[(1, 1)],
#                  [(1, 2)],
#                  [(2, 1)]]
#     cos_list = cosine_distance_0(x_list, data_list)
#     print(cos_list)