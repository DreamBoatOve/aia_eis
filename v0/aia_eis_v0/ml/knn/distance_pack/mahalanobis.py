import numpy as np
from ml.knn.distance_pack.data_wrapper import single_list_2_arr, pack_list_2_arr

# 马哈拉诺比斯距离（Mahalanobis Distance）
# 模仿Scipy中pdist的MD距离计算，结果是一致的
def mahalanobis_distance_zzy(data):
    if isinstance(data, list):
        data_length = len(data)
        data_arr = pack_list_2_arr(data)
    else:
        data_arr = data
        data_length = data.shape[0]

    data_arr_T = data_arr.T
    cov_mat = np.cov(data_arr_T)
    in_cov_mat = np.linalg.inv(cov_mat)

    d_list = []
    for i in range(data_length):
        arr_i = data_arr[i]
        for j in range(i+1, data_length):
            arr_j = data_arr[j]
            d = np.sqrt((arr_i - arr_j).T @ in_cov_mat @ (arr_i - arr_j))
            d_list.append(d)
    return d_list

# if __name__ == '__main__':
#     np.random.seed(1)
#     x = np.random.random(10)
#     y = np.random.random(10)
#     print('x', x.shape, x)
#     print('y', y)
#
#     # mat 2 * 10
#     mat = np.vstack([x, y])
#     # mat_T 10 * 2
#     mat_T = mat.T
#     print('mat_T shape', mat_T.shape)
#
#     zzy_md = mahalanobis_distance_zzy(mat_T)
#     print(len(zzy_md))
#     print(zzy_md)
#     print('-------------------------------------------')

# Get Mahalanobis distance by Scipy package
def mahalanobis_distance_sp(class_data_arr):
    # 计算的是数据中的每个样本A与其他样本B之间的MD
    from scipy.spatial.distance import pdist
    md = pdist(class_data_arr, 'mahalanobis')
    return md

# if __name__ == '__main__':
#     np.random.seed(1)
#     x = np.random.random(10)
#     y = np.random.random(10)
#     print('x', x.shape, x)
#     print('y', y)
#
#     mat = np.vstack([x, y])
#     mat_T = mat.T
#     print('mat_T shape', mat_T.shape)
#
#     sp_md = mahalanobis_distance_sp(mat_T)
#     print(sp_md.shape)
#     print(sp_md)
#     print('-------------------------------------------')

def mahalanobis_distance_0(x_list, data):
    # calculate the average of each column in a_class_data_list
    data_list = []
    for a_class_list in data:
        tmp_list = []
        for coor_pair in a_class_list:
            # tmp_list.extend([[c] for c in coor_pair])
            tmp_list.extend(coor_pair)
        data_list.append(tmp_list)

    # calculate the average of each column in a_class_data_list (By list)
    # avg_list = []
    # for i in range(len(data_list[0])):
    #     col_list = []
    #     for d_list in data_list:
    #         col_list.append(d_list[i])
    #     avg_list.append(sum(col_list) / len(col_list))
    # data_arr 2 * 6
    data_arr = np.array(data_list)

    # calculate the average of each column in a_class_data_list (By Numpy Array)
    # np.mean(axis)
    #     axis 不设置值，对 m * n 个数求均值，返回一个实数
    #     axis = 0 ：压缩行，对各列求均值，返回 1 * n 矩阵
    #     axis = 1 ：压缩列，对各行求均值，返回 m * 1 矩阵
    # col_avg_arr 6 * 0
    col_avg_arr = np.mean(data_arr, axis=0)
    print(col_avg_arr)
    print(col_avg_arr.shape)
    # col_avg_arr.reshape(col_avg_arr.shape[0], 1)

    # x_arr = np.array(x_list)
    # x_arr.reshape(x_arr.shape[0], 1)

    # x_arr = np.array([[c] for coor_pair in x_list for c in coor_pair ])
    # x_arr 6 * 0
    x_arr = np.array([c for coor_pair in x_list for c in coor_pair ])

    # cov_mat 2 * 2
    # cov_mat = np.cov(data_arr)

    # cov_mat 6 * 6
    cov_mat = np.cov(data_arr.T)

    # inv_cov_mat = sp.linalg.inv(cov_mat)
    from scipy import linalg
    inv_cov_mat = linalg.inv(cov_mat)

    m1 = (x_arr - col_avg_arr).reshape(1, x_arr.shape[0]) @ inv_cov_mat
    m2 = m1 @ (x_arr - col_avg_arr).reshape(x_arr.shape[0], 1)

    d_square_arr = (x_arr - col_avg_arr) @ inv_cov_mat @ (x_arr - col_avg_arr).T
    print(d_square_arr)
    # d_arr = np.sqrt(d_square_arr)
    # return d_arr

def mahalanobis_distance_1(x, data):
    """
    :param
        x:
            a list or an np.ndarray
        data:
            a list or an np.ndarray
    :return:
        md_list
            list of mahalanobis distance between x and each point in data
            [md(x, data[0]), md(x, data[1]), md(x, data[2]), ..., md(x, data[n-2]), md(x, data[n-1])]
    """
    # data 如果是一个列表，列表中的每个元素是一个x_list
    if isinstance(x, list):
        x_arr = single_list_2_arr(x)
    elif isinstance(x, np.ndarray):
        # 如果不是列表，那就传入一个ARRAY
        x_arr = x

    if isinstance(data, list):
        data_length = len(data)
        data_arr = pack_list_2_arr(data)
    else:
        data_arr = data
        data_length = data.shape[0]

    data_arr_T = data_arr.T
    cov_mat = np.cov(data_arr_T)
    in_cov_mat = np.linalg.inv(cov_mat)

    md_list = []
    for i in range(data_length):
        arr_i = data_arr[i]
        md = np.sqrt((x_arr - arr_i).T @ in_cov_mat @ (x_arr - arr_i))
        md_list.append(md)
    return md_list

# if __name__ == '__main__':
#     np.random.seed(1)
#     x = np.random.random(10)
#     y = np.random.random(10)
#     print('x', x.shape, x)
#     print('y', y)
#
#     # mat 2 * 10
#     mat = np.vstack([x, y])
#     # mat_T 10 * 2
#     mat_T = mat.T
#     print('mat_T shape', mat_T.shape)
#
#     md_list = mahalanobis_distance_1(mat_T[0], mat_T)
#     print(len(md_list))
#     print(md_list)
#     print('-------------------------------------------')