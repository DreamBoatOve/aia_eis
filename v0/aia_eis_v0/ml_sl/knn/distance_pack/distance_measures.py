from ml_sl.knn.distance_pack.bray_curtis import bray_curtis_distance_0
from ml_sl.knn.distance_pack.chebyshev import chebyshev_distance_1
from ml_sl.knn.distance_pack.cosine import cosine_distance_0
from ml_sl.knn.distance_pack.dynamic_time_warping import dtw_distance_0
from ml_sl.knn.distance_pack.earth_mover import earth_mover_distance_0
from ml_sl.knn.distance_pack.euclidean import euclidean_distance_1
from ml_sl.knn.distance_pack.jensen_shanon_divergence import jsd_distance_1
from ml_sl.knn.distance_pack.mahalanobis import mahalanobis_distance_1
from ml_sl.knn.distance_pack.manhattan import manhattan_distance_1
from ml_sl.knn.distance_pack.pearson_correlation import pcc_distance_1
from ml_sl.knn.distance_pack.standardized_euclidean import standardized_euclidean_distance_1

def distance_measure_0(x_list, data_list, d_type_str):
    """
    :param
        x_list :
            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        data_list :
            [x_list_0 (same data structure as x_list), x_list_1, x_list_2, ..., x_list_n-2, x_list_n-1]
        d_type_str :
            measure type
                measure distance between numbers
                measure distance between points
                measure distance between norm of impedance
            'bc_d'
                bray_curtis_distance
                measure distance between numbers
            'cheb_d'
                chebyshev_distance
                measure distance between points
            'cos_d'
                cosine_distance
                measure distance between numbers
            'dtw_d'
                Dynamic time warping distance
                measure distance between points
            'em_d'
                earth_mover_distance
                measure distance between points
            'e_d'
                euclidean_distance
                measure distance between points
            'jsd_d'
                Jansen-Shanon Divergence
                measure distance between norm of impedance
                Why do not measure the distance between numbers or points?
                    KLD calculate the difference between two distribution that all the number are above
                    if use numbers or points directly, the calculation will involve negative values
            'maha_d'
                mahalanobis_distance
                measure distance between numbers
            'manha_d'
                manhattan_distance
                measure distance between points == measure distance between numbers
            'pcc_d'
                Pearson correlation coefficient
                measure distance between norm of impedance
            'se_d'
                standardized_euclidean_distance
                measure distance between points
    :return:
        d_list
    """
    data_list = [d[1] for d in data_list]
    if d_type_str == 'bc_d':
        d_list = bray_curtis_distance_0(x_list, data_list)
        return d_list
    elif d_type_str == 'cheb_d':
        d_list = chebyshev_distance_1(x_list, data_list)
        return d_list
    elif d_type_str == 'cos_d':
        d_list = cosine_distance_0(x_list, data_list)
        d_list = [1 - d for d in d_list]
        return d_list
    elif d_type_str == 'dtw_d':
        d_list = dtw_distance_0(x_list, data_list)
        return d_list
    elif d_type_str == 'e_d':
        d_list = euclidean_distance_1(x_list, data_list)
        return d_list
    elif d_type_str == 'em_d':
        d_list = earth_mover_distance_0(x_list, data_list)
        return d_list
    elif d_type_str == 'jsd_d':
        d_list = jsd_distance_1(x_list, data_list)
        return d_list
    elif d_type_str == 'maha_d':
        d_list = mahalanobis_distance_1(x_list, data_list)
        return d_list
    elif d_type_str == 'manha_d':
        d_list = manhattan_distance_1(x_list, data_list)
        return d_list
    elif d_type_str == 'pcc_d':
        d_list = pcc_distance_1(x_list, data_list)
        d_list = [abs(d) for d in d_list]
        return d_list
    elif d_type_str == 'se_d':
        d_list = standardized_euclidean_distance_1(x_list, data_list)
        return d_list
    else:
        print('Select a distance type')