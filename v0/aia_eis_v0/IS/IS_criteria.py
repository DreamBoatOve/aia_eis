import math
import numpy as np

from circuits.circuit_pack import aRCb

def cal_ChiSquare_0(z_arr, z_sim_arr, weight_type='modulus'):
    """
    :param
        weight_type: str
            'modulus': default
                Wreal = Wimag = 1 / ||Zi||
            'unity':
                Wreal = Wimag = 1
            'proportional':
                Wreal =
                Wimag =
    :return:
    """
    # weight_arr = [data_number * 2]
    weight_arr = np.zeros((z_arr.shape[0], 2))
    if weight_type == 'modulus':  # default
        for i, z in enumerate(z_arr):
            z_norm2 = 1.0 / (z.real ** 2 + z.imag ** 2)
            weight_arr[i, :] = [z_norm2, z_norm2]
    elif weight_type == 'proportional':
        # 尚不清楚
        weight_arr = None
    elif weight_type == 'unity':
        weight_arr = np.ones((z_arr.shape[0], 2))

    chi_square = 0.0
    for weight, z, z_sim in zip(weight_arr, z_arr, z_sim_arr):
        chi_square += weight[0] * ((z.real - z_sim.real) ** 2) + weight[1] * ((z.imag - z_sim.imag) ** 2)
    return chi_square

def cal_ZSimpWin_ChiSquare(data_num, para_num, z_arr, z_sim_arr, weight_type='modulus'):
    chi_square = cal_ChiSquare_0(z_arr, z_sim_arr, weight_type)

    # N测试的阻抗点数，如赖师兄N=30
    N = data_num
    # M为等效电路中需要拟合的参数数目，CPE元件包含两个参数Y和n
    M = para_num
    # v为系统的自由度，严格讲 v = N - M - 1，此处为了和ZSimpWIn一样，令 v = N - M
    v = N - M

    return chi_square / v

def cal_residual(impSpe, R_arr, tao_arr, obj_fun_mode='imag', obj_fun_weighting_type='modulus') -> np.array:
    """
    Used For Linear-KK in ECM-Vogit
    residual = (观测数据 - vogit拟合的数据) * 权重
    :param
        obj_fun_mode: str
            'real' obj_fun == loss_fun == measure the error between the fitted and experimental Z_real data
            'imag' obj_fun == loss_fun == measure the error between the fitted and experimental Z_imag data
            'both' obj_fun == loss_fun == measure the error between the fitted and experimental Z_real_and_imag data
        obj_fun_weighting_type: str
            'unity':
                Wreal = Wimag = 1
            'modulus':
                Wreal = Wimag = 1 / (z.real ** 2 + z.imag ** 2)
            'proportional':
                Wreal = Wimag = 不会写
        tao_arr： tao = time constant
        R_arr: ndarray(complex, N*0)
            [R0, R1, ..., R_M-1] or
            [Rs, R0, R1, ..., R_M-1]
        w_list: ndarray(complex, N*0)
        RC_para_list:
    :return:
        f(x, para) = weight * [(Z_sim_imag - Z_imag) ** 2]
        chiSquare_pointWise_arr: ndarray(complex, N*0, N=data points)
    """
    z_arr = impSpe.z_arr
    w_arr = impSpe.w_arr

    if obj_fun_mode == 'imag':
        # R_arr = [R0, R1, ..., R_M-1]
        RC_para_list = [[R, tao / R] for R, tao in zip(R_arr, tao_arr)]

        z_sim_arr = np.empty(shape=(len(RC_para_list), z_arr.shape[0]), dtype=complex)
        for i, RC_list in enumerate(RC_para_list):
            R, C = RC_list
            tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
            # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            z_sim_arr[i, :] = tmp_z_sim_list
        z_sim_arr = z_sim_arr.sum(axis=0)

    elif (obj_fun_mode == 'real') or (obj_fun_mode == 'both'):
        # R_arr = [*Rs*, R0, R1, ..., R_M-1]
        Rs = R_arr[0]
        RC_para_list = [[R, tao / R] for R, tao in zip(R_arr[1:], tao_arr)]

        # -------------- 计算M个RC各自产生的阻抗 --------------
        z_sim_arr = np.empty(shape=(len(RC_para_list), z_arr.shape[0]), dtype=complex)
        for i, RC_list in enumerate(RC_para_list):
            R, C = RC_list
            tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
            # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            z_sim_arr[i, :] = tmp_z_sim_list
        # -------------- 计算M个RC各自产生的阻抗 --------------
        # 合并M个RC各自产生的阻抗
        z_sim_arr = z_sim_arr.sum(axis=0)

        # 合并Rs + M个RC各自产生的阻抗
        z_sim_arr += Rs

    residual_list = []
    for sim_z, z in zip(z_sim_arr, z_arr):
        if obj_fun_mode == 'imag':
            # imag_residual = (z.imag - sim_z.imag) / np.sqrt(z.imag**2 + z.real**2)
            residual = (z.imag - sim_z.imag) / np.sqrt(z.imag**2 + z.real**2)
        elif obj_fun_mode == 'real':
            # real_residual = (z.real - sim_z.real) / np.sqrt(z.imag**2 + z.real**2)
            residual = (z.real - sim_z.real) / np.sqrt(z.imag**2 + z.real**2)
        elif obj_fun_mode == 'both':
            # both_residual = (z - sim_z) / np.sqrt(z.imag**2 + z.real**2)
            residual = (z - sim_z) / np.sqrt(z.imag**2 + z.real**2)
        residual_list.append(residual)
    return np.array(residual_list)

def cal_ChiSquare_pointWise_0(impSpe, R_arr, tao_arr, obj_fun_mode='imag', obj_fun_weighting_type='modulus') -> np.array:
    """
    Used For Linear-KK in ECM-Vogit
    :param
        obj_fun_mode: str
            'real' obj_fun == loss_fun == measure the error between the fitted and experimental Z_real data
            'imag' obj_fun == loss_fun == measure the error between the fitted and experimental Z_imag data
            'both' obj_fun == loss_fun == measure the error between the fitted and experimental Z_real_and_imag data
        obj_fun_weighting_type: str
            'unity':
                Wreal = Wimag = 1
            'modulus':
                Wreal = Wimag = 1 / (z.real ** 2 + z.imag ** 2)
            'proportional':
                Wreal = Wimag = 不会写
        tao_arr： tao = time constant
        R_arr: ndarray(complex, N*0)
            [R0, R1, ..., R_M-1] or
            [Rs, R0, R1, ..., R_M-1]
        w_list: ndarray(complex, N*0)
        RC_para_list:
    :return:
        f(x, para) = weight * [(Z_sim_imag - Z_imag) ** 2]
        chiSquare_pointWise_arr: ndarray(complex, N*0, N=data points)
    """
    z_arr = impSpe.z_arr
    w_arr = impSpe.w_arr

    if obj_fun_mode == 'imag':
        # R_arr = [R0, R1, ..., R_M-1]
        RC_para_list = [[R, tao / R] for R, tao in zip(R_arr, tao_arr)]

        z_sim_arr = np.empty(shape=(len(RC_para_list), z_arr.shape[0]), dtype=complex)
        for i, RC_list in enumerate(RC_para_list):
            R, C = RC_list
            tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
            # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            z_sim_arr[i, :] = tmp_z_sim_list
        z_sim_arr = z_sim_arr.sum(axis=0)

    elif (obj_fun_mode == 'real') or (obj_fun_mode == 'both'):
        # R_arr = [*Rs*, R0, R1, ..., R_M-1]
        Rs = R_arr[0]
        RC_para_list = [[R, tao / R] for R, tao in zip(R_arr[1:], tao_arr)]

        # -------------- 计算M个RC各自产生的阻抗 --------------
        z_sim_arr = np.empty(shape=(len(RC_para_list), z_arr.shape[0]), dtype=complex)
        for i, RC_list in enumerate(RC_para_list):
            R, C = RC_list
            tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
            # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            z_sim_arr[i, :] = tmp_z_sim_list
        # -------------- 计算M个RC各自产生的阻抗 --------------
        # 合并M个RC各自产生的阻抗
        z_sim_arr = z_sim_arr.sum(axis=0)

        # 合并Rs + M个RC各自产生的阻抗
        z_sim_arr += Rs

    chiSquare_pointWise_list = []
    for sim_z, z in zip(z_sim_arr, z_arr):
        if obj_fun_mode == 'imag':
            chiSquare_pointWise = (sim_z.imag - z.imag) ** 2
        elif obj_fun_mode == 'real':
            chiSquare_pointWise = (sim_z.real - z.real) ** 2
        elif obj_fun_mode == 'both':
            chiSquare_pointWise = (sim_z.imag - z.imag) ** 2 + (sim_z.real - z.real) ** 2
        chiSquare_pointWise_list.append(chiSquare_pointWise)

    chiSquare_pointWise_arr = None
    if obj_fun_weighting_type == 'modulus':
        # paper: A Linear Kronig-Kramers Transform Test for Immittance Data Validation - Eq 13
        weight_list = [1 / (z.real ** 2 + z.imag ** 2) for z in z_arr]
        weighted_chiSquare_pointWise_list = [weight * zimag_residual for weight, zimag_residual in
                                             zip(weight_list, chiSquare_pointWise_list)]
        chiSquare_pointWise_arr = np.array(weighted_chiSquare_pointWise_list)
    elif obj_fun_weighting_type == 'proportional':
        pass
    elif obj_fun_weighting_type == 'unity':
        chiSquare_pointWise_arr = np.array(chiSquare_pointWise_list)

    return chiSquare_pointWise_arr