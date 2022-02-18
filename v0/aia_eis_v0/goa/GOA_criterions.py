import math
from circuits.circuit_pack import *

"""
GOA算法终止条件
    GOA算法终止条件1- GOA前后两次迭代的全局最优解之间，前后两代每个x的误差 < 1%
        全局最优解的最后两个list

    GOA算法终止条件2- Chi-Squared在GOA前后两次迭代的全局最优解之间，将元件参数带入阻抗数据计算得到的Chi-Squared，相差 < 1e-5
        **fre_list or w_list + ECM-num ==> Z-prediction
        Z-Raw

    GOA算法终止条件3- 超过最大迭代次数
        最大迭代次数

    每个函数判断一条标准，返回判断结果criterion_flag，
        criterion_flag
            True    符合要求
            False   不符合要求
    最后汇总到一起，任意一个标准符合即可终止算法
"""

# GOA算法终止条件1- GOA前后两次迭代的全局最优解之间，前后两代每个x的误差 < 1%
def goa_criterion_1_check(x1_list, x2_list):
    """
    function
        Calculate the difference between the current and previous x_list
        if the difference between each pair in the same index of the current and previous x_list is less than 1%:
            the x_list is good enough to be returned
    :param
        x1_list:
            previous x_list
        x2_list:
            current x_list
    :return:
    """
    criterion_flag = False
    criterion_list = []
    for x1, x2 in zip(x1_list, x2_list):
        if abs((x2 - x1) / x1) < 0.01:
            criterion_list.append(0)
        else:
            criterion_list.append(1)
    if sum(criterion_list) == 0:
        criterion_flag = True
    return criterion_flag

# GOA算法终止条件2- Chi-Squared (cs) between z_raw_list and z_pre_list, if cs < CS_limit, cs is small enough and good.
def goa_criterion_2_check(ECM_paras_list, ECM_num, fre_list, z_raw_complex_list, CS_limit=1e-5):
    """
    function:
        Calculate the Chi-Squared (cs) between z_raw_complex_list and the predicted z_pre_complex_list
        if cs < 1e-5: return True
        else: return False
    :param
        ECM_paras_list:
            ECM para list Predicted by GOAs
        ECM_num:
        fre_list:
        z_raw_complex_list:
            Raw EIS complex list
    :return:
    """
    w_list = [2 * math.pi * f for f in fre_list]

    if ECM_num == 0:
        R0, R1, C0 = ECM_paras_list
        z_pre_complex_list = [RaCRb(w, R0, R1, C0) for w in w_list]
    elif ECM_num == 1:
        R0, Q0_pair_q, Q0_pair_n, R1 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        z_pre_complex_list = [RaQRb(w, R0, Q0_pair, R1) for w in w_list]
    elif ECM_num == 2:
        R0, Q0_pair_q, Q0_pair_n, R1, Q1_pair_q, Q1_pair_n, R2 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        Q1_pair = [Q1_pair_q, Q1_pair_n]
        z_pre_complex_list = [RaQRbaQRb(w, R0, Q0_pair, R1, Q1_pair, R2) for w in w_list]
    elif ECM_num == 3:
        R0, Q0_pair_q, Q0_pair_n, R1, L0, R2 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        z_pre_complex_list = [RaQRaLRbb(w, R0, Q0_pair, R1, L0, R2) for w in w_list]
    elif ECM_num == 4:
        R0, Q0_pair_q, Q0_pair_n, R1, W0 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        z_pre_complex_list = [RaQaRWbb(w, R0, Q0_pair, R1, W0) for w in w_list]
    elif ECM_num == 5:
        R0, Q0_pair_q, Q0_pair_n, R1, Q1_pair_q, Q1_pair_n, R2, W0 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        Q1_pair = [Q1_pair_q, Q1_pair_n]
        z_pre_complex_list = [RaQRbaQRbW(w, R0, Q0_pair, R1, Q1_pair, R2, W0) for w in w_list]
    elif ECM_num == 6:
        R0, Q0_pair_q, Q0_pair_n, R1, Q1_pair_q, Q1_pair_n, R2, W0 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        Q1_pair = [Q1_pair_q, Q1_pair_n]
        z_pre_complex_list = [RaQRbaQaRWbb(w, R0, Q0_pair, R1, Q1_pair, R2, W0) for w in w_list]
    elif ECM_num == 7:
        R0, Q0_pair_q, Q0_pair_n, R1, W0 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        z_pre_complex_list = [RaQRbW(w, R0, Q0_pair, R1, W0) for w in w_list]
    elif ECM_num == 8:
        R0, Q0_pair_q, Q0_pair_n, R1, W0, Q1_pair_q, Q1_pair_n = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        Q1_pair = [Q1_pair_q, Q1_pair_n]
        z_pre_complex_list = [RaQaRWbbQ(w, R0, Q0_pair, R1, W0, Q1_pair) for w in w_list]
    elif ECM_num == 9:
        R0, Q0_pair_q, Q0_pair_n, R1, Q1_pair_q, Q1_pair_n, R2 = ECM_paras_list
        Q0_pair = [Q0_pair_q, Q0_pair_n]
        Q1_pair = [Q1_pair_q, Q1_pair_n]
        z_pre_complex_list = [RaQaRaQRbbb(w, R0, Q0_pair, R1, Q1_pair, R2) for w in w_list]
    else:
        print('I just set 10 kinds of ECM, you must choose ECM type from 0 ~ 9')
        return

    """
    Refer:
        Right:
            ZSimpWin: Tech Note 37: Least Squares Fit Formulation --> 3. Definition of the chi-squared --> Eq 10中
            对Chi-Square的定义
        Wrong:
            ZSimpWin-Help/Introduction/Definition of the Chi-squared,这个版本没写全
    """
    cs = 0.0
    for z_raw, z_pre in zip(z_raw_complex_list, z_pre_complex_list):
        d_square = (z_raw.real - z_pre.real) ** 2 + (z_raw.imag - z_pre.imag) ** 2
        # mw, Modulus Weighting
        mw_square = z_raw.real ** 2 + z_raw.imag ** 2
        cs += d_square / mw_square
    # N测试的阻抗点数，如赖师兄N=30
    N = len(fre_list)
    # M为等效电路中需要拟合的参数数目，CPE元件包含两个参数Y和n
    M = len(ECM_paras_list)
    # v为系统的自由度，严格讲 v = N - M - 1，此处为了和ZSimpWIn一样，令 v = N - M
    v = N - M
    cs /= v

    criterion_flag = False
    # if cs < 1e-5:
    if cs < CS_limit:
        criterion_flag = True
    return criterion_flag, cs

# GOA算法终止条件3- 检验是否超过最大迭代次数
def goa_criterion_3_check(iter, max_iter_time):
    criterion_flag = False
    if iter >= max_iter_time:
        criterion_flag = True
    return criterion_flag

def goa_criterion_pack(x_lists_list, iter, max_iter_time, data_dict, CS_limit=1e-25):
    """
    :param
        x_lists_list:
            x1_list + x2_list
            x1_list
                第 n 次迭代中，最优个体的位置
            x2_list
                第 n+1 次迭代中，最优个体的位置
        iter:
            当前的迭代次数
        max_iter_time:
            人为设定的算法最大迭代次数
        data_dict:
            包含EIS参数拟合的标准信息
                ECM型号              ECM_num
                测试频率              fre_list
                测试所得复数阻抗       z_raw_complex_list
    :return:
    """
    x1_list, x2_list = x_lists_list

    ECM_paras_list = x2_list
    ECM_num = data_dict['ecm_num']
    fre_list = data_dict['f']
    if 'z_raw' in data_dict.keys():
        z_raw_complex_list = data_dict['z_raw']
    elif 'z_sim' in data_dict.keys():
        z_raw_complex_list = data_dict['z_sim']

    # GOA算法终止条件1- GOA前后两次迭代的全局最优解之间，前后两代每个x的误差 < 1%
    goa_criterion_flag_1 = goa_criterion_1_check(x1_list, x2_list)
    # GOA算法终止条件2- Chi-Squared (cs) between z_raw_list and z_pre_list, if cs < CS_limit, cs is small enough and good.
    goa_criterion_flag_2, chi_squared = goa_criterion_2_check(ECM_paras_list, ECM_num, fre_list, z_raw_complex_list, CS_limit)
    # GOA算法终止条件3- 检验是否超过最大迭代次数
    goa_criterion_flag_3 = goa_criterion_3_check(iter, max_iter_time)

    goa_criterion = False
    if goa_criterion_flag_3:
        goa_criterion = True
        return goa_criterion, chi_squared
    else:
        if goa_criterion_flag_1 and goa_criterion_flag_2:
            goa_criterion = True
            return goa_criterion, chi_squared
        else:
            return goa_criterion, chi_squared

def goa_rel_std_err(x_lists_list):
    x_len = len(x_lists_list[0])
    rel_std_err_list = []
    for col_i in range(x_len):
        col_list = [x_list[col_i] for x_list in x_lists_list]
        col_avg = sum(col_list) / len(col_list)
        col_std_err = math.sqrt(sum([(col_avg - c)**2 for c in col_list]) / len(col_list))
        col_rel_std_err = 100 * col_std_err / col_list[-1]
        rel_std_err_list.append(col_rel_std_err)
    return rel_std_err_list