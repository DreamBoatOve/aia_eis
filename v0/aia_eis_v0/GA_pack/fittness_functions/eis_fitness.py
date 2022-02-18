import numpy as np
import math
from circuits.circuit_pack import *

def cal_EIS_sim_complex_list(fre_list, ECM_num, ECM_paras_list):
    """
    :param
        fre_list:
            list(float, float,...)
            [0.002, 0.05, 0.8, ..., 1000000]
            These frequency are read from raw experiment data, like DTA files
        w_list:
            list(float)
            w, angle frequency, w = 2 * pi * frequency
        ECM_num:
            int
            Each number corresponds to a specific ECM
                Circuit(ECM) No.    CDC             Function
                0                   R(CR)           RaCRb, Simplified Randles Cell
                0                                   R0aC0R1b

                1                   R(QR)           RaQRb
                1                   R(QR)           R0aQ0R1b

                2                   R(QR)(QR)       RaQRbaQRb
                2                   R(QR)(QR)       R0aQ0R1baQ1R2b

                3                   R(QR(LR))       RaQRaLRbb
                3                   R(QR(LR))       R0aQ0R1aL0R2bb

                4                   R(Q(RW))        RaQaRWbb
                4                   R(Q(RW))        R0aQ0aR1W0bb

                5                   R(QR)(QR)W      RaQRbaQRbW
                5                   R(QR)(QR)W      R0aQ0R1baQ1R2bW0

                6                   R(QR)(Q(RW))    RaQRbaQaRWbb
                6                   R(QR)(Q(RW))    R0aQ0R1baQ1aR2W0bb

                7                   R(QR)W          RaQRbW
                7                   R(QR)W          R0aQ0R1bW0

                8                   R(Q(RW))Q       RaQaRWbbQ
                8                   R(Q(RW))Q       R0aQ0aR1W0bbQ1

                9                   R(Q(R(QR)))     RaQaRaQRbbb
                9                   R(Q(R(QR)))     R0aQ0aR1aQ1R2bbb
        ECM_paras_list
            list(float)
            Store the values of elements in the ECM
            Store order of values are the same as the [circuit function's name], like:
                RaCRb(w, R0, R1, C0) ==> [R0, R1, C0]
                RaQRb(w, R0, Q0_pair, R1) ==> [R0, Q0_pair_q, Q0_pair_n, R1]
    :return:
        z_pre_complex_list:
            list(complex)
            Take the calculated circuit elements' values into the ECM to get the simulated Impedance
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
    return z_pre_complex_list

# Square Error, many inputs
def cal_EIS_SE_fitness_0(fre_list, ECM_num, ECM_paras_list, z_raw_complex_list):
    """
    :param
        fre_list:
            list(float, float,...)
            [0.002, 0.05, 0.8, ..., 1000000]
            These frequency are read from raw experiment data, like DTA files
        w_list:
            list(float)
            w, angle frequency, w = 2 * pi * frequency
        ECM_num:
            int
            Each number corresponds to a specific ECM
                Circuit(ECM) No.    CDC             Function
                0                   R(CR)           RaCRb, Simplified Randles Cell
                0                                   R0aC0R1b

                1                   R(QR)           RaQRb
                1                   R(QR)           R0aQ0R1b

                2                   R(QR)(QR)       RaQRbaQRb
                2                   R(QR)(QR)       R0aQ0R1baQ1R2b

                3                   R(QR(LR))       RaQRaLRbb
                3                   R(QR(LR))       R0aQ0R1aL0R2bb

                4                   R(Q(RW))        RaQaRWbb
                4                   R(Q(RW))        R0aQ0aR1W0bb

                5                   R(QR)(QR)W      RaQRbaQRbW
                5                   R(QR)(QR)W      R0aQ0R1baQ1R2bW0

                6                   R(QR)(Q(RW))    RaQRbaQaRWbb
                6                   R(QR)(Q(RW))    R0aQ0R1baQ1aR2W0bb

                7                   R(QR)W          RaQRbW
                7                   R(QR)W          R0aQ0R1bW0

                8                   R(Q(RW))Q       RaQaRWbbQ
                8                   R(Q(RW))Q       R0aQ0aR1W0bbQ1

                9                   R(Q(R(QR)))     RaQaRaQRbbb
                9                   R(Q(R(QR)))     R0aQ0aR1aQ1R2bbb
        ECM_paras_list
            list(float)
            Store the values of elements in the ECM
            Store order of values are the same as the [circuit function's name], like:
                RaCRb(w, R0, R1, C0) ==> [R0, R1, C0]
                RaQRb(w, R0, Q0_pair, R1) ==> [R0, Q0_pair_q, Q0_pair_n, R1]
        z_raw_complex_list:
            list(complex)
            This list is read from raw experiment data, like DTA files
        z_pre_complex_list:
            list(complex)
            Take the calculated circuit elements' values into the ECM to get the simulated Impedance
    :return:
        z_square_err
            float
            The square of the difference between the raw-Z and simulate-Z
            z_square_err = (raw_Z1.real - sim_Z1.real) ** 2 + (raw_Z1.imag - sim_Z1.imag) ** 2
                            +(raw_Z2.real - sim_Z2.real) ** 2 + (raw_Z2.imag - sim_Z2.imag) ** 2
                            +...
    """
    z_sim_complex_list = cal_EIS_sim_complex_list(fre_list, ECM_num, ECM_paras_list)

    # Calculate the Square of difference between the raw-Z and simulate-Z
    z_square_err = sum([(rz.real - sz.real)**2 + (rz.imag - sz.imag)**2 for rz, sz in zip(z_raw_complex_list, z_sim_complex_list)])
    return z_square_err

# Square Error, two inputs
def cal_EIS_SE_fitness_1(exp_data_dict, ECM_paras_list):
    """
    :param
        exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试频率              'f',         fre_list
                    测试所得复数阻抗       'z_raw',     z_raw_complex_list
        fre_list:
            list(float, float,...)
            [0.002, 0.05, 0.8, ..., 1000000]
            These frequency are read from raw experiment data, like DTA files
        w_list:
            list(float)
            w, angle frequency, w = 2 * pi * frequency
        ECM_num:
            int
            Each number corresponds to a specific ECM
                Circuit(ECM) No.    CDC             Function
                0                   R(CR)           RaCRb, Simplified Randles Cell
                0                                   R0aC0R1b

                1                   R(QR)           RaQRb
                1                   R(QR)           R0aQ0R1b

                2                   R(QR)(QR)       RaQRbaQRb
                2                   R(QR)(QR)       R0aQ0R1baQ1R2b

                3                   R(QR(LR))       RaQRaLRbb
                3                   R(QR(LR))       R0aQ0R1aL0R2bb

                4                   R(Q(RW))        RaQaRWbb
                4                   R(Q(RW))        R0aQ0aR1W0bb

                5                   R(QR)(QR)W      RaQRbaQRbW
                5                   R(QR)(QR)W      R0aQ0R1baQ1R2bW0

                6                   R(QR)(Q(RW))    RaQRbaQaRWbb
                6                   R(QR)(Q(RW))    R0aQ0R1baQ1aR2W0bb

                7                   R(QR)W          RaQRbW
                7                   R(QR)W          R0aQ0R1bW0

                8                   R(Q(RW))Q       RaQaRWbbQ
                8                   R(Q(RW))Q       R0aQ0aR1W0bbQ1

                9                   R(Q(R(QR)))     RaQaRaQRbbb
                9                   R(Q(R(QR)))     R0aQ0aR1aQ1R2bbb
        ECM_paras_list
            list(float)
            Store the values of elements in the ECM
            Store order of values are the same as the [circuit function's name], like:
                RaCRb(w, R0, R1, C0) ==> [R0, R1, C0]
                RaQRb(w, R0, Q0_pair, R1) ==> [R0, Q0_pair_q, Q0_pair_n, R1]
        z_raw_complex_list:
            list(complex)
            This list is read from raw experiment data, like DTA files
        z_pre_complex_list:
            list(complex)
            Take the calculated circuit elements' values into the ECM to get the simulated Impedance
    :return:
        z_square_err
            float
            The square of the difference between the raw-Z and simulate-Z
            z_square_err = (raw_Z1.real - sim_Z1.real) ** 2 + (raw_Z1.imag - sim_Z1.imag) ** 2
                            +(raw_Z2.real - sim_Z2.real) ** 2 + (raw_Z2.imag - sim_Z2.imag) ** 2
                            +...
    """
    fre_list = exp_data_dict['f']
    ECM_num = exp_data_dict['ecm_num']
    if 'z_raw' in exp_data_dict.keys():
        z_raw_complex_list = exp_data_dict['z_raw']
    elif 'z_sim' in exp_data_dict.keys():
        z_raw_complex_list = exp_data_dict['z_sim']

    z_sim_complex_list = cal_EIS_sim_complex_list(fre_list, ECM_num, ECM_paras_list)

    # Calculate the Square of difference between the raw-Z and simulate-Z
    z_square_err = sum([(rz.real - sz.real)**2 + (rz.imag - sz.imag)**2 for rz, sz in zip(z_raw_complex_list, z_sim_complex_list)])
    return z_square_err

# Weighted Square Error, many inputs
def cal_EIS_WSE_fitness_0(fre_list, ECM_num, ECM_paras_list, z_raw_complex_list):
    """
    :param
        fre_list:
            list(float, float,...)
            [0.002, 0.05, 0.8, ..., 1000000]
            These frequency are read from raw experiment data, like DTA files
        w_list:
            list(float)
            w, angle frequency, w = 2 * pi * frequency
        ECM_num:
            int
            Each number corresponds to a specific ECM
                Circuit(ECM) No.    CDC             Function
                0                   R(CR)           RaCRb, Simplified Randles Cell
                0                                   R0aC0R1b

                1                   R(QR)           RaQRb
                1                   R(QR)           R0aQ0R1b

                2                   R(QR)(QR)       RaQRbaQRb
                2                   R(QR)(QR)       R0aQ0R1baQ1R2b

                3                   R(QR(LR))       RaQRaLRbb
                3                   R(QR(LR))       R0aQ0R1aL0R2bb

                4                   R(Q(RW))        RaQaRWbb
                4                   R(Q(RW))        R0aQ0aR1W0bb

                5                   R(QR)(QR)W      RaQRbaQRbW
                5                   R(QR)(QR)W      R0aQ0R1baQ1R2bW0

                6                   R(QR)(Q(RW))    RaQRbaQaRWbb
                6                   R(QR)(Q(RW))    R0aQ0R1baQ1aR2W0bb

                7                   R(QR)W          RaQRbW
                7                   R(QR)W          R0aQ0R1bW0

                8                   R(Q(RW))Q       RaQaRWbbQ
                8                   R(Q(RW))Q       R0aQ0aR1W0bbQ1

                9                   R(Q(R(QR)))     RaQaRaQRbbb
                9                   R(Q(R(QR)))     R0aQ0aR1aQ1R2bbb
        ECM_paras_list
            list(float)
            Store the values of elements in the ECM
            Store order of values are the same as the [circuit function's name], like:
                RaCRb(w, R0, R1, C0) ==> [R0, R1, C0]
                RaQRb(w, R0, Q0_pair, R1) ==> [R0, Q0_pair_q, Q0_pair_n, R1]
        z_raw_complex_list:
            list(complex)
            This list is read from raw experiment data, like DTA files
        z_pre_complex_list:
            list(complex)
            Take the calculated circuit elements' values into the ECM to get the simulated Impedance
    :return:
        z_weighted_square_err
            float
            The square of the difference between the raw-Z and simulate-Z
            Weight, W = 1 / ||Z||, Z (Raw Impedance)
            z_weighted_square_err = W1 * ( (raw_Z1.real - sim_Z1.real) ** 2 + (raw_Z1.imag - sim_Z1.imag) ** 2 )
                                   +W2 * ( (raw_Z2.real - sim_Z2.real) ** 2 + (raw_Z2.imag - sim_Z2.imag) ** 2 )
                                   +...
    """
    z_sim_complex_list = cal_EIS_sim_complex_list(fre_list, ECM_num, ECM_paras_list)

    # Calculate the Weighted Square of difference between the raw-Z and simulated-Z
    def cal_w(raw_Z):
        return 1 / (raw_Z.real**2 + raw_Z.imag**2)
    z_weighted_square_err = sum([cal_w(rz) * ((rz.real - sz.real)**2 + (rz.imag - sz.imag)**2) for rz, sz in zip(z_raw_complex_list, z_sim_complex_list)])
    return z_weighted_square_err

def cal_EIS_WSE_fitness_1(exp_data_dict, ECM_paras_list):
    """
    Function
        Calculate Weighted Square Error, two inputs
        It is the same as the calculation of Chi-Square Error
    :param
        exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试频率              'f',         fre_list
                    测试所得复数阻抗       'z_raw',     z_raw_complex_list
        fre_list:
            list(float, float,...)
            [0.002, 0.05, 0.8, ..., 1000000]
            These frequency are read from raw experiment data, like DTA files
        w_list:
            list(float)
            w, angle frequency, w = 2 * pi * frequency
        ECM_num:
            int
            Each number corresponds to a specific ECM
                Circuit(ECM) No.    CDC             Function
                0                   R(CR)           RaCRb, Simplified Randles Cell
                0                                   R0aC0R1b

                1                   R(QR)           RaQRb
                1                   R(QR)           R0aQ0R1b

                2                   R(QR)(QR)       RaQRbaQRb
                2                   R(QR)(QR)       R0aQ0R1baQ1R2b

                3                   R(QR(LR))       RaQRaLRbb
                3                   R(QR(LR))       R0aQ0R1aL0R2bb

                4                   R(Q(RW))        RaQaRWbb
                4                   R(Q(RW))        R0aQ0aR1W0bb

                5                   R(QR)(QR)W      RaQRbaQRbW
                5                   R(QR)(QR)W      R0aQ0R1baQ1R2bW0

                6                   R(QR)(Q(RW))    RaQRbaQaRWbb
                6                   R(QR)(Q(RW))    R0aQ0R1baQ1aR2W0bb

                7                   R(QR)W          RaQRbW
                7                   R(QR)W          R0aQ0R1bW0

                8                   R(Q(RW))Q       RaQaRWbbQ
                8                   R(Q(RW))Q       R0aQ0aR1W0bbQ1

                9                   R(Q(R(QR)))     RaQaRaQRbbb
                9                   R(Q(R(QR)))     R0aQ0aR1aQ1R2bbb
        ECM_paras_list
            list(float)
            Store the values of elements in the ECM
            Store order of values are the same as the [circuit function's name], like:
                RaCRb(w, R0, R1, C0) ==> [R0, R1, C0]
                RaQRb(w, R0, Q0_pair, R1) ==> [R0, Q0_pair_q, Q0_pair_n, R1]
        z_raw_complex_list:
            list(complex)
            This list is read from raw experiment data, like DTA files
        z_pre_complex_list:
            list(complex)
            Take the calculated circuit elements' values into the ECM to get the simulated Impedance
    :return:
        z_weighted_square_err
            refer:
                paper:
			1-Study on Genetic Algorithm in Estimating the Initial Value of EIS Equivalent Circuit, eq 5
			2-A hybrid genetic algorithm for the fitting of ovo_models to electrochemical impedance data, eq 3
            float
            The square of the difference between the raw-Z and simulate-Z
            Weight, W = 1 / ||Z||, Z (Raw Impedance)
            z_weighted_square_err = W1 * ( (raw_Z1.real - sim_Z1.real) ** 2 + (raw_Z1.imag - sim_Z1.imag) ** 2 )
                          +W2 * ( (raw_Z2.real - sim_Z2.real) ** 2 + (raw_Z2.imag - sim_Z2.imag) ** 2 )
                            +...
    """
    fre_list = exp_data_dict['f']
    ECM_num = exp_data_dict['ecm_num']
    if 'z_raw' in exp_data_dict.keys():
        z_raw_complex_list = exp_data_dict['z_raw']
    elif 'z_sim' in exp_data_dict.keys():
        z_raw_complex_list = exp_data_dict['z_sim']

    z_sim_complex_list = cal_EIS_sim_complex_list(fre_list, ECM_num, ECM_paras_list)

    """
    Refer:
        ZSimpWin: Tech Note 37: Least Squares Fit Formulation --> 3. Definition of the chi-squared --> Eq 10中
        对Chi-Square的定义
    """
    # Calculate the Weighted Square of difference between the raw-Z and simulated-Z
    # the same as the calculation of Chi-Square Error
    def cal_w(raw_Z):
        return 1 / (raw_Z.real ** 2 + raw_Z.imag ** 2)

    # N测试的阻抗点数，如赖师兄N=30
    N = len(fre_list)
    # M为等效电路中需要拟合的参数数目，CPE元件包含两个参数Y和n
    M = len(ECM_paras_list)
    # v为系统的自由度，严格讲 v = N - M - 1，此处为了和ZSimpWIn一样，令 v = N - M
    v = N - M

    z_weighted_square_err = sum([cal_w(rz) * ((rz.real - sz.real)**2 + (rz.imag - sz.imag)**2)
                                 for rz, sz in zip(z_raw_complex_list, z_sim_complex_list)])
    z_weighted_square_err /= v
    return z_weighted_square_err