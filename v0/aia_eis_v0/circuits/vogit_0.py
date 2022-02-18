import sys
sys.path.append('../')
import numpy as np
import math
import copy
import os

from circuits.circuit_pack import aRCb
from circuits.elements import ele_C, ele_L
from loa.l_m.l_m_0 import Levenberg_Marquart_0, vogit_obj_fun_0, vogit_obj_fun_1
from IS.IS import IS_0
from IS.IS_criteria import cal_residual, cal_ChiSquare_pointWise_0
from utils.file_utils.pickle_utils import pickle_file
from utils.visualize_utils.impedance_plots import nyquist_multiPlots_1, nyquist_plot_1

"""
Special ECM
    Vogit
        是否有可能在M个RC的基础上 再包含N个RL？？
            目前MS的Lin-KK 和 Impedance 都没考虑，暂时不管，不考虑太复杂的情况
    maybe some other ECMs, like transmission line, etc.
"""

class Vogit_2:
    """
    Refer
        papers:
            paper1: A Linear Kronig-Kramers Transform Test for Immittance Data Validation
            paper0: A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
    Note:
        Vogit 最基本的电路为
            Rs-Ls-M*(RC)-[Cs]
            Ls: inductive effects are considered byadding an additional inductivity [1]
            Cs:
                option to add a serial capacitance that helps validate data with no low-frequency intercept
                due to their capacitive nature an additional capacityis added to the ECM.

            1- 只考虑 complex / imag / real -fit中的complex-fit
            2- 三种加权方式只考虑 modulus
            3- add Capacity / Inductance 中 只考虑 add Capacity
    Version:
        2: 之前的Vogit中没有加入电感L，在这一版本中加上
    """

    def __init__(self, impSpe, add_C=False):
        """
        因为Vogit是一个measurement model，所以使用vogit之前一定会传进来一个IS
        :param
            impSpe: IS cls
            M: int
                number of (RC)
            w: list(float)
            RC_para_list:[
                [R0, C0],
                [R1, C1],
                ...
                [Rm-1, Cm-1],
            ]
            Rs: float
            add_C: Bool
        """
        self.impSpe = impSpe
        self.w_arr = self.impSpe.w_arr
        self.M = 1

        """
        Paper1: As a rule of thumb we can conclude that, for the single fit and transformation, the v range should be
                equal to the inverse w range with a distribution of 6 or 7 Tcs per decade. 在这里再稍微取的更大一些 8 * decades
        """
        self.M_max = int(math.log10(self.w_arr.max() / self.w_arr.min())) * 8
        self.add_C = add_C

    def calc_timeConstant(self):
        """
        timeConstant = tao = R * C
        Refer:
            A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
                2.2. Distribution of Time Constants Eq 10-12
        :return:
        """
        sorted_w_arr = np.sort(copy.deepcopy(self.w_arr))  # small --> big number
        w_min, w_max = sorted_w_arr[0], sorted_w_arr[-1]

        # Time Constant τ 用 tao表示
        tao_min = 1 / w_max
        tao_max = 1 / w_min

        tao_list = []
        if self.M == 1:
            tao_list.append(tao_min)
        elif self.M == 2:
            tao_list.extend([tao_min, tao_max])
        elif self.M > 2:
            tao_list.append(tao_min)
            K = self.M - 1
            for i in range(1, K):
                tao = 10 ** (math.log10(tao_min) + i * math.log10(tao_max / tao_min) / (self.M - 1))
                tao_list.append(tao)
            tao_list.append(tao_max)
        self.tao_arr = np.array(tao_list)

    def init_para(self):
        # refer the initialization of <impedance.py>
        self.Rs = min(np.real(self.impSpe.z_arr))
        self.Ls = 1e-3
        self.M_R_arr = [(max(np.real(self.impSpe.z_arr)) - min(np.real(self.impSpe.z_arr))) / self.M for i in range(self.M)]

        if self.add_C:
            self.Cs = 1e-3

        self.calc_timeConstant()

    def init_para_0(self):
        """
        1-由于时间常数Tao已经确定，Tao = Ri * Ci，所以只需要初始化M个Ri，i = 0，1，2，。。。，M-
        2-根据paper《A Linear Kronig-Kramers Transform Test for Immittance Data Validation》 fig 6的结果，拟合得到的Ri大多数情况
        下是一正一负，所以初始Ri为：R0=1，R1=-1，R2=1，R3=-1，。。。
        :return:
        """
        # 第一次初始化RC M = 1
        if self.RC_para_list is None:
            self.calc_timeConstant()
            Ri_list = []
            for i in range(self.M):
                # even number: 0,2,4, Ri = 1
                if i % 2 == 0:
                    Ri = 1.0
                # odd number: 1,3,5, Ri = -1.0
                else:
                    Ri = -1.0
                Ri_list.append(Ri)
            self.RC_para_list = [[Ri, self.tao_arr[i] / Ri] for i, Ri in enumerate(Ri_list)]
            self.Rs = self.cal_Rs()
        else:
            # M > 1 , 如果M增加，保留之前的拟合结果，只初始化新加的RC
            self.calc_timeConstant()
            RC_para_existed_len = len(self.RC_para_list)
            add_R_list = []
            for i in range(RC_para_existed_len, self.M):
                # even number: 0,2,4, Ri = 1
                if i % 2 == 0:
                    R = 1.0
                # odd number: 1,3,5, Ri = -1.0
                else:
                    R = -1.0
                add_R_list.append(R)

            old_RC_para_list = copy.deepcopy(self.RC_para_list)
            self.RC_para_list = []
            # 之前的R
            for i, RC in enumerate(old_RC_para_list):
                self.RC_para_list.append([RC[0], self.tao_arr[i] / RC[0]])
            # 新加的R
            for i, R in enumerate(add_R_list):
                self.RC_para_list.append([R, self.tao_arr[RC_para_existed_len + i] / R])
            self.Rs = self.cal_Rs()

    # def connect_circuit(self):
    #     """
    #     默认 Vogit = Rs + (RC)_0 + (RC)_1 + ... + (RC)_m-1
    #     :return:
    #     """
    #     pass

    def cal_Rs(self):
        """
        根据 paper1-Eq7 计算 Rs
        :return:
        """
        z_arr = self.impSpe.z_arr
        weight_arr = np.array([1 / (z.real ** 2 + z.imag ** 2) for z in z_arr])
        Rs = 0.0
        for i, weight in enumerate(weight_arr):
            res_in_square_bracket = z_arr[i].real - \
                                    sum([self.RC_para_list[k][0] / (1 + (self.w_arr[i] * self.tao_arr[k]) ** 2) for k in
                                         range(self.M)])
            Rs += weight * res_in_square_bracket
        Rs /= weight_arr[:-1].sum()
        return Rs

    def update_para(self, tmp_para_arr):
        """
        R_list / R_arr:
            [Rs, R0, R1, ..., R_M-1]
            优化算法迭代产生新的阻抗值，替换原来的R
            同时更新对应的电容C
        :return:
        """
        if self.OA_obj_fun_mode == 'imag':
            pass
        elif (self.OA_obj_fun_mode == 'real') or (self.OA_obj_fun_mode == 'both'):
            # para_arr = [*Rs*, *Ls*, (*Cs*), R0, R1, R2, ..., R_M-1]
            self.Rs = tmp_para_arr[0]
            self.Ls = tmp_para_arr[1]
            RC_start_index = 2
            if self.add_C:
                # para_arr = [*Rs*, *Ls*, *Cs*, R0, R1, R2, ..., R_M-1]
                self.Cs = tmp_para_arr[RC_start_index]
                RC_start_index = 3
            self.M_R_arr = tmp_para_arr[RC_start_index:]

    def update_u(self):
        """
        refer paper0-eq21
        :return:
        """
        positive_R_list = []
        negtive_R_list = []
        for R in self.M_R_arr:
            if R >= 0:
                positive_R_list.append(R)
            elif R < 0:
                negtive_R_list.append(R)
        self.u = 1 - abs(sum(negtive_R_list)) / sum(positive_R_list)

    def lin_KK(self, OA=Levenberg_Marquart_0, OA_obj_fun_mode='both', OA_obj_fun_weighting_type='modulus',
               save_iter=False, u_optimum=0.85, manual_M=None):
        self.OA_obj_fun_mode = OA_obj_fun_mode
        self.OA_obj_fun_weighting_type = OA_obj_fun_weighting_type

        if manual_M is not None:
            self.M = manual_M

        self.init_para()
        self.update_u()

        # init Levenberg_Marquardt
        # OA: Optimization Algorithm
        oa = OA(impSpe=self.impSpe,
                obj_fun=vogit_obj_fun_1,
                obj_fun_mode=OA_obj_fun_mode,
                obj_fun_weighting_type=OA_obj_fun_weighting_type,
                iter_max=500,
                add_C=self.add_C)

        while (self.u >= u_optimum) and (self.M <= self.M_max):
            if OA_obj_fun_mode == 'imag':
                # oa.get_initial_para_arr(para_arr=np.array([RC[0] for RC in self.RC_para_list]))
                pass
            elif (OA_obj_fun_mode == 'real') or (OA_obj_fun_mode == 'both'):
                if self.add_C:
                    para_arr = np.array([self.Rs, self.Ls, self.Cs] + [R for R in self.M_R_arr])
                    print('Para into OA:', para_arr)
                    oa.get_initial_para_arr(para_arr)
                else:
                    oa.get_initial_para_arr(para_arr=np.array([self.Rs, self.Ls] + [R for R in self.M_R_arr]))

            oa.iterate(timeConstant_arr=self.tao_arr)
            tmp_para_arr = oa.para_arr  # N * 1
            print('Para out from OA:', tmp_para_arr)

            # update R
            self.update_para(tmp_para_arr)

            # update u
            self.update_u()

            if manual_M is not None:
                chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                self.chiSquare_list = [chiSquare]
                self.chiSquare_real_list = [chiSquare_real]
                self.chiSquare_imag_list = [chiSquare_imag]
                self.real_residual_list = [real_residual_list]
                self.imag_residual_list = [imag_residual_list]
                break

            # The value of c (u_max) is a design parameter,
            # however from the author’s experience c = 0.85 has proven to be an excellent choice.
            if (self.u >= u_optimum) and (self.M <= self.M_max):  # underfitting
                # 打印输出、保存迭代的中间结果
                print('M=', self.M, 'u=', self.u)
                # print('M=', self.M, 'u=', self.u, 'Rs=', self.Rs, '(RC)s=', self.RC_para_list)
                if save_iter == True:
                    if self.M == 1:
                        self.M_list = [1]
                        self.u_list = [copy.deepcopy(self.u)]
                        self.Rs_list = [copy.deepcopy(self.Rs)]
                        self.Ls_list = [copy.deepcopy(self.Ls)]
                        self.R_pack_list = [copy.deepcopy(self.M_R_arr)]

                        chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                        self.chiSquare_list = [chiSquare]
                        self.chiSquare_real_list = [chiSquare_real]
                        self.chiSquare_imag_list = [chiSquare_imag]
                        self.real_residual_list = [real_residual_list]
                        self.imag_residual_list = [imag_residual_list]

                    elif self.M > 1:
                        self.M_list.append(copy.deepcopy(self.M))
                        self.u_list.append(copy.deepcopy(self))
                        self.Rs_list.append(copy.deepcopy(self.Rs))
                        self.R_pack_list.append(copy.deepcopy(self.M_R_arr))

                        chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                        self.chiSquare_list.append(chiSquare)
                        self.chiSquare_real_list.append(chiSquare_real)
                        self.chiSquare_imag_list.append(chiSquare_imag)
                        self.real_residual_list.append(real_residual_list)
                        self.imag_residual_list.append(imag_residual_list)
                        print('M=', self.M, 'u=', self.u, chiSquare)

                self.M += 1
                self.init_para()
            else:
                print('M=', self.M, 'u=', self.u)
                break

    def simulate_Z(self):
        """
        使用拟合的各种参数：Rs + M * RC
        :return:
        """
        self.z_sim_arr = np.empty(shape=(self.M, self.impSpe.z_arr.shape[0]), dtype=complex)
        for i in range(self.M):
            R = self.M_R_arr[i]
            tao = self.tao_arr[i]
            tmp_z_sim_list = [aRCb(w, R, tao/R) for w in self.w_arr]
            self.z_sim_arr[i, :] = np.array(tmp_z_sim_list)

        L_Z_sim_arr = np.array([ele_L(w, self.Ls) for w in self.w_arr]).reshape((1, self.w_arr.size))
        if self.add_C:
            # self.z_sim_arr[-1, :] = [ele_C(w, self.C) for w in self.w_arr]
            c_z_arr = np.array([ele_C(w, self.Cs) for w in self.w_arr]).reshape((1, self.w_arr.shape[0]))
            self.z_sim_arr = np.concatenate((self.z_sim_arr, L_Z_sim_arr, c_z_arr), axis=0)
        else:
            self.z_sim_arr = np.concatenate((self.z_sim_arr, L_Z_sim_arr), axis=0)

        self.z_sim_arr = self.z_sim_arr.sum(axis=0)
        self.z_sim_arr += self.Rs

    def cal_various_criteria(self):
        """
        calculate
            weight = 1 / (z.real ** 2 + z.imag ** 2)
            X^2, defined in paper0 - Eq 15
                在这里没有办法计算ZSimpWin中的X^2，因为 过程ECM未知 == 代求参数的数量未知 --》 系统的自由度无法确定
                这里的X^2计算如下：
                    N = data points
                    X^2 = (1/N) * ∑{ weight * [(Z(w)i.real - Zi.real) ** 2 + (Z(w)i.imag - Zi.imag) **2] }
            X^2_imag, defined in paper0 - Eq 20
            X^2_real, 模仿 X^2_imag 的计算
            🔺Real, defined in paper0 - Eq 15
            🔺Imag, defined in paper0 - Eq 16
        :return:
        """
        chiSquare = 0.0
        chiSquare_real = 0.0
        chiSquare_imag = 0.0
        imag_residual_list = []
        real_residual_list = []

        self.simulate_Z()
        z_arr = self.impSpe.z_arr

        modulus_weight_list = [1 / (z.real ** 2 + z.imag ** 2) for z in z_arr]

        for weight, z_sim, z in zip(modulus_weight_list, self.z_sim_arr, z_arr):
            real_residual_list.append(math.sqrt(weight) * (z.real - z_sim.real))
            imag_residual_list.append(math.sqrt(weight) * (z.imag - z_sim.imag))

            chiSquare_real += (1 / z_arr.shape[0]) * weight * ((z_sim.real - z.real) ** 2)
            chiSquare_imag += (1 / z_arr.shape[0]) * weight * ((z_sim.imag - z.imag) ** 2)

            chiSquare += chiSquare_imag + chiSquare_real
        return chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list

    def save2pkl(self, fp, fn):
        pickle_file(obj=self, fn=fn, fp=fp)

# ---------------------------------- Test Vogit_2 on Lin-KK-Ex1_LIB_time_invariant ----------------------------------
# 1- load data
lib_res_fp = '../plugins_test/jupyter_code/rbp_files/2/example_data_sets/LIB_res'
ex1_data_dict = np.load(os.path.join(lib_res_fp, 'Ex1_LIB_time_invariant_res.npz'))
ex1_z_arr = ex1_data_dict['z_arr']
ex1_f_arr = ex1_data_dict['fre']
ex1_z_MS_sim_arr = ex1_data_dict['z_sim']
ex1_real_residual_arr = ex1_data_dict['real_residual']
ex1_imag_residual_arr = ex1_data_dict['imag_residual']

ex1_IS = IS_0()
ex1_IS.raw_z_arr = ex1_z_arr
ex1_IS.exp_area = 1.0
ex1_IS.z_arr = ex1_z_arr
ex1_IS.fre_arr = ex1_f_arr
ex1_IS.w_arr = ex1_IS.fre_arr * 2 * math.pi

ex1_vogit = Vogit_2(impSpe=ex1_IS, add_C=True)
OA_obj_fun_mode = 'both'
ex1_vogit.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode, save_iter=False, u_optimum=0.85, manual_M=30)
# ex1_vogit.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode, save_iter=False, u_optimum=0.85, manual_M=None)

# compare nyquist plots of MS-Lin-KK and Mine
ex1_z_MS_sim_list = ex1_z_MS_sim_arr.tolist()
ex1_vogit.simulate_Z()
z_pack_list = [ex1_z_arr.tolist(), ex1_z_MS_sim_list, ex1_vogit.z_sim_arr.tolist()]
nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[-0.015, 0.045], y_lim=[0, 0.02], plot_label_list=['Ideal IS', 'MS-Fit','Mine-Fit'])
# nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[0., 10], y_lim=[0, 20], plot_label_list=['Ideal IS', 'MS-Fit','Mine-Fit'])
# nyquist_plot_1(z_list=ex1_vogit.z_sim_arr, x_lim=[-10.015, 10.045], y_lim=[-10, 150.02])
# ---------------------------------- Test Vogit_1 on Lin-KK-Ex1_LIB_time_invariant ----------------------------------

class Vogit_1:
    """
    Refer
        papers:
            paper1: A Linear Kronig-Kramers Transform Test for Immittance Data Validation
            paper0: A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
    Note:
        Vogit 最基本的电路为
            Rs-Ls-M*(RC)-[Cs]
            Ls: inductive effects are considered byadding an additional inductivity [1]
            Cs:
                option to add a serial capacitance that helps validate data with no low-frequency intercept
                due to their capacitive nature an additional capacityis added to the ECM.

            1- 只考虑 complex / imag / real -fit中的complex-fit
            2- 三种加权方式只考虑 modulus
            3- add Capacity / Inductance 中 只考虑 add Capacity
    """
    
    def __init__(self, impSpe, add_C=False):
        """
        因为Vogit是一个measurement model，所以使用vogit之前一定会传进来一个IS
        :param
            impSpe: IS cls
            M: int
                number of (RC)
            w: list(float)
            RC_para_list:[
                [R0, C0],
                [R1, C1],
                ...
                [Rm-1, Cm-1],
            ]
            Rs: float
            add_C: Bool
        """
        self.impSpe = impSpe
        self.w_arr = self.impSpe.w_arr
        self.M = 1
        
        """
        Paper1: As a rule of thumb we can conclude that, for the single fit and transformation, the v range should be
                equal to the inverse w range with a distribution of 6 or 7 Tcs per decade. 在这里再稍微取的更大一些 8 * decades
        """
        self.M_max = int(math.log10(self.w_arr.max() / self.w_arr.min())) * 8
        
        self.Rs = 1e-2
        self.add_L = 1e-3
        self.RC_para_list = None

        self.add_C = add_C
        if self.add_C:
            self.C = 1e-3
    
    def calc_timeConstant(self):
        """
        timeConstant = tao = R * C
        Refer:
            A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
                2.2. Distribution of Time Constants Eq 10-12
        :return:
        """
        sorted_w_arr = np.sort(copy.deepcopy(self.w_arr))  # small --> big number
        w_min, w_max = sorted_w_arr[0], sorted_w_arr[-1]
        
        # Time Constant τ 用 tao表示
        tao_min = 1 / w_max
        tao_max = 1 / w_min
        
        tao_list = []
        if self.M == 1:
            tao_list.append(tao_min)
        elif self.M == 2:
            tao_list.extend([tao_min, tao_max])
        elif self.M > 2:
            tao_list.append(tao_min)
            K = self.M - 1
            for i in range(1, K):
                tao = 10 ** (math.log10(tao_min) + i * math.log10(tao_max / tao_min) / (self.M - 1))
                tao_list.append(tao)
            tao_list.append(tao_max)
        self.tao_arr = np.array(tao_list)
    
    # def init_para(self):
        # refer the initialization of impedance
        # self.calc_timeConstant()
        # self.Rs = min(np.real(self.impSpe.z_arr))
        # R_list = [(max(np.real(self.impSpe.z_arr)) - min(np.real(self.impSpe.z_arr))) / self.M for i in range(self.M)]
        # self.RC_para_list = [[Ri, self.tao_arr[i] / Ri] for i, Ri in enumerate(R_list)]
    
    def init_para_0(self):
        """
        1-由于时间常数Tao已经确定，Tao = Ri * Ci，所以只需要初始化M个Ri，i = 0，1，2，。。。，M-
        2-根据paper《A Linear Kronig-Kramers Transform Test for Immittance Data Validation》 fig 6的结果，拟合得到的Ri大多数情况
        下是一正一负，所以初始Ri为：R0=1，R1=-1，R2=1，R3=-1，。。。
        :return:
        """
        # 第一次初始化RC M = 1
        if self.RC_para_list is None:
            self.calc_timeConstant()
            Ri_list = []
            for i in range(self.M):
                # even number: 0,2,4, Ri = 1
                if i % 2 == 0:
                    Ri = 1.0
                # odd number: 1,3,5, Ri = -1.0
                else:
                    Ri = -1.0
                Ri_list.append(Ri)
            self.RC_para_list = [[Ri, self.tao_arr[i] / Ri] for i, Ri in enumerate(Ri_list)]
            self.Rs = self.cal_Rs()
        else:
            # M > 1 , 如果M增加，保留之前的拟合结果，只初始化新加的RC
            self.calc_timeConstant()
            RC_para_existed_len = len(self.RC_para_list)
            add_R_list = []
            for i in range(RC_para_existed_len, self.M):
                # even number: 0,2,4, Ri = 1
                if i % 2 == 0:
                    R = 1.0
                # odd number: 1,3,5, Ri = -1.0
                else:
                    R = -1.0
                add_R_list.append(R)
            
            old_RC_para_list = copy.deepcopy(self.RC_para_list)
            self.RC_para_list = []
            # 之前的R
            for i, RC in enumerate(old_RC_para_list):
                self.RC_para_list.append([RC[0], self.tao_arr[i] / RC[0]])
            # 新加的R
            for i, R in enumerate(add_R_list):
                self.RC_para_list.append([R, self.tao_arr[RC_para_existed_len + i] / R])
            self.Rs = self.cal_Rs()
    
    # def connect_circuit(self):
    #     """
    #     默认 Vogit = Rs + (RC)_0 + (RC)_1 + ... + (RC)_m-1
    #     :return:
    #     """
    #     pass
    
    def cal_Rs(self):
        """
        根据 paper1-Eq7 计算 Rs
        :return:
        """
        z_arr = self.impSpe.z_arr
        weight_arr = np.array([1 / (z.real ** 2 + z.imag ** 2) for z in z_arr])
        Rs = 0.0
        for i, weight in enumerate(weight_arr):
            res_in_square_bracket = z_arr[i].real - \
                                    sum([self.RC_para_list[k][0] / (1 + (self.w_arr[i] * self.tao_arr[k]) ** 2) for k in
                                         range(self.M)])
            Rs += weight * res_in_square_bracket
        Rs /= weight_arr[:-1].sum()
        return Rs
    
    def update_para(self, tmp_para_arr):
        """
        R_list / R_arr:
            [Rs, R0, R1, ..., R_M-1]
            优化算法迭代产生新的阻抗值，替换原来的R
            同时更新对应的电容C
        :return:
        """
        if self.OA_obj_fun_mode == 'imag':
            pass
            # C_list = [tao / R for tao, R in zip(self.tao_arr, tmp_para_arr)]
            # self.RC_para_list = [[R, C] for R, C in zip(tmp_para_arr, C_list)]
        elif (self.OA_obj_fun_mode == 'real') or (self.OA_obj_fun_mode == 'both'):
            # para_arr = [*Rs*, *Ls*, (*Cs*), R0, R1, R2, ..., R_M-1]
            self.Rs = tmp_para_arr[0]
            C_start_index = 1
            if self.add_C:
                # para_arr = [*Rs*, *Ls*, *Cs*, R0, R1, R2, ..., R_M-1]
                self.C = tmp_para_arr[1]
                C_start_index = 2
            C_list = [tao / R for tao, R in zip(self.tao_arr, tmp_para_arr[C_start_index:])]
            self.RC_para_list = [[R, C] for R, C in zip(tmp_para_arr[C_start_index:], C_list)]
    
    def update_u(self):
        """
        refer paper0-eq21
        :return:
        """
        positive_R_list = []
        negtive_R_list = []
        for RC_list in self.RC_para_list:
            R = RC_list[0]
            if R >= 0:
                positive_R_list.append(R)
            elif R < 0:
                negtive_R_list.append(R)
        self.u = 1 - abs(sum(negtive_R_list)) / sum(positive_R_list)
    
    def cal_Zimag_residual(self):
        pass
    
    def lin_KK(self, OA=Levenberg_Marquart_0, OA_obj_fun_mode='both', OA_obj_fun_weighting_type='modulus',
               save_iter=False, u_optimum=0.85, manual_M=None):
        self.OA_obj_fun_mode = OA_obj_fun_mode
        self.OA_obj_fun_weighting_type = OA_obj_fun_weighting_type
        
        if manual_M is not None:
            self.M = manual_M
        
        self.init_para()
        self.update_u()
        
        # init Levenberg_Marquardt
        # OA: Optimization Algorithm
        oa = OA(impSpe=self.impSpe,
                obj_fun=vogit_obj_fun_0,
                # obj_fun=cal_ChiSquare_pointWise_0,
                obj_fun_mode=OA_obj_fun_mode,
                obj_fun_weighting_type=OA_obj_fun_weighting_type,
                iter_max=1000,
                add_C=True)
        
        while (self.u >= u_optimum) and (self.M <= self.M_max):
            if OA_obj_fun_mode == 'imag':
                oa.get_initial_para_arr(para_arr=np.array([RC[0] for RC in self.RC_para_list]))
            elif (OA_obj_fun_mode == 'real') or (OA_obj_fun_mode == 'both'):
                if self.add_C:
                    para_arr = np.array([self.Rs] + [self.C] + [RC[0] for RC in self.RC_para_list])
                    oa.get_initial_para_arr(para_arr)
                else:
                    oa.get_initial_para_arr(para_arr=np.array([self.Rs] + [RC[0] for RC in self.RC_para_list]))
            
            """
            oa.iterate 传入z_arr, w_arr, tao_arr的目的：
                z_arr是观测数据，和vogit的拟合数据对比来计算残差
                w_arr, tao_arr用来确定Vogit模型的
                在L-M中，R每变动一次，C要由 C = tao / R 计算更新
                只有RC中的R是待求的未知数
            """
            oa.iterate(timeConstant_arr=self.tao_arr)
            tmp_para_arr = oa.para_arr  # N * 1
            
            # update RC
            self.update_para(tmp_para_arr)
            
            # update u
            self.update_u()
            
            if manual_M is not None:
                chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                self.chiSquare_list = [chiSquare]
                self.chiSquare_real_list = [chiSquare_real]
                self.chiSquare_imag_list = [chiSquare_imag]
                self.real_residual_list = [real_residual_list]
                self.imag_residual_list = [imag_residual_list]
                break
            
            # The value of c (u_max) is a design parameter,
            # however from the author’s experience c = 0.85 has proven to be an excellent choice.
            if (self.u >= u_optimum) and (self.M <= self.M_max):  # underfitting
                # 打印输出、保存迭代的中间结果
                print('M=', self.M, 'u=', self.u)
                # print('M=', self.M, 'u=', self.u, 'Rs=', self.Rs, '(RC)s=', self.RC_para_list)
                if save_iter == True:
                    if self.M == 1:
                        self.M_list = [1]
                        self.u_list = [copy.deepcopy(self.u)]
                        self.Rs_list = [copy.deepcopy(self.Rs)]
                        self.RC_para_pack_list = [copy.deepcopy(self.RC_para_list)]
                        
                        chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                        self.chiSquare_list = [chiSquare]
                        self.chiSquare_real_list = [chiSquare_real]
                        self.chiSquare_imag_list = [chiSquare_imag]
                        self.real_residual_list = [real_residual_list]
                        self.imag_residual_list = [imag_residual_list]
                    
                    elif self.M > 1:
                        self.M_list.append(copy.deepcopy(self.M))
                        self.u_list.append(copy.deepcopy(self))
                        self.Rs_list.append(copy.deepcopy(self.Rs))
                        self.RC_para_pack_list.append(copy.deepcopy(self.RC_para_list))
                        
                        chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                        self.chiSquare_list.append(chiSquare)
                        self.chiSquare_real_list.append(chiSquare_real)
                        self.chiSquare_imag_list.append(chiSquare_imag)
                        self.real_residual_list.append(real_residual_list)
                        self.imag_residual_list.append(imag_residual_list)
                        print('M=', self.M, 'u=', self.u, chiSquare)
                
                self.M += 1
                self.init_para()
            else:
                print('M=', self.M, 'u=', self.u)
                break
    
    def simulate_Z(self):
        """
        使用拟合的各种参数：Rs + M * RC
        :return:
        """
        self.z_sim_arr = np.empty(shape=(self.M, self.impSpe.z_arr.shape[0]), dtype=complex)
        for i in range(self.M):
            R, C0 = self.RC_para_list[i]
            tmp_z_sim_list = [aRCb(w, R, C0) for w in self.w_arr]
            self.z_sim_arr[i, :] = np.array(tmp_z_sim_list)
        
        if self.add_C:
            # self.z_sim_arr[-1, :] = [ele_C(w, self.C) for w in self.w_arr]
            c_z_arr = np.array([ele_C(w, self.C) for w in self.w_arr]).reshape((1, self.w_arr.shape[0]))
            self.z_sim_arr = np.concatenate((self.z_sim_arr, c_z_arr), axis=0)
            self.z_sim_arr = self.z_sim_arr.sum(axis=0)
        else:
            self.z_sim_arr = self.z_sim_arr.sum(axis=0)
        
        self.z_sim_arr += self.Rs
    
    def cal_various_criteria(self):
        """
        calculate
            weight = 1 / (z.real ** 2 + z.imag ** 2)
            X^2, defined in paper0 - Eq 15
                在这里没有办法计算ZSimpWin中的X^2，因为 过程ECM未知 == 代求参数的数量未知 --》 系统的自由度无法确定
                这里的X^2计算如下：
                    N = data points
                    X^2 = (1/N) * ∑{ weight * [(Z(w)i.real - Zi.real) ** 2 + (Z(w)i.imag - Zi.imag) **2] }
            X^2_imag, defined in paper0 - Eq 20
            X^2_real, 模仿 X^2_imag 的计算
            🔺Real, defined in paper0 - Eq 15
            🔺Imag, defined in paper0 - Eq 16
        :return:
        """
        chiSquare = 0.0
        chiSquare_real = 0.0
        chiSquare_imag = 0.0
        imag_residual_list = []
        real_residual_list = []
        
        self.simulate_Z()
        z_arr = self.impSpe.z_arr
        
        modulus_weight_list = [1 / (z.real ** 2 + z.imag ** 2) for z in z_arr]
        
        for weight, z_sim, z in zip(modulus_weight_list, self.z_sim_arr, z_arr):
            real_residual_list.append(math.sqrt(weight) * (z.real - z_sim.real))
            imag_residual_list.append(math.sqrt(weight) * (z.imag - z_sim.imag))
            
            chiSquare_real += (1 / z_arr.shape[0]) * weight * ((z_sim.real - z.real) ** 2)
            chiSquare_imag += (1 / z_arr.shape[0]) * weight * ((z_sim.imag - z.imag) ** 2)
            
            chiSquare += chiSquare_imag + chiSquare_real
        return chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list
    
    def save2pkl(self, fp, fn):
        pickle_file(obj=self, fn=fn, fp=fp)

# ---------------------------------- Test Vogit_1 on Lin-KK-Ex1_LIB_time_invariant ----------------------------------
# 1- load data
# lib_res_fp = '../plugins_test/jupyter_code/rbp_files/2/example_data_sets/LIB_res'
# ex1_data_dict = np.load(os.path.join(lib_res_fp, 'Ex1_LIB_time_invariant_res.npz'))
# ex1_z_arr = ex1_data_dict['z_arr']
# ex1_f_arr = ex1_data_dict['fre']
# ex1_z_MS_sim_arr = ex1_data_dict['z_sim']
# ex1_real_residual_arr = ex1_data_dict['real_residual']
# ex1_imag_residual_arr = ex1_data_dict['imag_residual']
#
# ex1_IS = IS_0()
# ex1_IS.raw_z_arr = ex1_z_arr
# ex1_IS.exp_area = 1.0
# ex1_IS.z_arr = ex1_z_arr
# ex1_IS.fre_arr = ex1_f_arr
# ex1_IS.w_arr = ex1_IS.fre_arr * 2 * math.pi
#
# ex1_vogit = Vogit_1(impSpe=ex1_IS, add_C=False)
# # ex1_vogit = Vogit_1(impSpe=ex1_IS, add_C=True)
# OA_obj_fun_mode = 'both'
# ex1_vogit.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode, save_iter=True, u_optimum=0.85, manual_M=None)
#
# # compare nyquist plots of MS-Lin-KK and Mine
# ex1_z_MS_sim_list = ex1_z_MS_sim_arr.tolist()
# z_pack_list = [ex1_z_arr.tolist(), ex1_z_MS_sim_list, ex1_vogit.z_sim_arr.tolist()]
# nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[0.015, 0.045], y_lim=[0, 0.02], plot_label_list=['Ideal IS', 'MS-Fit','Mine-Fit'])
# nyquist_plot_1(z_list=ex1_vogit.z_sim_arr, x_lim=[-10.015, 10.045], y_lim=[-10, 150.02])
# ---------------------------------- Test Vogit_1 on Lin-KK-Ex1_LIB_time_invariant ----------------------------------

# ------------------------------------- Test Vogit_1 on my simulated/ecm_001/ -------------------------------------
# impS = IS_0()
# dpfc_src\datasets\goa_datasets\simulated\ecm_001\2020_07_04_sim_ecm_001_pickle.file
# impS.read_from_simPickle(fp='../datasets/goa_datasets/simulated/ecm_001/',
#                          fn='2020_07_04_sim_ecm_001_pickle.file')
# vogit_1 = Vogit_1(impSpe=impS, add_C=True)

# OA_obj_fun_mode = 'both'
# print(OA_obj_fun_mode)
# vogit_1.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode)
# print('M=', vogit_1.M, 'u=',vogit_1.u, 'Rs=',vogit_1.Rs,'(RC)s=',vogit_1.RC_para_list)
# python vogit_0.py
# ------------------------------------- Test Vogit_1 on my simulated/ecm_001/ -------------------------------------

class Vogit_0:
    """
    Refer
        papers:
            paper1: A Linear Kronig-Kramers Transform Test for Immittance Data Validation
            paper0: A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
    """
    def __init__(self, impSpe):
        """
        因为Vogit是一个measurement model，所以使用vogit之前一定会传进来一个IS
        :param
            impSpe: IS cls
            M: int
                number of (RC)
            w: list(float)
            RC_para_list:[
                [R0, C0],
                [R1, C1],
                ...
                [Rm-1, Cm-1],
            ]
            Rs: float
            add_C: Bool
            add_L: Bool
        """
        self.impSpe = impSpe
        self.w_arr = self.impSpe.w_arr
        self.M = 1

        """
        Paper1: As a rule of thumb we can conclude that, for the single fit and transformation, the v range should be 
                equal to the inverse w range with a distribution of 6 or 7 Tcs per decade. 在这里再稍微取的更大一些 8 * decades
        """
        self.M_max = int(math.log10(self.w_arr.max() / self.w_arr.min())) * 8

        self.Rs = 0
        self.RC_para_list = None

    def calc_timeConstant(self):
        """
        timeConstant = tao = R * C
        Refer:
            A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
                2.2. Distribution of Time Constants Eq 10-12
        :return:
        """
        sorted_w_arr = np.sort(copy.deepcopy(self.w_arr)) # small --> big number
        w_min, w_max = sorted_w_arr[0], sorted_w_arr[-1]

        # Time Constant τ 用 tao表示
        tao_min = 1 / w_max
        tao_max = 1 / w_min

        tao_list = []
        if self.M == 1:
            tao_list.append(tao_min)
        elif self.M == 2:
            tao_list.extend([tao_min, tao_max])
        elif self.M > 2:
            tao_list.append(tao_min)
            K = self.M - 1
            for i in range(1, K):
                tao = 10 ** (math.log10(tao_min) + i * math.log10(tao_max / tao_min) / (self.M - 1))
                tao_list.append(tao)
            tao_list.append(tao_max)
        self.tao_arr = np.array(tao_list)

    def init_para(self):
        # refer the initialization of impedance
        self.calc_timeConstant()
        self.Rs = min(np.real(self.impSpe.z_arr))
        R_list = [(max(np.real(self.impSpe.z_arr)) - min(np.real(self.impSpe.z_arr))) / self.M for i in range(self.M)]
        self.RC_para_list = [[Ri, self.tao_arr[i] / Ri] for i, Ri in enumerate(R_list)]
        
    def init_para_0(self):
        """
        1-由于时间常数Tao已经确定，Tao = Ri * Ci，所以只需要初始化M个Ri，i = 0，1，2，。。。，M-
        2-根据paper《A Linear Kronig-Kramers Transform Test for Immittance Data Validation》 fig 6的结果，拟合得到的Ri大多数情况
        下是一正一负，所以初始Ri为：R0=1，R1=-1，R2=1，R3=-1，。。。
        :return:
        """
        # 第一次初始化RC M = 1
        if self.RC_para_list is None:
            self.calc_timeConstant()
            Ri_list = []
            for i in range(self.M):
                # even number: 0,2,4, Ri = 1
                if i % 2 == 0:
                    Ri = 1.0
                # odd number: 1,3,5, Ri = -1.0
                else:
                    Ri = -1.0
                Ri_list.append(Ri)
            self.RC_para_list = [[Ri, self.tao_arr[i] / Ri] for i, Ri in enumerate(Ri_list)]
            self.Rs = self.cal_Rs()
        else:
            # M > 1 , 如果M增加，保留之前的拟合结果，只初始化新加的RC
            self.calc_timeConstant()
            RC_para_existed_len = len(self.RC_para_list)
            add_R_list = []
            for i in range(RC_para_existed_len, self.M):
                # even number: 0,2,4, Ri = 1
                if i % 2 == 0:
                    R = 1.0
                # odd number: 1,3,5, Ri = -1.0
                else:
                    R = -1.0
                add_R_list.append(R)

            old_RC_para_list = copy.deepcopy(self.RC_para_list)
            self.RC_para_list = []
            # 之前的R
            for i, RC in enumerate(old_RC_para_list):
                self.RC_para_list.append([RC[0], self.tao_arr[i] / RC[0]])
            # 新加的R
            for i, R in enumerate(add_R_list):
                self.RC_para_list.append([R, self.tao_arr[RC_para_existed_len+i] / R])
            self.Rs = self.cal_Rs()

    def connect_circuit(self):
        """
        默认 Vogit = Rs + (RC)_0 + (RC)_1 + ... + (RC)_m-1
        :return:
        """
        pass

    def cal_Rs(self):
        """
        根据 paper1-Eq7 计算 Rs
        :return:
        """
        z_arr = self.impSpe.z_arr
        weight_arr = np.array([1 / (z.real ** 2 + z.imag ** 2) for z in z_arr])
        Rs = 0.0
        for i, weight in enumerate(weight_arr):
            res_in_square_bracket = z_arr[i].real - \
                                    sum([self.RC_para_list[k][0] / (1 + (self.w_arr[i] * self.tao_arr[k])**2) for k in range(self.M)])
            Rs += weight * res_in_square_bracket
        Rs /= weight_arr[:-1].sum()
        return Rs

    def update_para(self, R_arr):
        """
        R_list / R_arr:
            [Rs, R0, R1, ..., R_M-1]
            优化算法迭代产生新的阻抗值，替换原来的R
            同时更新对应的电容C
        :return:
        """
        if self.OA_obj_fun_mode == 'imag':
            C_list = [tao / R for tao, R in zip(self.tao_arr, R_arr)]
            self.RC_para_list = [[R, C] for R, C in zip(R_arr, C_list)]
        elif (self.OA_obj_fun_mode == 'real') or (self.OA_obj_fun_mode == 'both'):
            self.Rs = R_arr[0]
            C_list = [tao / R for tao, R in zip(self.tao_arr, R_arr[1:])]
            self.RC_para_list = [[R, C] for R, C in zip(R_arr[1:], C_list)]

    def update_u(self):
        """
        refer paper0-eq21
        :return:
        """
        positive_R_list = []
        negtive_R_list = []
        for RC_list in self.RC_para_list:
            R = RC_list[0]
            if R >= 0:
                positive_R_list.append(R)
            elif R < 0:
                negtive_R_list.append(R)
        self.u = 1 - abs(sum(negtive_R_list)) / sum(positive_R_list)

    def cal_Zimag_residual(self):
        pass

    def lin_KK(self, OA=Levenberg_Marquart_0, OA_obj_fun_mode='both', OA_obj_fun_weighting_type='modulus',
               save_iter=False, u_optimum=0.85, manual_M=None):
        self.OA_obj_fun_mode = OA_obj_fun_mode
        self.OA_obj_fun_weighting_type = OA_obj_fun_weighting_type

        if manual_M is not None:
            self.M = manual_M
            
        self.calc_timeConstant()
        self.init_para()
        self.update_u()

        # init Levenberg_Marquardt
        # OA: Optimization Algorithm
        oa = OA(impSpe=self.impSpe,
                obj_fun=vogit_obj_fun_0,
                # obj_fun=cal_ChiSquare_pointWise_0,
                obj_fun_mode=OA_obj_fun_mode,
                obj_fun_weighting_type=OA_obj_fun_weighting_type,
                iter_max=100)
        
        while (self.u >= u_optimum) and (self.M <= self.M_max):
            if OA_obj_fun_mode == 'imag':
                oa.get_initial_para_arr(para_arr=np.array([RC[0] for RC in self.RC_para_list]))
            elif (OA_obj_fun_mode == 'real') or (OA_obj_fun_mode == 'both'):
                oa.get_initial_para_arr(para_arr=np.array([self.Rs] + [RC[0] for RC in self.RC_para_list]))

            """
            oa.iterate 传入z_arr, w_arr, tao_arr的目的：
                z_arr是观测数据，和vogit的拟合数据对比来计算残差
                w_arr, tao_arr用来确定Vogit模型的
                在L-M中，R每变动一次，C要由 C = tao / R 计算更新
                只有RC中的R是待求的未知数
            """
            oa.iterate(timeConstant_arr=self.tao_arr)
            R_arr = oa.para_arr # N * 1

            # update RC
            self.update_para(R_arr)

            # update u
            self.update_u()
            
            if manual_M is not None:
                chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                self.chiSquare_list = [chiSquare]
                self.chiSquare_real_list = [chiSquare_real]
                self.chiSquare_imag_list = [chiSquare_imag]
                self.real_residual_list = [real_residual_list]
                self.imag_residual_list = [imag_residual_list]
                break
                
            # The value of c (u_max) is a design parameter,
            # however from the author’s experience c = 0.85 has proven to be an excellent choice.
            if (self.u >= u_optimum) and (self.M <= self.M_max): # underfitting
                # 打印输出、保存迭代的中间结果
                print('M=', self.M, 'u=', self.u)
                # print('M=', self.M, 'u=', self.u, 'Rs=', self.Rs, '(RC)s=', self.RC_para_list)
                if save_iter == True:
                    if self.M == 1:
                        self.M_list = [1]
                        self.u_list = [copy.deepcopy(self.u)]
                        self.Rs_list = [copy.deepcopy(self.Rs)]
                        self.RC_para_pack_list = [copy.deepcopy(self.RC_para_list)]

                        chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                        self.chiSquare_list = [chiSquare]
                        self.chiSquare_real_list = [chiSquare_real]
                        self.chiSquare_imag_list = [chiSquare_imag]
                        self.real_residual_list = [real_residual_list]
                        self.imag_residual_list = [imag_residual_list]

                    elif self.M > 1:
                        self.M_list.append(copy.deepcopy(self.M))
                        self.u_list.append(copy.deepcopy(self))
                        self.Rs_list.append(copy.deepcopy(self.Rs))
                        self.RC_para_pack_list.append(copy.deepcopy(self.RC_para_list))

                        chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list = self.cal_various_criteria()
                        self.chiSquare_list.append(chiSquare)
                        self.chiSquare_real_list.append(chiSquare_real)
                        self.chiSquare_imag_list.append(chiSquare_imag)
                        self.real_residual_list.append(real_residual_list)
                        self.imag_residual_list.append(imag_residual_list)
                        print('M=', self.M, 'u=', self.u, chiSquare)

                self.M += 1
                self.init_para()
            else:
                print('M=', self.M, 'u=', self.u)
                break

    def simulate_Z(self):
        """
        使用拟合的各种参数：Rs + M * RC
        :return:
        """
        self.z_sim_arr = np.empty(shape=(self.M, self.impSpe.z_arr.shape[0]), dtype=complex)
        for i in range(self.M):
            R, C = self.RC_para_list[i]
            tmp_z_sim_list = [aRCb(w, R, C) for w in self.w_arr]
            self.z_sim_arr[i, :] = np.array(tmp_z_sim_list)
        self.z_sim_arr = self.z_sim_arr.sum(axis=0)
        self.z_sim_arr += self.Rs

    def cal_various_criteria(self):
        """
        calculate
            weight = 1 / (z.real ** 2 + z.imag ** 2)
            X^2, defined in paper0 - Eq 15
                在这里没有办法计算ZSimpWin中的X^2，因为 过程ECM未知 == 代求参数的数量未知 --》 系统的自由度无法确定
                这里的X^2计算如下：
                    N = data points
                    X^2 = (1/N) * ∑{ weight * [(Z(w)i.real - Zi.real) ** 2 + (Z(w)i.imag - Zi.imag) **2] }
            X^2_imag, defined in paper0 - Eq 20
            X^2_real, 模仿 X^2_imag 的计算
            🔺Real, defined in paper0 - Eq 15
            🔺Imag, defined in paper0 - Eq 16
        :return:
        """
        chiSquare = 0.0
        chiSquare_real = 0.0
        chiSquare_imag = 0.0
        imag_residual_list = []
        real_residual_list = []

        self.simulate_Z()
        z_arr = self.impSpe.z_arr

        modulus_weight_list = [1 / (z.real ** 2 + z.imag ** 2) for z in z_arr]

        for weight, z_sim, z in zip(modulus_weight_list, self.z_sim_arr, z_arr):
            real_residual_list.append(math.sqrt(weight) * (z.real - z_sim.real))
            imag_residual_list.append(math.sqrt(weight) * (z.imag - z_sim.imag))

            chiSquare_real += (1 / z_arr.shape[0]) * weight * ((z_sim.real - z.real)**2)
            chiSquare_imag += (1 / z_arr.shape[0]) * weight * ((z_sim.imag - z.imag)**2)

            chiSquare += chiSquare_imag + chiSquare_real
        return chiSquare, chiSquare_real, chiSquare_imag, real_residual_list, imag_residual_list

    def save2pkl(self, fp, fn):
        pickle_file(obj=self, fn=fn, fp=fp)

# impS = IS_0()
# dpfc_src\datasets\goa_datasets\simulated\ecm_001\2020_07_04_sim_ecm_001_pickle.file
# impS.read_from_simPickle(fp='../datasets/goa_datasets/simulated/ecm_001/',
#                          fn='2020_07_04_sim_ecm_001_pickle.file')
# vogit = Vogit(impSpe=impS)

# OA_obj_fun_mode = 'both'
# print(OA_obj_fun_mode)
# vogit.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode)
# print('M=', vogit.M, 'u=',vogit.u, 'Rs=',vogit.Rs,'(RC)s=',vogit.RC_para_list)
# python vogit_0.py