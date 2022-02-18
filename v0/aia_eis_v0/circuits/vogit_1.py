import sys
sys.path.append('../')
import numpy as np
import math
import copy
import os

from circuits.elements import ele_C, ele_L
from IS.IS import IS_0
from IS.IS_criteria import cal_ChiSquare_0
from utils.file_utils.pickle_utils import pickle_file
from utils.visualize_utils.IS_plots.ny import nyquist_multiPlots_1, nyquist_plot_1

def RC(para_arr, w_arr):
    R, tao = para_arr[0], para_arr[1]
    z = R / (1+1j*w_arr*tao)
    return z

class Vogit_3:
    """
    Refer
        papers:
            paper1: A Linear Kronig-Kramers Transform Test for Immittance Data Validation
            paper0: A Method for Improving the Robustness of linear Kramers-Kronig Validity Tests
    Note:
        Vogit 最基本的电路为
            Rs-M*(RC)-[Cs]-Ls
            Ls: inductive effects are considered byadding an additional inductivity [1]
            Cs:
                option to add a serial capacitance that helps validate data with no low-frequency intercept
                due to their capacitive nature an additional capacityis added to the ECM.

            1- 只考虑 complex / imag / real -fit中的complex-fit
            2- 三种加权方式只考虑 modulus
            3- add Capacity / Inductance 中 只考虑 add Capacity
    Version:
        v3:
            更新2：取消手动设置M的选择，合理设置M的上限，达到上限在停止
            更新1：仿照《Impedance.py》构造Ax=Y，直接求解
                class vogit的前两个版本在 \dpfc_src\circuits\vogit_0.py 中，都不好使
        v2: 之前的Vogit中没有加入电感L，在这一版本中加上
    """

    def __init__(self, impSpe, fit_type='complex', u_optimum=0.85, add_C=False, M_max=None):
        """
        因为Vogit是一个measurement model，所以使用vogit之前一定会传进来一个IS
        :param
            impSpe: IS cls
            fit_type: str
                'real',
                'imag',
                'complex',
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
        self.z_arr = self.impSpe.z_arr

        self.fit_type = fit_type
        self.u_optimum = u_optimum
        self.add_C = add_C

        self.M = 1

        if (M_max is not None) and (type(M_max) == int):
            self.M_max = M_max
        else:
            self.get_Mmax()

    def get_Mmax(self):
        """
        M_max 设置条件
            condition 1- Paper1: As a rule of thumb we can conclude that, for the single fit and transformation, the v range should be
                equal to the inverse w range with a distribution of 6 or 7 Tcs per decade. 在这里再稍微取的更大一些 8 * decades
            condition 2- 在Vogit 单独使用 实部/虚部拟合时，由于系数矩阵A (row col) 要求 rol=tested points > col=number of parameters
        """
        # condition 1
        M1 = int(math.log10(self.w_arr.max() / self.w_arr.min())) * 7

        # condition 2
        num_points = self.w_arr.size
        if self.add_C:
            M2 = num_points - 3 - 1
        else:
            M2 = num_points - 2 - 1
        self.M_max = min(M1, M2)

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

    def update_u(self):
        """
        refer paper0-eq21
        :return:
        """
        if self.fit_type == 'complex':
            self.M_R_arr = self.para_arr[1:-2]

        positive_R_list = []
        negtive_R_list = []
        for R in self.M_R_arr:
            if R >= 0:
                positive_R_list.append(R)
            elif R < 0:
                negtive_R_list.append(R)
        self.u = 1 - abs(sum(negtive_R_list)) / sum(positive_R_list)

    def lin_KK(self):
        self.u = 1
        self.calc_timeConstant()

        while (self.u > self.u_optimum) and (self.M <= self.M_max):
            self.M += 1
            self.calc_timeConstant()
            self.fit_kk()
            # print('M = ', self.M, 'U = ', self.u)
            self.update_u()
        # print('M = ', self.M, 'U = ', self.u)

    def fit_kk(self):
        """
        Are/im
            N row
            M+2 or M+3(with capacity) col
            Are
                col 0: Rs(w0) / |Z(w0)|, Rs(w1) / |Z(w1)|, Rs(w2) / |Z(w2)|, ..., Rs(w_N-1) / |Z(w_N-1)|
                col 1: Z_RCk_0(w0)_re = Rk_0 / {[1+(w0*tao0)**2]*|Z(w0)|},
                        Z_RCk_0(w1)_re = Rk_0 / {[1+(w1*tao0)**2]*|Z(w1)|}
                        Z_RCk_0(w2)_re = Rk_0 / {[1+(w2*tao0)**2]*|Z(w2)|},
                        ...,
                        Z_RCk_0(w_N-1)_re = Rk_0 / {[1+(w_N-1*tao_0)**2]*|Z(w_N-1)|}
                ...
                col k(M): Z_RCk_k(w0)_re = Rk_k / {[1+(w0*taok)**2]*|Z(w0)|},
                        Z_RCk_k(w1)_re = Rk_k / {[1+(w1*taok)**2]*|Z(w1)|}
                        Z_RCk_k(w2)_re = Rk_k / {[1+(w2*taok)**2]*|Z(w2)|},
                        ...,
                        Z_RCk_k(w_N-1)_re = Rk_k / {[1+(w_N-1*tao_k)**2]*|Z(w_N-1)|}
                col -2(C): 如果加capacity，它对阻抗实部的贡献为0
                    0, 0, 0, ..., 0
                col -1(L): L对阻抗实部的贡献为0
                    0, 0, 0, ..., 0
            Aim
                col 0: Rs(wi)_im = 0, 0,0,0,...,0,0
                col 1: Z_RCk_0(w0)_im = (-1 * w0 * Rk_0 * tao0) / {[1+(w0*tao0)**2]*|Z(w0)|},
                        Z_RCk_0(w1)_im = (-1 * w1 * Rk_0 * tao0) / {[1+(w1*tao0)**2]*|Z(w1)|},
                        Z_RCk_0(w2)_im = (-1 * w2 * Rk_0 * tao0) / {[1+(w2*tao0)**2]*|Z(w2)|},
                        ...,
                        Z_RCk_0(w_N-1)_im = (-1 * w_N-1 * Rk_0 * tao0) / {[1+(w_N-1*tao0)**2]*|Z(w0_N-1)|},
                ...
                col k(M):
                col -2(C):
                col -1(L):
        :return:
        """
        Are = np.zeros(shape=(self.w_arr.size, self.M + 2))
        Aim = np.zeros(shape=(self.w_arr.size, self.M + 2))

        if self.add_C:
            Are = np.zeros(shape=(self.w_arr.size, self.M + 3))
            Aim = np.zeros(shape=(self.w_arr.size, self.M + 3))

        # Rs col
        Are[:,0] = 1 / np.abs(self.z_arr)
        # Aim[:,0] = np.zeros(shape=(self.w_arr.size)) 本来就是0

        # RC_1~M col
        for i in range(self.M):
            Are[:, i+1] = RC(para_arr=np.array([1, self.tao_arr[i]]), w_arr=self.w_arr).real / np.abs(self.z_arr)
            Aim[:, i+1] = RC(para_arr=np.array([1, self.tao_arr[i]]), w_arr=self.w_arr).imag / np.abs(self.z_arr)

        if self.add_C:
            # Are[:, -2] = np.zeros(shape=(self.w_arr.size)) 本来就是0
            Aim[:, -2] = -1 / (self.w_arr * np.abs(self.z_arr))

        Aim[:, -1] = self.w_arr / np.abs(self.z_arr)

        if self.fit_type == 'real':
            self.para_arr = np.linalg.pinv(Are).dot(self.z_arr.real / np.abs(self.z_arr))

            XLim = np.zeros(shape=(self.w_arr.size, 2))
            
            # 根据paper0-Lin-KK-Eq10 再构造一组方程 求C和L, X= 1/C
            # data for L-col
            # Aim[:, -1] = self.w_arr / np.abs(self.z_arr)
            XLim[:, -1] = self.w_arr / np.abs(self.z_arr)

            # data for C-col
            if self.add_C:
                XLim[:, -2] = -1 / self.w_arr / np.abs(self.z_arr)
                # Aim[:, -2] = -1 / self.w_arr / np.abs(self.z_arr)
                """
                self.para_arr[-2] = 一个很小的正数 如1e-18 的原因：
                    在fit_type == 'real'时， self.para_arr = np.linalg.pinv(Are).dot(self.z_arr.real / np.abs(self.z_arr))
                    得到的 para_arr【-2：】 = 【X，L】 == 【0， 0】，由于下方代码马上需要计算 拟合参数所得的阻抗，计算Cs的阻抗时，
                    Cs=1/X，因X=0，Cs-》Inf，所有要给X一个必要的、很小的正数，来防止计算上溢
                """
                # self.para_arr[-2] = 1e-20

            # self.simulate_Z()
            # tmp_para_arr = np.linalg.pinv(Aim).dot((self.z_arr.imag - self.z_sim_arr.imag) / np.abs(self.z_arr))
            
            z_vogit_arr = self.simulate_vogit()
            XL = np.linalg.pinv(Aim).dot((self.z_arr.imag - z_vogit_arr.imag) / np.abs(self.z_arr))
            
            # self.para_arr[-1] = tmp_para_arr[-1]
            self.para_arr[-1] = XL[-1]
            if self.add_C:
                # self.para_arr[-2] = tmp_para_arr[-2]
                self.para_arr[-2] = XL[-2]

        elif self.fit_type == 'imag':
            self.para_arr = np.linalg.pinv(Aim).dot(self.z_arr.imag / np.abs(self.z_arr))
            """
            根据 paper1-lin-KK-Eq7 计算 Rs
                Eq7中方括号里的叠加 == Vogit中M个RC的阻抗对于实部的贡献
            """
            self.simulate_Z()
            weight_arr = 1 / (np.abs(self.z_arr) ** 2)
            
            # paper1-Eq 7
            # ValueError: setting an array element with a sequence.
            Rs = np.sum(weight_arr * (self.z_arr.real - self.z_sim_arr.real)) / np.sum(weight_arr)
            self.para_arr[0] = Rs

        elif self.fit_type == 'complex':
            A_inv = np.linalg.inv(Are.T.dot(Are) + Aim.T.dot(Aim))
            Y = Are.T.dot(self.z_arr.real / np.abs(self.z_arr)) + Aim.T.dot(self.z_arr.imag / np.abs(self.z_arr))
            self.para_arr = A_inv.dot(Y)

    def simulate_vogit(self):
        """
        这里的Vogit是纯的 Rs + M * RC
        :return:
        """
        self.Rs = self.para_arr[0]
        self.M_R_arr = self.para_arr[1: self.M+1]
        z_vogit_arr = np.empty(shape=(self.M, self.w_arr.size), dtype=complex)
        
        # Z of M RC
        for i, R in enumerate(self.M_R_arr):
            z_RC_arr = RC(para_arr=np.array([R, self.tao_arr[i]]), w_arr=self.w_arr)
            z_vogit_arr[i, :] = z_RC_arr
        z_vogit_arr = z_vogit_arr.sum(axis=0)
        z_vogit_arr += self.Rs
        return z_vogit_arr
        
    def simulate_Z(self):
        self.Rs = self.para_arr[0]
        self.Ls = self.para_arr[-1]

        if self.add_C:
            self.M_R_arr = self.para_arr[1: -2]
            # X = 1/C
            self.Cs = 1 / self.para_arr[-2]
            # print('Cs:', self.Cs)
            self.z_sim_arr = np.empty(shape=(self.M + 2, self.w_arr.size), dtype=complex)
        else:
            self.M_R_arr = self.para_arr[1: -1]
            self.z_sim_arr = np.empty(shape=(self.M + 1, self.w_arr.size), dtype=complex)

        # ---------- 依次按照 M个RC的阻抗 -》 【C的阻抗】 -》 L的阻抗 -》 Rs的阻抗 拼接--------------
        # Z of M RC
        for i, R in enumerate(self.M_R_arr):
            z_RC_arr = RC(para_arr=np.array([R, self.tao_arr[i]]), w_arr=self.w_arr)
            self.z_sim_arr[i, :] = z_RC_arr

        if self.add_C:
            # Z of Cs
            self.z_sim_arr[self.M, :] = np.array([ele_C(w, C=self.Cs) for w in self.w_arr])
            # Z of Ls
            self.z_sim_arr[self.M+1, :] = np.array([ele_L(w, L=self.Ls) for w in self.w_arr])
        else:
            # Z of Ls
            self.z_sim_arr[self.M, :] = np.array([ele_L(w, L=self.Ls) for w in self.w_arr])

        self.z_sim_arr = self.z_sim_arr.sum(axis=0)

        # Z of Rs
        self.z_sim_arr += self.Rs
        # ---------- 依次按照 M个RC的阻抗 -》 C的阻抗 -》 L的阻抗 -》 Rs的阻抗 拼接--------------

    def cal_residual(self):
        """
        按照paper0-Eq 15 and Eq 16
        residual_arr = Z_arr - Z_sim_arr
        :return:
        """
        self.simulate_Z()
        z_abs_arr = np.abs(self.z_arr)
        self.residual_arr = (self.z_arr - self.z_sim_arr) / z_abs_arr

    def residual_statistic(self, type):
        """
        我定义衡量残差的几种定量标准；
            1 残差的绝对值
                实部残差的绝对值
                虚部残差的绝对值
            2 残差的 平方
                实部残差的 平方
                虚部残差的 平方
        :param
            type: str
                'abs'
                'square'
        """
        self.cal_residual()
        if type == 'abs':
            residual_real_abs_arr = np.abs(self.residual_arr.real)
            residual_imag_abd_arr = np.abs(self.residual_arr.imag)
            return residual_real_abs_arr, residual_imag_abd_arr
        elif type == 'square':
            residual_real_square_arr = self.residual_arr.real ** 2
            residual_imag_square_arr = self.residual_arr.imag ** 2
            return residual_real_square_arr, residual_imag_square_arr

    def cal_chiSquare(self, weight_type='modulus'):
        """
        这里不能按照ZSimpWin的方式计算，因ZSimpWin的方式计算 涉及到 ECM中参数的数量，删除点前后的ECM可能不一样，没法计算
        故只能按照 chiSquare = weight * [▲Re**2 + ▲Im**2]
        :return:
        """
        self.simulate_Z()
        if weight_type == 'modulus':
            self.chi_square = cal_ChiSquare_0(z_arr=self.z_arr, z_sim_arr=self.z_sim_arr, weight_type=weight_type)
        return self.chi_square

    def save2pkl(self, fp, fn):
        pickle_file(obj=self, fn=fn, fp=fp)

# ---------------------------------- Test Vogit_3 on Lin-KK-Ex1_LIB_time_invariant ----------------------------------
# 1- load data
# fit_type = 'real'
# fit_type = 'imag'
# fit_type = 'complex'

# lib_res_fp = '../plugins_test/jupyter_code/rbp_files/2/example_data_sets/LIB_res'
# if fit_type == 'complex':
#     ex1_data_dict = np.load(os.path.join(lib_res_fp, 'Ex1_LIB_time_invariant_res.npz'))
# elif fit_type == 'real':
#     ex1_data_dict = np.load(os.path.join(lib_res_fp, 'Ex1_LIB_time_invariant_real_addC_res.npz'))
# elif fit_type == 'imag':
#     ex1_data_dict = np.load(os.path.join(lib_res_fp, 'Ex1_LIB_time_invariant_imag_addC_res.npz'))

# ex1_z_arr = ex1_data_dict['z_arr']
# ex1_f_arr = ex1_data_dict['fre']
# ex1_z_MS_sim_arr = ex1_data_dict['z_sim']
# ex1_real_residual_arr = ex1_data_dict['real_residual']
# ex1_imag_residual_arr = ex1_data_dict['imag_residual']

# ex1_IS = IS_0()
# ex1_IS.raw_z_arr = ex1_z_arr
# ex1_IS.exp_area = 1.0
# ex1_IS.z_arr = ex1_z_arr
# ex1_IS.fre_arr = ex1_f_arr
# ex1_IS.w_arr = ex1_IS.fre_arr * 2 * math.pi

# --------------- real Fit ---------------
# ex1_vogit = Vogit_3(impSpe=ex1_IS, fit_type=fit_type, add_C=True)
# ex1_vogit.lin_KK()
# # compare nyquist plots of MS-Lin-KK and Mine
# ex1_z_MS_sim_list = ex1_z_MS_sim_arr.tolist()
# ex1_vogit.simulate_Z()
# z_pack_list = [ex1_z_arr.tolist(), ex1_z_MS_sim_list, ex1_vogit.z_sim_arr.tolist()]
# nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[0.015, 0.045], y_lim=[0, 0.02], plot_label_list=['Ideal IS', 'MS-real-Fit','Mine-real-Fit'])
# --------------- real Fit ---------------

# --------------- imag Fit ---------------
# ex1_vogit = Vogit_3(impSpe=ex1_IS, fit_type=fit_type, add_C=True)
# ex1_vogit.lin_KK()
# # compare nyquist plots of MS-Lin-KK and Mine
# ex1_z_MS_sim_list = ex1_z_MS_sim_arr.tolist()
# ex1_vogit.simulate_Z()
# z_pack_list = [ex1_z_arr.tolist(), ex1_z_MS_sim_list, ex1_vogit.z_sim_arr.tolist()]
# nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[0.015, 0.045], y_lim=[0, 0.02], plot_label_list=['Ideal IS', 'MS-imag-Fit','Mine-imag-Fit'])
# --------------- imag Fit ---------------

# --------------- Complex Fit ---------------
# ex1_vogit = Vogit_3(impSpe=ex1_IS, add_C=True)
# ex1_vogit.lin_KK()
# # compare nyquist plots of MS-Lin-KK and Mine
# ex1_z_MS_sim_list = ex1_z_MS_sim_arr.tolist()
# ex1_vogit.simulate_Z()
# z_pack_list = [ex1_z_arr.tolist(), ex1_z_MS_sim_list, ex1_vogit.z_sim_arr.tolist()]
# nyquist_multiPlots_1(z_pack_list=z_pack_list, x_lim=[0.015, 0.045], y_lim=[0, 0.02], plot_label_list=['Ideal IS', 'MS-Fit','Mine-Fit'])
# --------------- Complex Fit ---------------
# ---------------------------------- Test Vogit_1 on Lin-KK-Ex1_LIB_time_invariant ----------------------------------