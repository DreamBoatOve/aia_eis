import copy
import numpy as np

from circuits.circuit_pack import aRCb
# from circuits.ecm import ecm_serial_matcher
from circuits.ecm_simulator import ecm_simulator_1
from circuits.elements import ele_C, ele_L

class L_M_0:
    """
    简单实现L-M的核心部分，并能拟合简单的函数
    """
    def __init__(self, input_x_arr, input_y_arr, para_arr, obj_fun, iter_max=100):
        """
        :param
            观测数据
                input_x_arr: ndarray(float)
                input_y_arr: ndarray(float)
            para_arr: ndarray(float)
                待拟合的参数
            obj_fun:
            iter_max:
        """
        self.input_x_arr = input_x_arr
        self.input_y_arr = input_y_arr
        self.para_arr = para_arr
        self.obj_fun = obj_fun
        self.iter_max = iter_max
        
        # defined in paper0 eq 3.15a
        self.e1 = 1e-7

        # defined in paper0 eq 3.15b
        self.e2 = 1e-18
        self.e2 = 1e-15

        # defined in paper0 eq 3.14
        self.tao = 1e-6

        self.iter_count = 0
    
    def cal_residual(self):
        y_arr = self.obj_fun(input_x_arr=self.input_x_arr, para_arr=self.para_arr)
        residual_arr = self.input_y_arr - y_arr
        return residual_arr
    
    def cal_derivative(self, para_index):
        step = 1e-10
        # step = 1e-3
        big_para_arr = copy.deepcopy(self.para_arr)
        big_para_arr[para_index] += step
        small_para_arr = copy.deepcopy(self.para_arr)
        small_para_arr[para_index] -= step
        
        big_output_arr = self.obj_fun(input_x_arr = self.input_x_arr, para_arr = big_para_arr)
        small_output_arr = self.obj_fun(input_x_arr = self.input_x_arr, para_arr = small_para_arr)

        derivative_arr = (big_output_arr - small_output_arr) / (2 * step)
        return derivative_arr
    
    def cal_Jacobian(self):
        M = self.input_x_arr.shape[0]
        N = self.para_arr.shape[0]
        J_arr = np.zeros(shape=(M,N))
        for i in range(N):
            J_arr[:, i] = self.cal_derivative(para_index=i)
        return J_arr

    def iterate(self):
        v = 2
        jacob_arr = self.cal_Jacobian()
        A = jacob_arr.T.dot(jacob_arr)

        # defined in paper0 eq 3.14
        mu = self.tao * max([A[i, i] for i in range(A.shape[0])])
        
        # g = jacob_arr.T.dot(self.obj_fun(input_x_arr = self.input_x_arr, para_arr = self.para_arr).reshape(jacob_arr.shape[0], 1))
        g = jacob_arr.T.dot(self.cal_residual())

        found = np.linalg.norm(g, ord=np.inf) <= self.e1
        iter_count = 0
        while (not found) and (iter_count < self.iter_max):
            hessian_LM_arr = A + mu * np.eye(A.shape[0])
            
            # h同时包含参数更新的大小和方向
            # h = np.linalg.inv(hessian_LM_arr).dot(-1 * g)
            h = np.linalg.inv(hessian_LM_arr).dot(g)

            if (np.linalg.norm(self.cal_residual(), ord=2) <= self.e2 * np.linalg.norm(self.para_arr, ord=2)):
                found = True
                break
            else:
                # cal gain ratio, defined in paper0 eq 2.18
                # F(x) = 0.5 * || f(x) ||
                F_0 = 0.5 * (np.linalg.norm(self.cal_residual(), ord=2) ** 2)
                new_para_arr = self.para_arr + h.ravel()
                F_h = 0.5 * (np.linalg.norm(self.input_y_arr - self.obj_fun(input_x_arr=self.input_x_arr, para_arr=new_para_arr),
                                           ord=2) ** 2)
                # cal L(0) - L(h), defined in paper0 eq 3.14的下方
                L0_minus_Lh = 0.5 * h.T.dot(mu * h + g)

                rou = (F_0 - F_h) / L0_minus_Lh

                if rou > 0:  # accept h (step)
                    self.para_arr = new_para_arr

                    # update Jacobian, A, g
                    jacob_arr = self.cal_Jacobian()
                    A = jacob_arr.T.dot(jacob_arr)
                    g = jacob_arr.T.dot(self.cal_residual())

                    found1 = np.linalg.norm(g, ord=np.inf) <= self.e1
                    found2 = np.linalg.norm(self.cal_residual(), ord=2) <= 1e-6
                    if found1 or found2:
                        break

                    # update mu, v
                    mu = mu * max(1 / 3, 1 - (2 * mu - 1) ** 3)
                    v = 2
                else:
                    mu = mu * v
                    v = 2 * v

            iter_count += 1
            print('LM-iter:',iter_count, self.para_arr)

# def f1(input_x_arr, para_arr):
#     a, b = para_arr
#     y_arr = a * np.exp(b * input_x_arr)
#     return y_arr
# add noise
# def add_noise(y_arr):
#     mu, sigma = 0, 5
#     y_distubed_arr = y_arr + np.random.normal(mu, sigma, y_arr.shape[0])
#     return y_distubed_arr
# input_x_arr = np.linspace(0, 10, 100)
# y_arr = f1(input_x_arr=input_x_arr, para_arr=np.array([10.0, 0.8]))
# y_distubed_arr = add_noise(y_arr)

# lm = L_M_0(input_x_arr=input_x_arr, input_y_arr=y_distubed_arr, para_arr=np.array([1.0, 1.0]), obj_fun=f1)
# lm.iterate()

# import matplotlib.pyplot as plt
# plt.scatter(input_x_arr, y_distubed_arr)
# y_fit_arr = f1(input_x_arr, para_arr=lm.para_arr)
# plt.plot(input_x_arr, y_fit_arr)
# plt.show()

"""
Require:
    1- 可以用于KKT进行IS的数据有效性校验
        object function = weight * (Zimag_residual ** 2)
    2- 可用于IS拟合ECM参数
        object function = weight * (Zreal_residual ** 2 + Zimag_residual ** 2)
refer
    papers:
        **paper-0: LOA: <METHODS FOR NON-LINEAR LEAST SQUARES PROBLEMS>
            3.2. The Levenberg–Marquardt Method
    blogs:
        blog1: **[优化]Levenberg-Marquardt 最小二乘优化
            https://zhuanlan.zhihu.com/p/42415718
        blog0: LM(Levenberg–Marquardt)算法原理及其python自定义实现
            https://blog.csdn.net/wolfcsharp/article/details/89674973
"""
def ecm_obj_fun_1(w_arr, para_arr, ecm_serial, z_arr):
    z_sim_list = ecm_simulator_1(ecm_serial, para_list=para_arr.tolist(), fre=None , w=w_arr.tolist())
    z_sim_arr = np.array(z_sim_list)
    return z_sim_arr / np.abs(z_arr)

def ecm_obj_fun(w_arr, para_arr, ecm_serial):
    z_sim_list = ecm_simulator_1(ecm_serial, para_list=para_arr.tolist(), fre=None , w=w_arr.tolist())
    z_sim_arr = np.array(z_sim_list)

    # residual_arr = z_arr - z_sim_arr
    # ZSimpWin_ChiSquare = (1 / z_arr.shape[0]) * (residual_arr.real ** 2 + residual_arr.imag ** 2) / (np.abs(z_arr) ** 2)

    # modulus_weight_arr = 1 / (np.abs(z_arr) ** 2)
    # (residual_arr.real ** 2) / modulus_weight_arr
    # pass
    return z_sim_arr

def vogit_obj_fun_1(w_arr, para_arr, tao_arr, obj_fun_mode='both', add_C=False):
# def vogit_obj_fun_1(w_arr, para_arr, tao_arr, obj_fun_mode='both', add_C=False):
    """
    Function

    :param
        w_arr:
        para_arr:
        tao_arr:
        obj_fun_mode:
        add_C:
    :return:
    Version:
        1: 将电感的添加设置为默认， add-capacity为可选
    """
    if obj_fun_mode == 'imag':
        # para_arr = [R0, R1, ..., R_M-1]
        # RC_para_list = [[R, tao / R] for R, tao in zip(para_arr, tao_arr)]
        #
        # z_sim_arr = np.empty(shape=(len(RC_para_list), w_arr.shape[0]), dtype=complex)
        # for i, RC_list in enumerate(RC_para_list):
        #     R, C = RC_list
        #     tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
        #     # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
        #     z_sim_arr[i, :] = tmp_z_sim_list
        # z_sim_arr = z_sim_arr.sum(axis=0)
        # return z_sim_arr.imag
        pass

    elif (obj_fun_mode == 'real') or (obj_fun_mode == 'both'):
        if add_C:
            # para_arr = [*Rs*, *Ls*, *Cs*, R0, R1, ..., R_M-1]
            Rs = para_arr[0]
            Ls = para_arr[1]
            Cs = para_arr[2]
            M_R_arr = para_arr[3:]

            # -------------- 计算M个RC各自产生的阻抗 --------------
            z_sim_arr = np.empty(shape=(M_R_arr.size, w_arr.shape[0]), dtype=complex)
            for i, R in enumerate(M_R_arr):
                tao = tao_arr[i]
                tmp_z_sim_list = [aRCb(w, R0=R, C0=tao/R) for w in w_arr]
                # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
                z_sim_arr[i, :] = tmp_z_sim_list
            # -------------- 计算M个RC各自产生的阻抗 --------------

            # 加上Ls产生的阻抗
            L_z_sim_arr = np.array([ele_L(w, Ls) for w in w_arr]).reshape((1, w_arr.size))
            # 加上Cs产生的阻抗
            C_z_sim_arr = np.array([ele_C(w, Cs) for w in w_arr]).reshape((1, w_arr.size))

            # 合并M个RC + Ls + Cs 各自产生的阻抗
            z_sim_arr = np.concatenate((z_sim_arr, L_z_sim_arr, C_z_sim_arr), axis=0)
            z_sim_arr = z_sim_arr.sum(axis=0)

            # 合并M个RC + Ls + Cs + Rs 各自产生的阻抗
            z_sim_arr += Rs

            if obj_fun_mode == 'real':
                return z_sim_arr.real
            elif obj_fun_mode == 'both':
                return z_sim_arr
        else:
            # para_arr = [*Rs*, *Ls*, R0, R1, ..., R_M-1]
            Rs = para_arr[0]
            Ls = para_arr[1]
            M_R_arr = para_arr[2:]

            # -------------- 计算M个RC各自产生的阻抗 --------------
            z_sim_arr = np.empty(shape=(M_R_arr.size, w_arr.shape[0]), dtype=complex)
            for i, R in enumerate(M_R_arr):
                tao = tao_arr[i]
                tmp_z_sim_list = [aRCb(w, R0=R, C0=tao / R) for w in w_arr]
                # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
                z_sim_arr[i, :] = tmp_z_sim_list
            # -------------- 计算M个RC各自产生的阻抗 --------------

            # 加上Ls产生的阻抗
            L_z_sim_arr = np.array([ele_L(w, Ls) for w in w_arr]).reshape((1, w_arr.size))

            # 合并M个RC + Ls 各自产生的阻抗
            z_sim_arr = np.concatenate((z_sim_arr, L_z_sim_arr), axis=0)
            z_sim_arr = z_sim_arr.sum(axis=0)

            # 合并Rs + M个RC各自产生的阻抗
            z_sim_arr += Rs

            if obj_fun_mode == 'real':
                return z_sim_arr.real
            elif obj_fun_mode == 'both':
                return z_sim_arr

def vogit_obj_fun_0(w_arr, para_arr, tao_arr, obj_fun_mode='both', add_C=False, add_L=False):
    if obj_fun_mode == 'imag':
        # para_arr = [R0, R1, ..., R_M-1]
        RC_para_list = [[R, tao / R] for R, tao in zip(para_arr, tao_arr)]

        z_sim_arr = np.empty(shape=(len(RC_para_list), w_arr.shape[0]), dtype=complex)
        for i, RC_list in enumerate(RC_para_list):
            R, C = RC_list
            tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
            # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
            z_sim_arr[i, :] = tmp_z_sim_list
        z_sim_arr = z_sim_arr.sum(axis=0)
        return z_sim_arr.imag

    elif (obj_fun_mode == 'real') or (obj_fun_mode == 'both'):
        if add_C:
            # para_arr = [*Rs*, *C*, R0, R1, ..., R_M-1]
            Rs = para_arr[0]

            # Wrong,在下面的for循环中，C的值被改动了
            # C = para_arr[1]
            # Right
            Cs = para_arr[1]
            RC_para_list = [[R, tao / R] for R, tao in zip(para_arr[2:], tao_arr)]
            
            # -------------- 计算M个RC各自产生的阻抗 --------------
            z_sim_arr = np.empty(shape=(len(RC_para_list)+1, w_arr.shape[0]), dtype=complex)
            for i, RC_list in enumerate(RC_para_list):
                R, C = RC_list
                tmp_z_sim_list = [aRCb(w, R0=R, C0=C) for w in w_arr]
                # IndexError: too many indices for array: array is 1-dimensional, but 2 were indexed
                z_sim_arr[i, :] = tmp_z_sim_list

            # 加上Cs产生的阻抗
            z_sim_arr[-1, :] = [ele_C(w, Cs) for w in w_arr]
            # -------------- 计算M个RC各自产生的阻抗 --------------
            # 合并M个RC各自产生的阻抗
            z_sim_arr = z_sim_arr.sum(axis=0)

            # 合并Rs + M个RC各自产生的阻抗
            z_sim_arr += Rs
            if obj_fun_mode == 'real':
                return z_sim_arr.real
            elif obj_fun_mode == 'both':
                return z_sim_arr
        else:
            # para_arr = [*Rs*, R0, R1, ..., R_M-1]
            Rs = para_arr[0]
            RC_para_list = [[R, tao / R] for R, tao in zip(para_arr[1:], tao_arr)]
    
            # -------------- 计算M个RC各自产生的阻抗 --------------
            z_sim_arr = np.empty(shape=(len(RC_para_list), w_arr.shape[0]), dtype=complex)
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
            if obj_fun_mode == 'real':
                return z_sim_arr.real
            elif obj_fun_mode == 'both':
                return z_sim_arr

class Levenberg_Marquart_0:
    def __init__(self, impSpe, obj_fun, obj_fun_mode='real', obj_fun_weighting_type='modulus', iter_max=100, **kwargs):
        """
        :param
            impSpe: cls
            obj_fun:
                ECM-vogit
                -------------------
                wrong
                    default from IS.IS_criteria import cal_ChiSquare_pointWise_0
                    f(x)就是度量残差的函数
                    predicted_data = obj_fun(para_arr, x_arr or input_arr)
                    residual = Observed_data - predicted_data
            iter_max:
            obj_fun_mode: str
                'real' obj_fun == loss_fun == measure the error between the fitted and experimental Z_real data
                'imag' obj_fun == loss_fun == measure the error between the fitted and experimental Z_imag data
                'both' obj_fun == loss_fun == measure the error between the fitted and experimental Z_real_and_imag data
            obj_fun_weighting_type: str
                'unity':
                    Wreal = Wimag = 1
                'modulus':
                    Wreal = Wimag = 1 / (z.re **2 + z.im ** 2)
                'proportional':
                    Wreal = Wimag = 不会写
        """
        # defined in paper0 eq 3.15a
        self.e1 = 1e-7

        # defined in paper0 eq 3.15b
        self.e2 = 1e-8
        
        self.e3 = 1e-15

        # defined in paper0 eq 3.14
        self.tao = 1e-6

        self.iter_count = 0
        self.iter_max = iter_max

        """
        obj_fun
            = f(wi, para)
            = Wreal * ((Zi.real - Z(wi, para).real)**2) + Wimag * ((Zi.imag - Z(wi, para))**2)
             Wreal and Wimag are weights and generated from unity / modulus / proportional weighting 
        """
        self.obj_fun = obj_fun
        # self.get_initial_para_arr()

        self.impSpe = impSpe

        self.obj_fun_mode = obj_fun_mode
        self.obj_fun_weighting_type = obj_fun_weighting_type
        
        if 'add_C' in kwargs.keys():
            self.vogit_add_C = kwargs.get('add_C')
        else:
            self.vogit_add_C = False
            
        if 'ecm_serial' in kwargs.keys():
            self.ecm_serial = kwargs.get('ecm_serial')
        else:
            self.ecm_serial = None
        
    def get_initial_para_arr(self, para_arr=None, para_initializer_fun=None, para_initializer_dict=None):
        """
        para:
            para_initializer_fun:
            para_initializer_dict:{
                'limit': para_limit_list[
                    [bottom_boundary, up_boundary],
                    ...
                ]
                'strategy': str
                    'random': randomly select a number in [bottom_boundary, up_boundary]
            }
        :return:
             self.para_arr m*1 array(float)
        """
        if para_arr is not None:
            self.para_arr = para_arr
        if para_initializer_fun is not None:
            self.para_arr = para_initializer_fun()
        elif para_initializer_dict is not None:
            para_limit_list = para_initializer_dict['limit']
            initial_strategy = para_initializer_dict['strategy']

            self.para_arr = np.zeros((len(para_limit_list), 1))
            if initial_strategy == 'random':
                random_arr = np.random.random((len(para_limit_list), 1))
                for i in range(self.para_arr.shape[0]):
                    para_limit = para_limit_list[i]
                    bottom_boundary = para_limit[0]
                    up_boundary = para_limit[1]
                    self.para_arr[i] = bottom_boundary + (up_boundary - bottom_boundary) * random_arr[i]

    def cal_residual(self, para_arr, timeConstant_arr=None):
        # 残差 = 观测值 - 拟合值
        z_arr = self.impSpe.z_arr

        if self.obj_fun == vogit_obj_fun_1:
            z_sim_arr = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=para_arr, tao_arr=timeConstant_arr,
                                     obj_fun_mode=self.obj_fun_mode, add_C=self.vogit_add_C)

        elif self.obj_fun == ecm_obj_fun:
            z_sim_arr = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=para_arr, ecm_serial=self.ecm_serial)
        # elif self.obj_fun == ecm_obj_fun_1:
        #     z_sim_arr = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=para_arr,
        #                              ecm_serial=self.ecm_serial, z_arr=z_arr)

        if self.obj_fun_mode == 'imag':
            # [△Im0, △Im1, △Im2, ..., △Im_N-1]
            imag_residual_arr = z_arr.imag - z_sim_arr.imag
            return imag_residual_arr / np.abs(z_arr)
        elif self.obj_fun_mode == 'real':
            # [△Re0, △Re1, ..., △Re_N-1]
            real_residual_arr = z_arr.real - z_sim_arr.real
            return real_residual_arr / np.abs(z_arr)
        elif self.obj_fun_mode == 'both':
            # both_residual_arr = [△Re0, △Re1, ..., △Re_N-1, △Im0, △Im1, △Im2, ..., △Im_N-1]
            both_residual_arr = z_arr - z_sim_arr
            
            arr = np.zeros(shape=(both_residual_arr.shape[0] * 2,))
            
            real_residual_arr = both_residual_arr.real / np.abs(z_arr)
            imag_residual_arr = both_residual_arr.imag / np.abs(z_arr)
            
            arr[:both_residual_arr.shape[0]] += real_residual_arr
            arr[both_residual_arr.shape[0]:] += imag_residual_arr
            return arr

    def cal_residual_derivative(self, para_index, timeConstant_arr=None):
        # numpy.copy is a shallow copy, copy.deepcopy() is more preferred
        new_big_para_arr = copy.deepcopy(self.para_arr)
        new_big_para_arr[para_index] += 1e-8
        new_small_para_arr = copy.deepcopy(self.para_arr)
        new_small_para_arr[para_index] -= 1e-8

        # if self.obj_fun == ecm_obj_fun_1:
        #     big_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_big_para_arr,
        #                               ecm_serial=self.ecm_serial, z_arr=self.impSpe.z_arr)
        #     small_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_small_para_arr,
        #                                 ecm_serial=self.ecm_serial, z_arr=self.impSpe.z_arr)
        if self.obj_fun == ecm_obj_fun:
            big_output = self.cal_residual(para_arr=new_big_para_arr, timeConstant_arr=timeConstant_arr)
            small_output = self.cal_residual(para_arr=new_small_para_arr, timeConstant_arr=timeConstant_arr)
            # big_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_big_para_arr, ecm_serial=self.ecm_serial)
            # small_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_small_para_arr, ecm_serial=self.ecm_serial)

        elif self.obj_fun == vogit_obj_fun_1:
            big_output = self.cal_residual(para_arr=new_big_para_arr, timeConstant_arr=timeConstant_arr)
            small_output = self.cal_residual(para_arr=new_small_para_arr, timeConstant_arr=timeConstant_arr)

        if self.obj_fun_mode == 'both':
            residual_derivative_arr = (big_output - small_output) / (2 * 1e-8)
            return residual_derivative_arr

        #     # derivative_arr 是阻抗 复数
        #     arr = np.zeros(shape=(residual_derivative_arr.shape[0] * 2,))
        #     # 是否要除以|Z|?
        #     arr[: residual_derivative_arr.shape[0]] += residual_derivative_arr.real
        #     arr[residual_derivative_arr.shape[0]:] += residual_derivative_arr.imag
        #     return arr
        # else:
        #     # derivative_arr 是阻抗的实部或者虚部 是实数
        #     return residual_derivative_arr
        # pass

    def cal_derivative(self, para_index, timeConstant_arr=None):
        """
        **derivative = [f(x+d_x) - f(x-d_x)] / (2 * d_x)
        or
        derivative = [f(x+d_x) - f(x)] / d_x
        :param
            timeConstant_arr,只有计算vogit时采用的上，计算普通ecm时为None
        :return:
        """
        # numpy.copy is a shallow copy, copy.deepcopy() is more preferred
        new_big_para_arr = copy.deepcopy(self.para_arr)
        new_big_para_arr[para_index] += 1e-8
        new_small_parr_arr = copy.deepcopy(self.para_arr)
        new_small_parr_arr[para_index] -= 1e-8

        if self.obj_fun == ecm_obj_fun_1:
        # if self.obj_fun == ecm_obj_fun:
        #     big_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_big_para_arr, ecm_serial=self.ecm_serial)
        #     small_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_small_parr_arr, ecm_serial=self.ecm_serial)
            big_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_big_para_arr,
                                      ecm_serial=self.ecm_serial, z_arr=self.impSpe.z_arr)
            small_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_small_parr_arr,
                                        ecm_serial=self.ecm_serial, z_arr=self.impSpe.z_arr)
        elif self.obj_fun == vogit_obj_fun_1:
            big_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_big_para_arr, tao_arr=timeConstant_arr,
                                      obj_fun_mode=self.obj_fun_mode, add_C=self.vogit_add_C)
            small_output = self.obj_fun(w_arr=self.impSpe.w_arr, para_arr=new_small_parr_arr, tao_arr=timeConstant_arr,
                                        obj_fun_mode=self.obj_fun_mode, add_C=self.vogit_add_C)
        
        derivative_arr = (big_output - small_output) / (2 * 1e-8)
        if self.obj_fun_mode == 'both':
            # derivative_arr 是阻抗 复数
            arr = np.zeros(shape=(derivative_arr.shape[0] * 2, ))
            # 是否要除以|Z|?
            arr[: derivative_arr.shape[0]] += derivative_arr.real
            arr[derivative_arr.shape[0]: ] += derivative_arr.imag
            # arr[: derivative_arr.shape[0]] += derivative_arr.real / np.abs(self.impSpe.z_arr)
            # arr[derivative_arr.shape[0]: ] += derivative_arr.imag / np.abs(self.impSpe.z_arr)
            return arr
        else:
            # derivative_arr 是阻抗的实部或者虚部 是实数
            return derivative_arr

    def cal_Jacobian(self, timeConstant_arr=None):
        """
        Jacobian Matrix: M rows * N cols
            M = data point number
            N = para number
        :argument
        :param
            timeConstant_arr,只有计算vogit时采用的上，计算普通ecm时为None
        :return:
        """
        z_arr = self.impSpe.z_arr
        M = z_arr.shape[0]
        N = self.para_arr.shape[0]
        if self.obj_fun_mode == 'both':
            jacob_arr = np.zeros((2 * M, N))
        else:
            jacob_arr = np.zeros((M, N))

        for i in range(N):
            # ValueError: could not broadcast input array from shape (196) into shape (98)
            jacob_arr[:, i] = self.cal_residual_derivative(para_index=i, timeConstant_arr=timeConstant_arr)

            # jacob_arr[:, i] 表示jacob_arr所有行的第i列即 df(x)/dxi那一列的结果
            # jacob_arr[:, i] = self.cal_derivative(para_index=i, timeConstant_arr=timeConstant_arr)
        return jacob_arr

    def stop_criteria_checker(self, e1, e2) -> bool:
        if (e1 <= self.e1) or (e2 <= self.e2) or (self.iter_count >= self.iter_max):
            return True
        else:
            return False

    def iterate(self, timeConstant_arr=None):
        v = 2
        jacob_arr = self.cal_Jacobian(timeConstant_arr)
        A = jacob_arr.T.dot(jacob_arr)

        # defined in paper0 eq 3.14
        mu = self.tao * max([A[i, i] for i in range(A.shape[0])])

        g = jacob_arr.T.dot(self.cal_residual(para_arr=self.para_arr, timeConstant_arr=timeConstant_arr).reshape(jacob_arr.shape[0],1))

        found = np.linalg.norm(g, ord=np.inf) <= self.e1
        iter_count = 0
        while (not found) and (iter_count < self.iter_max):
            hessian_LM_arr = A + mu * np.eye(A.shape[0])
            # h同时包含参数更新的大小和方向
            h = np.linalg.inv(hessian_LM_arr).dot(-g)
            # h = np.linalg.inv(hessian_LM_arr).dot(g)

            if (np.linalg.norm(h, ord=2) <= self.e2):
                found = True
                break
            else:
                # cal gain ratio, defined in paper0 eq 2.18
                # F(x) = 0.5 * || f(x) ||
                F_0 = 0.5 * (np.linalg.norm(self.cal_residual(para_arr=self.para_arr, timeConstant_arr=timeConstant_arr), ord=2) ** 2)
                new_para_arr = self.para_arr + h.ravel()
                F_h = 0.5 * (np.linalg.norm(self.cal_residual(para_arr=new_para_arr, timeConstant_arr=timeConstant_arr), ord=2) ** 2)
                # cal L(0) - L(h), defined in paper0 eq 3.14的下方
                L0_minus_Lh = 0.5 * h.T.dot(mu * h - g)
                # L0_minus_Lh = 0.5 * h.T.dot(mu * h + g)

                rou = (F_0 - F_h) / L0_minus_Lh

                # If the updated parameter vector leads to a reduction in the residual, the update is accepted
                if rou > 0: # accept h (step)
                    self.para_arr = new_para_arr

                    # update Jacobian, A, g
                    jacob_arr = self.cal_Jacobian(timeConstant_arr)
                    A = jacob_arr.T.dot(jacob_arr)
                    g = jacob_arr.T.dot(self.cal_residual(para_arr=self.para_arr, timeConstant_arr=timeConstant_arr).reshape(jacob_arr.shape[0],1))

                    found1 = np.linalg.norm(g, ord=np.inf) <= self.e1
                    found2 = np.linalg.norm(self.cal_residual(para_arr=self.para_arr, timeConstant_arr=timeConstant_arr), ord=2) <= self.e3
                    if found1 or found2:
                        break

                    # update mu, v
                    mu = mu * max(1/3, 1 - (2 * mu - 1) ** 3)
                    v = 2
                    print('mu, v', mu, v)
                else:
                    mu = mu * v
                    v = 2 * v

            iter_count += 1
            # if iter_count % 20 == 0:
            if iter_count % 200 == 0:
                print('LM-iter:', iter_count)

# impSpe = IS()
# M = 1
# vogit = Vogit(M=M, impSpe=impSpe)
# lm = Levenberg_Marquart_0(impSpe, obj_fun, mode='validate', iter_max=1000)
# lm.get_initial_para_arr()