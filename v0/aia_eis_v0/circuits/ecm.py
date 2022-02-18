import math
import numpy as np
import pandas as pd
import sys

from circuits.circuit_pack import *
from circuits.ecm_simulator import ecm_simulator_1
from loa.l_m.l_m_0 import Levenberg_Marquart_0, ecm_obj_fun, ecm_obj_fun_1
from utils.frequency_generator import fre_generator
from utils.visualize_utils.IS_plots.bd import calPhase

def ecm_serial_matcher(ecm_serial: str):
    # 根据 ecm_serial 查找 dpfc_src\circuits\ecm_serial.xlsx中sheet【formal】对应的 CDC
    ecm_serial_sheet = pd.read_excel('ecm_serial.xlsx', sheet_name='formal')
    ecm_serial_Series = ecm_serial_sheet['Serial']
    ecm_fun_name_Series = ecm_serial_sheet['Function_Name']

    # find index in ecm_serial_Series by searching 'ecm_serial'
    # 根据值获取索引 like: data_index = Int64Index([6], dtype='int64')
    data_index = ecm_serial_Series[ecm_serial_Series.values == ecm_serial].index

    # ecm_fun_name pandas.core.series.Series
    # ecm_fun_name.values = array(['RQ'], dtype=object)
    ecm_fun_name = ecm_fun_name_Series[data_index]
    ecm_fun_name = ecm_fun_name.values[0]
    ecm_function = eval(ecm_fun_name)
    return ecm_function
# ecm_function = ecm_serial_matcher(ecm_serial='2_001')

class ECM():
    def __init__(self, ecm_serial, proba, fre=None, phase=None, limit=None, para=None, z_sim=None):
        """
        ecm_serial: str,
            like:
                '2_001' means ECM serial in excel <dpfc_src\circuits\ecm_serial.xlsx --> sheet:'formal'>
        proba: float
            0 ~ 1.0
            如果已经人工确定了一个ECM，那该ECM的Proba设为1.
            如果是ML预测的概率，那就直接记录该数值
        limit_list: list[para0:(low_boundary, up_boundary),
                         para1:(low_boundary, up_boundary),
                         ...],
                    tuple相比list应该占用资源更小一些
        para_list: list[float or complex],
        'z_sim':
        """
        self.ecm_serial = ecm_serial
        self.proba = proba

        if fre is not None: # not None
            if type(fre) == list:
                self.fre_arr = np.array(fre)
            else:
                self.fre_arr = fre
            self.w_arr = self.fre_arr * 2 * math.pi
        else:
            self.fre_arr = None
            self.w_arr = None

        if limit is not None:
            if type(limit) == list:
                self.limit_arr = np.array(limit)
            else:
                self.limit_arr = limit
        else:
            self.limit_arr = None

        if para is not None: # not None
            if type(para) == list:
                self.para_arr = np.array(para)
            else:
                self.para_arr = para
        else:
            self.para_arr = None

        # ValueError: The truth value of an array with more than one element is ambiguous.
        # Use a.any() or a.all()
        if z_sim is not None:
            if type(z_sim) == list:
                self.z_sim_arr = np.array(z_sim)
            else:
                self.z_sim_arr = z_sim
        else:
            self.z_sim_arr = None

        if phase is not None:
            if type(phase) == list:
                self.phase_arr = np.array(phase)
            elif self.z_sim_arr is not None:
                # 相位角
                self.phase_arr = calPhase(self.z_sim_arr)
        else:
            self.phase_arr = None

    def modify_para_limit_range(self, index:int, limit_para:tuple):
        """
        当某个参数的上下界范围设置不合理时，根据索引找到limit_list中的该参数的范围 进行替换修改
        :return:
        """
        pass

    def init_para(self):
        """
        Function
            初始化每个原件的数值
                如果有参数取值范围，在取值范围内随机选取一个数
                如果没有参数取值范围，参数按照默认范围内设置
        :return:
        """
        if self.limit_arr is not None:
            self.para_arr = np.empty(shape=(self.limit_arr.shape[0], ), dtype=float)
            for i, limit in enumerate(self.limit_arr):
                minimum, maximum = limit
                para = (maximum - minimum) * np.random.random(1) + minimum
                self.para_arr[i] = para
        else:
            pass

    def identify_para(self, IS, OA=Levenberg_Marquart_0):
        self.init_para()

        # init Levenberg_Marquardt
        # OA: Optimization Algorithm
        oa = OA(impSpe=IS,
                # obj_fun=ecm_obj_fun_1,
                obj_fun=ecm_obj_fun,
                obj_fun_mode='both',
                obj_fun_weighting_type='modulus',
                iter_max=2000,
                ecm_serial=self.ecm_serial)

        oa.get_initial_para_arr(para_arr=self.para_arr)
        oa.iterate()
        self.para_arr = oa.para_arr

    # def simulate(self, ecm_serial:str, fre):
    def simulate_Z(self, fre_setting=None, para_list=None):
        """
        :param
            fre_setting:
                Ex: {'start': 5, 'end': -1, 'points': 5} == Fre: 1e5 ~ 1e-1
            para_list:
        :return:
        """
        if fre_setting is not None:
            fre_list, w_list = fre_generator(f_start=fre_setting['start'],
                                             f_end=fre_setting['end'],
                                             pts_decade=fre_setting['points'])
            self.fre_arr = np.array(fre_list)
            self.w_arr = np.array(w_list)

            if para_list is not None:
                self.para_arr = np.array(para_list)

        z_sim_list = ecm_simulator_1(ecm_serial=self.ecm_serial, para_list=self.para_arr.tolist(),
                                     w=self.w_arr.tolist())
        self.z_sim_arr = np.array(z_sim_list)