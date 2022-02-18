from collections import OrderedDict
import math
import numpy as np

from circuits.ecm import ECM
from circuits.circuit_pack import ecm_oldSeq_2_newSerial
from utils.file_utils.pickle_utils import load_pickle_file

"""
文件目录
IS
    IS
    IS_source
在IS.IS中按照【from IS.IS_source import IS_scource】是无法导入IS.IS_scource的，应为IS.IS和IS.IS_scource同在一个大包下，
应该按照【from IS_source import IS_scource】导入IS.IS_scource
"""
from IS.IS_source import IS_scource
from utils.file_utils.pickle_utils import load_pickle_file

class IS_0:
    def __init__(self):
        """
        raw_z: np.array(complex)
            complex impedance
        exp_area: float
            IS experimental area
            unit == cm^2
        z: np.array(complex)
            == raw_z * exp_area
            complex impedance
        fre: np.array(float)
            frequency
        source: cls IS_source
            records where this IS comes from?
                experiment or paper or simulation
        ---------------------------------------------
        ecm_cls_list: list[
                        ecm-0: ecm class
                        ecm-1: ecm class
                        ecm-2: ecm class
                        ...
                            ]
                        记录这个IS可能的几个ECM，按照概率从大(index=0)到小(index=-1)的顺序，如果已经人工确定了一个ECM，
                        那该ECM的Proba设为1
        -------------------
        ecm_list: list[OrderedDict-0{
                        'ecm': str,
                            like:
                                '2_001' means ECM serial excel
                        'proba':
                        'limit':,
                        'para':,
                        'z_sim':},
                       OrderedDict-0{
                        'ecm':,
                        'limit':,
                        'para':,
                        'z_sim':},
                    ]
                记录这个IS可能的几个ECM，按照概率从
        -------------------
        ecm_dict: OrderedDict{

            }
        ---------------------------------------------
        note_dict: OrderedDict{
                'date:2021-08-02':'blablablablablabla',
                'date:2021-05-02':'blablablablablabla',
            }
            records notes
        """
        self.raw_z_arr = None
        self.exp_area = None
        self.z_arr = None
        self.fre_arr = None
        self.w_arr = None
        self.source = None
        self.ecm_cls_list = []
        self.note_dict = OrderedDict()
        # self. = None
        # self. = None
        # self. = None

    def calc_z(self):
        if (self.raw_z_arr is not None) and (self.exp_area is not None):
            self.z_arr = self.raw_z_arr * self.exp_area

    def read_from_mysql(self):
        pass

    # ----------------- 参照 《Impedance-preprocessing》模块去写 -----------------
    def read_from_AutoLab_File(self):
        pass
    def read_from_BioLogic_File(self):
        pass
    def read_from_Parstat_File(self):
        pass
    def read_from_VersaStudio_File(self):
        pass
    def read_from_ZPlot_File(self):
        pass
    def read_from_PowerSuite_File(self):
        pass
    def read_from_CHInstruments_File(self):
        pass
    def read_from_DTA_File(self):
        # can refer <impedance> and <pyEIS>
        pass
    # ----------------- 参照 《Impedance-preprocessing》模块去写 -----------------

    def readFromLaiPickle(self, laiNormedEisDict, limitList=None):
        """
        Function
            专门读取Lai的Pickle Dict文件，文件结构
            这些阻抗数据 已经 乘以 实验面积 1.01 * 1e6 cm^2
        :param
            laiNormedEisDict{
                'file_name': '1-1',
                'ecm_num': 9,
                'f': [100078.1, 63140.62, ..., 0.1588983, 0.1001603],
                'z_raw': [(0.005566658429999999-0.0112022736j),
                          (0.006214947129999999-0.0172324988j),
                          ...,
                          (285.52881799999994-486.4391289999999j),
                          (370.64242699999994-661.259928j)]
            }
        :return:
        """
        ecm_num = laiNormedEisDict['ecm_num']
        freArr = np.array(laiNormedEisDict['f'])
        rawZList = laiNormedEisDict['z_raw']

        if limitList is not None:
            limit_arr = np.array(limitList)
        else:
            limit_arr = None

        ecm_serial = ecm_oldSeq_2_newSerial(ecm_num=ecm_num)
        ecm_proba = 1.0

        ecm = ECM(ecm_serial, proba=ecm_proba, fre=freArr, limit=limit_arr, z_sim=rawZList)
        self.ecm_cls_list.append(ecm)

        # 这些阻抗数据 已经 乘以 实验面积 1.01 * 1e6 cm^2
        self.raw_z_arr = np.array(rawZList)
        self.z_arr = np.array(rawZList)
        self.exp_area = 1.0

        self.fre_arr = freArr
        self.w_arr = self.fre_arr * 2 * math.pi

        is_source = IS_scource()
        self.source = is_source.fill4Experiment(commercial=False, software='AIA-EIS-v0')

    def read_from_simPickle(self, fp, fn):
        """
        主要用于读取在DPFC工作中模拟的9个理想的IS数据
        read simulated EIS stored in pickle format
            fp: around dpfc_src\datasets\goa_datasets\simulated
        :return:
        """
        data_dict = load_pickle_file(fp, fn)
        ecm_num = data_dict['ecm_num']
        limit_list = data_dict['limit']
        para_list = data_dict['para']
        fre_list = data_dict['f']
        z_sim_list = data_dict['z_sim']

        ecm_serial = ecm_oldSeq_2_newSerial(ecm_num=ecm_num)
        ecm_proba = 1.0

        ecm = ECM(ecm_serial, proba=ecm_proba, fre=fre_list, limit=limit_list, para=para_list, z_sim=z_sim_list)
        self.ecm_cls_list.append(ecm)

        self.raw_z_arr = np.array(z_sim_list)
        self.z_arr = np.array(z_sim_list)
        self.exp_area = 1.0
        self.fre_arr = np.array(fre_list)
        self.w_arr = self.fre_arr * 2 * math.pi

        is_source = IS_scource()
        self.source = is_source.fill4Simulation(commercial=False, software='AIA-EIS-v0')

    def read_from_EcmCls(self, fp, fn):
        """
        用于RBP-EIS prj的模拟数据(jupyter lab写代码生成的)都以ECM class (EcmCls) 的形式保存，在此加载
        :return:
        """
        # load ECM cls pkl file
        ecm = load_pickle_file(fp, fn)

        self.ecm_cls_list.append(ecm)

        # self.raw_z_arr = np.array(ecm.z_sim_list)
        # self.z_arr = np.array(ecm.z_sim_list)
        self.raw_z_arr = ecm.z_sim_arr
        self.z_arr = ecm.z_sim_arr

        self.exp_area = 1.0
        self.fre_arr = np.array(ecm.fre_arr)
        self.w_arr = self.fre_arr * 2 * math.pi

        is_source = IS_scource()
        self.source = is_source.fill4Simulation(commercial=False, software='AIA-EIS-v0')

    def removeZByIndex(self, index):
        """
        在删除异常点时，要raw_z_arr，z_arr，fre_arr，w_arr在判断不为空后，都按照索引删除元素
        index: int or [int]
        """
        if self.raw_z_arr is not None:
            self.raw_z_arr = np.delete(self.raw_z_arr, index)
        if self.z_arr is not None:
            self.z_arr = np.delete(self.z_arr, index)
        if self.fre_arr is not None:
            self.fre_arr = np.delete(self.fre_arr, index)
        if self.w_arr is not None:
            self.w_arr = np.delete(self.w_arr, index)

    def f2(self):
        pass
    def f(self):
        pass
    def f3(self):
        pass
    def f4(self):
        pass
    def f5(self):
        pass

# eis = IS_0()
# eis.read_from_simPickle(fp='..\datasets\goa_datasets\simulated\ecm_001',
#                         fn='2020_07_04_sim_ecm_001_pickle.file')

# from circuits.ecm_sp import Vogit
#
# RaRCb_IS = IS_0()
# RaRCb_IS.read_from_EcmCls(fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)_ecm_pkl/', fn='2021_08_04_ecm.pkl')
#
# RaRCb_vogit = Vogit(impSpe=RaRCb_IS)
#
# OA_obj_fun_mode = 'imag'
# print(OA_obj_fun_mode)
# RaRCb_vogit.lin_KK(OA_obj_fun_mode=OA_obj_fun_mode, save_iter=True)
# print('M=', RaRCb_vogit.M, 'u=',RaRCb_vogit.u)