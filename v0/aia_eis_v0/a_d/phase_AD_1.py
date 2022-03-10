import copy
import os
import numpy as np

from circuits.ecm import ECM
from circuits.vogit_1 import Vogit_3
from IS.IS import IS_0
from smoothAlg.SavitzkyGolay import savitzkyGolay as SG

from a_d.ny_onePoint_AD import load_EIS, getMaxAndIndex

from utils.file_utils.pickle_utils import load_pickle_file
from utils.statistic.quartile import cal_quartile
from utils.visualize_utils.IS_plots.bd import bode_absZ
from utils.visualize_utils.plot_utils import plot_2D_util

"""
模块功能：
    经过对各种检验指标结果的比较，得出检测效果如下：
        [Ny+|ε_Im|+Q2 = 47/57 = 82.46%] 》
        [三种指标集成 = 46/57 = 80.7%] == [Bd-|Z|+|ε|+Q2]
        》 [Bd-Phase+|ε|+Q2]
    [Bd-Phase + |ε|+Q2]
    version
        1：将之前的代码 简化 + 优化
        0：phase_AD_0.py
"""

def calPhase(z):
    """
    :param
        z: 3+4j
    :return:
        phase: 45°
    """
    phase = np.arctan2(z.imag, z.real) * 180 / np.pi
    return phase

def detect(eis_source, vogitAddC, pointNum=3, chiSquareLimit=1e-5, printFlag=True):
    """
    Function
        根据之前的实验结果得到一下结论
            Bd-Phase + 【Q2 + |▲ε|】 有第3佳的检测结果
    :param
        eis_source
            ecm_fn:
            eis obj
        vogitAddC: bool
            EIS沿着Zreal收敛，不加C，False
            EIS沿着Zreal不收敛，加C，True
        pointNum: int
            the number of outliers you want to detect
        chiSquareLimit: float
            default 1e-5
        printFlag: Bool
            是否打印中间结果，调试时打印True，部署时不打印False
    :return:
    """

    if isinstance(eis_source, str):
        if pointNum == 1:
            eis = load_EIS(fn=eis_source, fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)(RW)_pkl/oneOutlier/')
        elif pointNum == 2:
            eis = load_EIS(fn=eis_source, fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)(RW)_pkl/twoOutliers/')
        elif pointNum == 3:
            eis = load_EIS(fn=eis_source, fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)(RW)_pkl/threeOutliers/')

    elif isinstance(eis_source, IS_0):
        eis = eis_source

    else:
        import sys
        print('detect中ecm_fn的参数给错了，查查')
        sys.exit()

    pointCount = 0
    chiSquare = 1.0
    deletedPointIndex_list = []

    while (pointCount < pointNum) and (chiSquare > chiSquareLimit):
        tmp_eis = copy.deepcopy(eis)

        for dpi in deletedPointIndex_list:
            tmp_eis.removeZByIndex(index=dpi)

        z_arr = tmp_eis.z_arr
        z_SG_arr = SG(z_arr, loop_time=20, convPoints=5)

        phase_arr = calPhase(z_arr)
        phase_SG_arr = calPhase(z_SG_arr)

        # |▲Phase| = | (Phase - Phase_SM) / Phase |
        # d_phase_arr = np.array([np.abs((phase - phase_SG) / phase) for phase, phase_SG in zip(phase_arr, phase_SG_arr)])
        d_phase_arr = np.abs([(phase - phase_SG) / phase for phase, phase_SG in zip(phase_arr, phase_SG_arr)])

        # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------
        vogit = Vogit_3(impSpe=tmp_eis, fit_type='complex', u_optimum=0.85, add_C=vogitAddC, M_max=None)
        vogit.lin_KK()
        vogit.cal_residual()
        residual_arr = vogit.residual_arr
        vogit_chi_square = vogit.cal_chiSquare(weight_type='modulus')
        chiSquare = vogit_chi_square

        if (pointCount == 0) and (chiSquare <= chiSquareLimit):
            print('Lin-KK ChiSquares Raw of EIS:', vogit_chi_square, 'Good Data Quality, No Outlier')
            break
        elif chiSquare <= chiSquareLimit:
            print('Lin-KK ChiSquares of EIS:', vogit_chi_square,
                  'Normal Data Quality, Outlier Index List:', deletedPointIndex_list)
            break
        if printFlag:
            print('Lin-KK ChiSquares of EIS:', vogit_chi_square)
        # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------

        maxAndIndex_list = getMaxAndIndex(arr=d_phase_arr)
        # -------------------------------------- 使用 四分位-Q2 过滤一批点 --------------------------------------
        qua_info_list = cal_quartile(arr=d_phase_arr)
        low_boundary, Q1, Q2, Q3, up_boundary = qua_info_list
        if printFlag:
            print('--------------\n|▲Z|: low_boundary, Q1, Q2, Q3, up_boundary')
            print(low_boundary, Q1, Q2, Q3, up_boundary)

        """
        因为Smooth（Conv=5）的时候，会舍弃曲线最开始和最末尾的各两个点
            例如：lai-EIS一条曲线有31个点【Z0，Z1，Z2，，，，Z29，Z30】，其中【Z0，Z1，Z29，Z30】因为没有被平滑，
                这四个点对应的下|ε_Smooth_Im| = 0，在使用四分位进行异常点预筛选的时候一定会被忽略，
                所以要把maxAndIndex中最后的四个点加到ny_maxAndIndex_list最前方的位置
        之前在理想的实验数据中没有发现这个问题的原因：
            理想数据中的异常点并不设计曲线开头和结果的几个点，所以对理想数据中的实验结果没有影响
        """
        maxAndIndex_list = [maxAndIndex for maxAndIndex in maxAndIndex_list if (maxAndIndex[0] > Q2) or (maxAndIndex[0] == 0.0)]
        # -------------------------------------- 使用 四分位-Q2 过滤一批点 --------------------------------------

        # 遍历 所有可疑的点
        dif_list = []
        chiSquare_list = []
        if printFlag:
            print('------------- Index and Difference and chiSquare For Bd-|▲Phase| --------------------')
        for maxAndIndex in maxAndIndex_list:
            maximum, index = maxAndIndex

            # -------------- delete one point --------------
            eis_delete_onePoint = copy.deepcopy(tmp_eis)
            eis_delete_onePoint.removeZByIndex(index)
            # -------------- delete one point --------------

            vogit_delete_onePoint = Vogit_3(impSpe=eis_delete_onePoint, fit_type='complex',
                                            u_optimum=0.85, add_C=vogitAddC, M_max=None)
            vogit_delete_onePoint.lin_KK()
            vogit_delete_onePoint.simulate_Z()
            vogit_delete_onePoint.cal_residual()
            vogit_delete_onePointResidual_arr = vogit_delete_onePoint.residual_arr

            # ∑|Residual|
            absResidual_dif = np.sum(np.abs(residual_arr)) - np.sum(np.abs(vogit_delete_onePointResidual_arr))
            if printFlag:
                tmp_chi_square = vogit_delete_onePoint.cal_chiSquare()
                chiSquare_list.append(tmp_chi_square)
                print('Index:', index, '∑|ε_beforeDeletion| - ∑|ε_afterDeletion|:', absResidual_dif,
                      'chiSquares:', tmp_chi_square)
            dif_list.append(absResidual_dif)

        if printFlag:
            sorted_dif_list = sorted(dif_list, reverse=True)  # Descending order
            sorted_chiSquare_list = sorted(chiSquare_list, reverse=False)  # Ascending order
            print(
                '------------- ∑|ε_beforeDeletion| - ∑|ε_afterDeletion| Order (Big->Small) + ChiSquare Order (Small->Big) --------------------')
            print(
                '∑|ε_beforeDeletion| - ∑|ε_afterDeletion|, sorted_dif_list.index(dif), chiS, sorted_chiSquare_list.index(chiS)')
            for dif, chiS in zip(dif_list, chiSquare_list):
                print(dif, sorted_dif_list.index(dif), chiS, sorted_chiSquare_list.index(chiS))

        # find and return the index of the maximum value in dif_list
        max_dif = max(dif_list)
        max_dif_index = dif_list.index(max_dif)
        deletedPointIndex = maxAndIndex_list[max_dif_index][1]
        deletedPointIndex_list.append(deletedPointIndex)
        pointCount += 1

        if printFlag:
            print('--------------\nThe index of the {0} outlier is: {1}'.format(pointCount, deletedPointIndex))
    return deletedPointIndex_list