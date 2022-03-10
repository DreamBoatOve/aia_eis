import copy
import os
import numpy as np

from circuits.ecm import ECM
from circuits.vogit_1 import Vogit_3
from IS.IS import IS_0
from smoothAlg.SavitzkyGolay import savitzkyGolay as SG
from utils.file_utils.pickle_utils import load_pickle_file
from utils.statistic.quartile import cal_quartile
from utils.visualize_utils.IS_plots.ny import nyquist_multiPlots_1, nyquist_plot_1
from utils.visualize_utils.IS_plots.residuals import real_residual_plot, residuals_plot

"""
oneOutlierDetection
Routine of 单点异常实验
    1- SM先拟合一条，计算残差，找出最可能的异常点
    2- 按照残差从大到小的顺序 依次尝试
        2.1- 每次去掉一个点
        
Note
    Outlier index:
        High=5/middle=25/low=55        
"""

def load_EIS(fn, fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)(RW)_pkl/oneOutlier/'):
    eis = IS_0()
    eis.read_from_EcmCls(fp, fn)
    return eis

def getMaxAndIndex(arr, printFlag=False):
    # 先把arr首尾的四个点剔除出去
    tmp_arr = np.delete(arr, [0, 1, arr.size-2, arr.size-1])

    # sortedArr = np.sort(arr) # small --> big
    sortedArr = np.sort(tmp_arr)[::-1] # big --> small
    maxAndIndex_list = []
    """
    如果直接按照 np.where(arr == sortedArr[i]) 搜索索引，由于SG平滑会 跳过最开始和最末尾的攻击四个点，所以会出现以下情况
        0 0.005374655031014165 (array([14], dtype=int64),)
        1 0.004724367449809488 (array([5], dtype=int64),)
        。。。
        54 1.353377835486382e-07 (array([56], dtype=int64),)
        55 1.0749574187690286e-07 (array([57], dtype=int64),)
        **56 0.0 (array([ 0,  1, 58, 59], dtype=int64),)**
        **57 0.0 (array([ 0,  1, 58, 59], dtype=int64),)**
        **58 0.0 (array([ 0,  1, 58, 59], dtype=int64),)**
        **59 0.0 (array([ 0,  1, 58, 59], dtype=int64),)**
    """
    if printFlag:
        print('--------')
    for i in range(sortedArr.size):
        # wrong 忘记开头 SG省去的两个点了
        # print(sortedArr[i], np.where(tmp_arr == sortedArr[i])[0][0])
        # Right
        if printFlag:
            print(np.where(tmp_arr == sortedArr[i])[0][0]+2, sortedArr[i])
        maxAndIndex_list.append([sortedArr[i], np.where(tmp_arr == sortedArr[i])[0][0] + 2])

    for i in [0,1,arr.size-2,arr.size-1]:
        if printFlag:
            print(i, arr[i])
        maxAndIndex_list.append([arr[i], i])
    return maxAndIndex_list

def detect(eis_source, criteria_type='absRealAndAbsImag', qua_flag=False, Q_flag='Q3'):
    """
    :param
        eis_source
            ecm_fn:
            eis obj
        criteria_type: str
            absRe == |▲Re|,
            absIm == |▲Im|,
            absRealAndAbsImag == |▲Re| + |▲Im|,
        qua_flag bool
            True    使用 四分位 过滤一批点
            False   不使用 四分位 过滤一批点
    :return:
    """
    if isinstance(eis_source, str):
        eis = load_EIS(fn=eis_source)
    elif isinstance(eis_source, IS_0):
        eis = eis_source
    else:
        import sys
        print('detect中ecm_fn的参数给错了，查查')
        sys.exit()

    z_arr = eis.z_arr
    z_SG_arr = SG(z_arr, loop_time=20, convPoints=5)

    # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------
    vogit = Vogit_3(impSpe=eis, fit_type='complex', u_optimum=0.85, add_C=True, M_max=None)
    vogit.lin_KK()
    vogit.cal_residual()
    residual_arr = vogit.residual_arr
    vogit_chi_square = vogit.cal_chiSquare(weight_type='modulus')
    print('Chi Square:', vogit_chi_square)
    # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------

    residual_SG_arr = (z_arr - z_SG_arr) / np.abs(z_arr)

    # |▲Re|
    residual_SG_absReal_arr = np.abs(residual_SG_arr.real)
    residual_absReal_arr = np.abs(residual_arr.real)

    # |▲Im|
    residual_SG_absImag_arr = np.abs(residual_SG_arr.imag)
    residual_absImag_arr = np.abs(residual_arr.imag)

    # |▲Re| + |▲Im|
    residual_SG_absRealAndAbsImag_arr = residual_SG_absReal_arr + residual_SG_absImag_arr
    residual_absRealAndAbsImag_arr = residual_absReal_arr + residual_absImag_arr

    # |Residual|
    residual_SG_absZ_arr = np.abs(residual_SG_arr)
    residual_absZ_arr = np.abs(residual_arr)

    if criteria_type == 'absRe':
        print('-----------------\n','|▲Re|')
        # print the maximums and their index
        # maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absReal_arr, num=residual_SG_absReal_arr.size)
        maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absReal_arr)
    elif criteria_type == 'absIm':
        print('-----------------\n', '|▲Im|')
        # maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absImag_arr, num=residual_SG_absImag_arr.size)
        maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absImag_arr)
    elif criteria_type == 'absRealAndAbsImag':
        print('-----------------\n', '|▲Re| + |▲Im|')
        # maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absRealAndAbsImag_arr, num=residual_SG_absRealAndAbsImag_arr.size)
        maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absRealAndAbsImag_arr)
    elif criteria_type == 'absResidual':
        print('-----------------\n', '|Residual|')
        # maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absZ_arr, num=residual_SG_absZ_arr.size)
        maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absZ_arr)

    # v1 - 现在考虑所有点，就不用考虑SG算法首尾遗留下来的4个点了
    # v0 - 因为SG的convPoints=5，所以 首尾 各有2个点，共4个点没有进行平滑，要加进去

    # ------------------------------------------- 使用 四分位 过滤一批点 -------------------------------------------
    if qua_flag:
        if criteria_type == 'absRe':
            qua_info_list = cal_quartile(arr=residual_SG_absReal_arr)
        elif criteria_type == 'absIm':
            qua_info_list = cal_quartile(arr=residual_SG_absImag_arr)
        elif criteria_type == 'absRealAndAbsImag':
            qua_info_list = cal_quartile(arr=residual_SG_absRealAndAbsImag_arr)
        elif criteria_type == 'absResidual':
            qua_info_list = cal_quartile(arr=residual_SG_absZ_arr)

        low_boundary, Q1, Q2, Q3, up_boundary = qua_info_list
        print('--------------\nlow_boundary, Q1, Q2, Q3, up_boundary')
        print(low_boundary, Q1, Q2, Q3, up_boundary)
        if Q_flag == 'up_boundary':
            print('---------Max and Index Filtered by up_boundary----------')
            maxAndIndex_list = [maxAndIndex for maxAndIndex in maxAndIndex_list if maxAndIndex[0] > up_boundary]
        elif Q_flag == 'Q3':
            print('---------Max and Index Filtered by Q3----------')
            maxAndIndex_list = [maxAndIndex for maxAndIndex in maxAndIndex_list if maxAndIndex[0] > Q3]
        elif Q_flag == 'Q2':
            print('---------Max and Index Filtered by Q2----------')
            maxAndIndex_list = [maxAndIndex for maxAndIndex in maxAndIndex_list if maxAndIndex[0] > Q2]
        elif Q_flag == 'Q1':
            print('---------Max and Index Filtered by Q1----------')
            maxAndIndex_list = [maxAndIndex for maxAndIndex in maxAndIndex_list if maxAndIndex[0] > Q1]
        elif Q_flag == 'low_boundary':
            print('---------Max and Index Filtered by low_boundary----------')
            maxAndIndex_list = [maxAndIndex for maxAndIndex in maxAndIndex_list if maxAndIndex[0] > low_boundary]

        for maxAndIndex in maxAndIndex_list:
            maximum, index = maxAndIndex
            print(index, maximum)
    # ------------------------------------------- 使用 四分位 过滤一批点 -------------------------------------------

    # 遍历 所有可疑的点
    dif_list = []
    chiSquare_list = []
    print('-------------Index anf Difference--------------------')
    for maxAndIndex in maxAndIndex_list:
        maximum, index = maxAndIndex
        eis_delete_onePoint = copy.deepcopy(eis)
        eis_delete_onePoint.raw_z_arr = np.delete(eis_delete_onePoint.raw_z_arr, index)
        eis_delete_onePoint.z_arr = np.delete(eis_delete_onePoint.z_arr, index)
        eis_delete_onePoint.fre_arr = np.delete(eis_delete_onePoint.fre_arr, index)
        eis_delete_onePoint.w_arr = np.delete(eis_delete_onePoint.w_arr, index)

        vogit_delete_onePoint = Vogit_3(impSpe=eis_delete_onePoint, fit_type='complex',
                                        u_optimum=0.85, add_C=True, M_max=None)
        vogit_delete_onePoint.lin_KK()
        vogit_delete_onePoint.cal_residual()
        tmp_chi_square = vogit_delete_onePoint.cal_chiSquare()
        chiSquare_list.append(tmp_chi_square)
        delete_onePoint_residual_arr = vogit_delete_onePoint.residual_arr

        # 计算 并 比较 各种 残差指标
        # ∑|▲Re|
        if criteria_type == 'absRe':
            absRe_dif = np.sum(residual_absReal_arr) - np.sum(np.abs(delete_onePoint_residual_arr.real))
            # print('Index:',index,'absRe_dif:',absRe_dif)
            print(index, absRe_dif)
            dif_list.append(absRe_dif)
        # ∑|▲Im|
        elif criteria_type == 'absIm':
            absIm_dif = np.sum(residual_absImag_arr) - np.sum(np.abs(delete_onePoint_residual_arr.imag))
            print('Index:', index, 'absIm_dif:', absIm_dif)
            dif_list.append(absIm_dif)
        # ∑(|▲Re| + |▲Im|)
        elif criteria_type == 'absRealAndAbsImag':
            absRealAndAbsImag_dif = np.sum(residual_absRealAndAbsImag_arr) - np.sum(np.abs(delete_onePoint_residual_arr.real)) - np.sum(np.abs(delete_onePoint_residual_arr.imag))
            print('Index:', index, 'absRealAndAbsImag_dif:', absRealAndAbsImag_dif)
            dif_list.append(absRealAndAbsImag_dif)
        # ∑|Residual|
        elif criteria_type == 'absResidual':
            absResidual_dif = np.sum(residual_absZ_arr) - np.sum(np.abs(delete_onePoint_residual_arr))
            print('Index:', index, 'absResidual_dif:', absResidual_dif)
            dif_list.append(absResidual_dif)

    sorted_dif_list = sorted(dif_list, reverse=True) # Descending order
    sorted_chiSquare_list = sorted(chiSquare_list, reverse=False) # Ascending order
    print('-------------∑|▲Re| Order (Big->Small) + ChiSquare Order (Small->Big)--------------------')
    print('dif, sorted_dif_list.index(dif), chiS, sorted_chiSquare_list.index(chiS)')
    for dif, chiS in zip(dif_list, chiSquare_list):
        print(dif, sorted_dif_list.index(dif), chiS, sorted_chiSquare_list.index(chiS))

    # find and return the index of the maximum value in dif_list
    max_dif = max(dif_list)
    max_dif_index = dif_list.index(max_dif)
    deletedPointIndex = maxAndIndex_list[max_dif_index][1]
    return deletedPointIndex, chiSquare_list[max_dif_index]

# detect(eis_source='2021_09_15_R(RC)(RW)_HF_S_ecm.pkl',
       # criteria_type='absRe',
       # criteria_type='absIm',
       # criteria_type='absRealAndAbsImag',
       # criteria_type='absResidual',
       #  --------------------------
       # qua_flag=False)
       # qua_flag=True, Q_flag='Q3')
       # qua_flag=True, Q_flag='Q2')