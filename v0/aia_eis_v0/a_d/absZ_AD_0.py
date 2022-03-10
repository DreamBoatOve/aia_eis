import copy
import os
import numpy as np

from circuits.ecm import ECM
from circuits.vogit_1 import Vogit_3
from IS.IS import IS_0
from smoothAlg.SavitzkyGolay import savitzkyGolay as SG

from ny_onePoint_AD import load_EIS, getMaxAndIndex

from utils.file_utils.pickle_utils import load_pickle_file
from utils.statistic.quartile import cal_quartile
from utils.visualize_utils.IS_plots.bd import bode_absZ
from utils.visualize_utils.plot_utils import plot_2D_util

def calAbsZ(z):
    return np.sqrt(z.real ** 2 + z.imag ** 2)

def detect(eis_source, pointNum=3, criteria_type='absRealAndAbsImag',
           qua_flag=False, Q_flag='Q3', chiSquareLimit=1e-5):
    """
    :param
        eis_source
            ecm_fn:
            eis obj
        pointNum: int
            the number of outliers you want to detect
        criteria_type: str
            absRe == |▲Re|,
            absIm == |▲Im|,
            absRealAndAbsImag == |▲Re| + |▲Im|,
            absResidual == sqrt(▲Re**2 + ▲Im**2)
        qua_flag bool
            True    使用 四分位 过滤一批点
            False   不使用 四分位 过滤一批点
        Q_flag: str
            up_boundary
            Q1
            Q2
            Q3
            low_boundary
        chiSquareLimit: float
            default 1e-5
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

        # 按照从大到小的索引，以此删除可能的异常点
        for dpi in sorted(deletedPointIndex_list, reverse=True): # dpi == deletedPointIndex
            tmp_eis.removeZByIndex(index=dpi)

        z_arr = tmp_eis.z_arr
        z_SG_arr = SG(z_arr, loop_time=20, convPoints=5)

        # ------ Plot the smoothed results ------
        # bode_absZ(fre_arr=tmp_eis.fre_arr, z_arr_list=[z_arr, z_SG_arr], plot_type='normal',
        #           fig_title='Comparison', label_list=['Raw', 'SG'])
        # ------ Plot the smoothed results ------

        absZ_arr = np.array([np.sqrt(z.real ** 2 + z.imag ** 2) for z in z_arr])
        absZ_SG_arr = np.array([np.sqrt(z.real ** 2 + z.imag ** 2) for z in z_SG_arr])

        # |▲|Z|| = | (|Z| - |Zsg|) / |Z| |
        abs_d_absZ_arr = np.array([np.abs((absZ - absZ_SG) / absZ) for absZ, absZ_SG in zip(absZ_arr, absZ_SG_arr)])

        # -------- Plot the |▲|Z|| vs. Fre --------
        # p2u = plot_2D_util(x_list=np.log10(tmp_eis.fre_arr).tolist(), y_list=abs_d_absZ_arr.tolist())
        # p2u.single_dot_plot(x_label='log10(Fre)', y_label='|▲|Z|| = | (|Z| - |Zsg|) / |Z| |', line_label_str='|▲|Z|| vs. log10(Fre)')
        # -------- Plot the |▲|Z|| vs. Fre --------

        # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------
        vogit = Vogit_3(impSpe=tmp_eis, fit_type='complex', u_optimum=0.85, add_C=True, M_max=None)
        vogit.lin_KK()
        vogit.cal_residual()
        residual_arr = vogit.residual_arr
        vogit_chi_square = vogit.cal_chiSquare(weight_type='modulus')
        print('Chi Square:', vogit_chi_square)
        # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------

        print('-----------------\n', '|▲|Z||')
        maxAndIndex_list = getMaxAndIndex(arr=abs_d_absZ_arr)

        # -------------------------------------- 使用 四分位 过滤一批点 --------------------------------------
        if qua_flag:
            qua_info_list = cal_quartile(arr=abs_d_absZ_arr)
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
        # -------------------------------------- 使用 四分位 过滤一批点 --------------------------------------

        # 遍历 所有可疑的点
        dif_list = []
        chiSquare_list = []
        print('-------------Index anf Difference--------------------')
        for maxAndIndex in maxAndIndex_list:
            maximum, index = maxAndIndex

            # -------------- delete one point --------------
            eis_delete_onePoint = copy.deepcopy(tmp_eis)
            eis_delete_onePoint.raw_z_arr = np.delete(eis_delete_onePoint.raw_z_arr, index)
            eis_delete_onePoint.z_arr = np.delete(eis_delete_onePoint.z_arr, index)
            eis_delete_onePoint.fre_arr = np.delete(eis_delete_onePoint.fre_arr, index)
            eis_delete_onePoint.w_arr = np.delete(eis_delete_onePoint.w_arr, index)
            # -------------- delete one point --------------

            vogit_delete_onePoint = Vogit_3(impSpe=eis_delete_onePoint, fit_type='complex',
                                            u_optimum=0.85, add_C=True, M_max=None)
            vogit_delete_onePoint.lin_KK()
            vogit_delete_onePoint.simulate_Z()
            vogit_delete_onePoint.cal_residual()
            vogit_delete_onePointResidual_arr = vogit_delete_onePoint.residual_arr
            # vogit_Zarr = vogit_delete_onePoint.z_sim_arr
            # vogit_absZarr = calAbsZ(vogit_Zarr)
            tmp_chi_square = vogit_delete_onePoint.cal_chiSquare()
            chiSquare_list.append(tmp_chi_square)

            """
            # 计算 残差指标 |▲|Z|| = | (|Z| - |Z_Vogit|) / |Z| | --> sum(|▲|Z||)
            abs_dif_absZ_sum = np.sum((vogit_absZarr - absZ_arr) / vogit_Zarr)
            --> ValueError: operands could not be broadcast together with shapes (59,) (60,) 
            len(vogit_absZarr) = 59, len(absZ_arr) = 60
            # tmp_absZ_arr = copy.deepcopy(absZ_arr)
            # tmp_absZ_arr = np.delete(tmp_absZ_arr, index)
            # abs_dif_absZ_sum = np.sum(np.abs((tmp_absZ_arr - vogit_absZarr) / tmp_absZ_arr))
            # abs_dif_absZ_sumDif_list.append(abs_dif_absZ_sum)
            """
            # ∑|▲Re|
            if criteria_type == 'absRe':
                absRe_dif = np.sum(np.abs(residual_arr.real)) - np.sum(np.abs(vogit_delete_onePointResidual_arr.real))
                print('Index:', index, 'absRe_dif:', absRe_dif)
                dif_list.append(absRe_dif)
            # ∑|▲Im|
            elif criteria_type == 'absIm':
                absIm_dif = np.sum(np.abs(residual_arr.imag)) - np.sum(np.abs(vogit_delete_onePointResidual_arr.imag))
                print('Index:', index, 'absIm_dif:', absIm_dif)
                dif_list.append(absIm_dif)
            # ∑(|▲Re| + |▲Im|)
            elif criteria_type == 'absRealAndAbsImag':
                absRealAndAbsImag_dif = np.sum(np.abs(residual_arr.real) + np.abs(residual_arr.imag)) - \
                                        np.sum(np.abs(vogit_delete_onePointResidual_arr.real) +
                                               np.abs(vogit_delete_onePointResidual_arr.imag))
                print('Index:', index, 'absRealAndAbsImag_dif:', absRealAndAbsImag_dif)
                dif_list.append(absRealAndAbsImag_dif)
            # ∑|Residual|
            elif criteria_type == 'absResidual':
                absResidual_dif = np.sum(calAbsZ(residual_arr)) - np.sum(calAbsZ(vogit_delete_onePointResidual_arr))
                print('Index:', index, 'absResidual_dif:', absResidual_dif)
                dif_list.append(absResidual_dif)

        sorted_dif_list = sorted(dif_list, reverse=True)  # Descending order
        sorted_chiSquare_list = sorted(chiSquare_list, reverse=False)  # Ascending order
        print('-------------{0} Order (Big->Small) + ChiSquare Order (Small->Big)--------------------'.format(criteria_type))
        print('dif, sorted_dif_list.index(dif), chiS, sorted_chiSquare_list.index(chiS)')
        for dif, chiS in zip(dif_list, chiSquare_list):
            print(dif, sorted_dif_list.index(dif), chiS, sorted_chiSquare_list.index(chiS))

        # find and return the index of the maximum value in dif_list
        max_dif = max(dif_list)
        max_dif_index = dif_list.index(max_dif)
        deletedPointIndex = maxAndIndex_list[max_dif_index][1]

        print('--------------\nThe index of the {0} outlier is: {1}'.format(pointCount, deletedPointIndex))
        deletedPointIndex_list.append(deletedPointIndex)
        pointCount += 1

# -------------------------------------- One Outlier --------------------------------------
# detect(eis_source = '2021_09_15_R(RC)(RW)_LF_S_ecm.pkl', pointNum = 1,
       # ----------------
       # criteria_type = 'absRe',
       # criteria_type = 'absIm',
       # criteria_type = 'absRealAndAbsImag',
       # criteria_type = 'absResidual',
       # ----------------
       # qua_flag = False,
       # qua_flag=True,
       # ----------------
       # Q_flag='up_boundary', chiSquareLimit=1e-5)
       # Q_flag='Q3', chiSquareLimit=1e-5)
       # Q_flag='Q2', chiSquareLimit=1e-5)
       # Q_flag='Q1', chiSquareLimit=1e-5)
# -------------------------------------- One Outlier --------------------------------------

# -------------------------------------- Two Outlier --------------------------------------
# detect(eis_source = '2021_10_13_R(RC)(RW)_HMF_MH_ecm.pkl', pointNum = 2,
       # ----------------
       # criteria_type = 'absRe',
       # criteria_type = 'absIm',
       # criteria_type = 'absRealAndAbsImag',
       # criteria_type = 'absResidual',
       # ----------------
       # qua_flag = False,
       # qua_flag=True,
       # ----------------
       # Q_flag='up_boundary', chiSquareLimit=1e-5)
       # Q_flag='Q3', chiSquareLimit=1e-5)
       # Q_flag='Q2', chiSquareLimit=1e-5)
       # Q_flag='Q1', chiSquareLimit=1e-5)
# -------------------------------------- Two Outlier --------------------------------------

# -------------------------------------- Three Outlier --------------------------------------
# detect(eis_source = '2021_10_13_R(RC)(RW)_MMM_ecm.pkl', pointNum = 3,
       # ----------------
       # criteria_type = 'absRe',
       # criteria_type = 'absIm',
       # criteria_type = 'absRealAndAbsImag',
       # criteria_type = 'absResidual',
       # ----------------
       # qua_flag = False,
       # qua_flag=True,
       # ----------------
       # Q_flag='up_boundary', chiSquareLimit=1e-5)
       # Q_flag='Q3', chiSquareLimit=1e-5)
       # Q_flag='Q2', chiSquareLimit=1e-5)
       # Q_flag='Q1', chiSquareLimit=1e-5)
# -------------------------------------- Three Outlier --------------------------------------