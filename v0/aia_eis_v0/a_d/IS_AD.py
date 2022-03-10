import copy
import numpy as np

from circuits.vogit_1 import Vogit_3
from IS.IS import IS_0
from smoothAlg.SavitzkyGolay import savitzkyGolay as SG

from ny_onePoint_AD import load_EIS, getMaxAndIndex

from utils.file_utils.pickle_utils import load_pickle_file
from utils.statistic.quartile import cal_quartile
from utils.visualize_utils.IS_plots.bd import bode_Phase
from utils.visualize_utils.plot_utils import plot_2D_util

def calAbsZ(z):
    return np.sqrt(z.real ** 2 + z.imag ** 2)

def calPhase(z):
    """
    :param
        z: 4+4j
    :return:
        phase: 45°
    """
    phase = np.arctan2(z.imag, z.real) * 180 / np.pi
    return phase

def detect(eis_source, pointNum=3, chiSquareLimit=1e-5):
    """
    Function
        根据之前的实验结果得到一下结论
            Ny + 【Q2 + |▲Im|】 有最佳的检测结果
            Bd-|Z| + 【Q2 + |ε|】 有最佳的检测结果
            Bd-Phase + 【Q2 + |ε|】 有最佳的检测结果
        所以流程为
            EIS平滑 --》 EIS_SM
            --》 Ny  --》Ny结果
            --》 Bd-|Z|  --》Bd-|Z|结果
            --》 Bd-Phase  --》Bd-Phase结果
            三种结果进行加权排序
    :param
        eis_source
            ecm_fn:
            eis obj
        pointNum: int
            the number of outliers you want to detect
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

        # --------------- calculate ε_SG ---------------
        residual_SG_arr = (z_arr - z_SG_arr) / np.abs(z_arr)
        # --------- calculate |ε_SG_Im| ---------
        # |▲Im|
        residual_SG_absImag_arr = np.abs(residual_SG_arr.imag)
        # --------- calculate |ε_SG_Im| ---------
        # --------------- calculate ε_SG ---------------

        # --------------- calculate |▲|Z|| = | (|Z| - |Zsg|) / |Z| | ---------------
        absZ_arr = np.abs(z_arr)
        absZ_SG_arr = np.abs(z_SG_arr)
        # |▲|Z|| = | (|Z| - |Zsg|) / |Z| |
        abs_d_absZ_arr = np.abs((absZ_arr - absZ_SG_arr) / absZ_arr)
        # --------------- calculate |▲|Z|| = | (|Z| - |Zsg|) / |Z| | ---------------

        # --------------- calculate |▲Phase| ---------------
        phase_arr = calPhase(z_arr)
        phase_SG_arr = calPhase(z_SG_arr)
        # |▲φ| = |(φ - φ_SG) / φ|
        d_absPhase_arr = np.abs((phase_arr - phase_SG_arr) / phase_arr)
        # --------------- calculate |▲Phase| ---------------

        # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------
        vogit = Vogit_3(impSpe=tmp_eis, fit_type='complex', u_optimum=0.85, add_C=True, M_max=None)
        vogit.lin_KK()
        vogit.cal_residual()
        residual_arr = vogit.residual_arr
        vogit_chi_square = vogit.cal_chiSquare(weight_type='modulus')
        print('Chi Square:', vogit_chi_square)
        # ------------------- 打印原始数据 lin-KK 的误差结果，和后面删除单点后的结果对比 -------------------

        ny_maxAndIndex_list = getMaxAndIndex(arr=residual_SG_absImag_arr)
        absZ_maxAndIndex_list = getMaxAndIndex(arr=abs_d_absZ_arr)
        phase_maxAndIndex_list = getMaxAndIndex(arr=d_absPhase_arr)
        # -------------------------------------- 使用 四分位-Q2 过滤一批点 --------------------------------------
        # ny
        qua_info_list = cal_quartile(arr=residual_SG_absImag_arr)
        low_boundary, Q1, Q2, Q3, up_boundary = qua_info_list
        print('--------------\nNy-|▲Im|: low_boundary, Q1, Q2, Q3, up_boundary')
        print(low_boundary, Q1, Q2, Q3, up_boundary)
        ny_maxAndIndex_list = [maxAndIndex for maxAndIndex in ny_maxAndIndex_list if maxAndIndex[0] > Q2]

        # Bd-|Z|
        qua_info_list = cal_quartile(arr=abs_d_absZ_arr)
        low_boundary, Q1, Q2, Q3, up_boundary = qua_info_list
        print('--------------\nBd-|Z|: low_boundary, Q1, Q2, Q3, up_boundary')
        print(low_boundary, Q1, Q2, Q3, up_boundary)
        absZ_maxAndIndex_list = [maxAndIndex for maxAndIndex in absZ_maxAndIndex_list if maxAndIndex[0] > Q2]

        # Bd-Phase
        qua_info_list = cal_quartile(arr=d_absPhase_arr)
        low_boundary, Q1, Q2, Q3, up_boundary = qua_info_list
        print('--------------\nBd-Phase: low_boundary, Q1, Q2, Q3, up_boundary')
        print(low_boundary, Q1, Q2, Q3, up_boundary)
        phase_maxAndIndex_list = [maxAndIndex for maxAndIndex in phase_maxAndIndex_list if maxAndIndex[0] > Q2]
        # -------------------------------------- 使用 四分位-Q2 过滤一批点 --------------------------------------

        # 遍历 所有可疑的点
        ny_dif_list = []
        ny_chiSquare_list = []
        print('-------------Index and Difference and chiSquare For Ny--------------------')
        for maxAndIndex in ny_maxAndIndex_list:
            maximum, index = maxAndIndex

            # -------------- delete one point --------------
            eis_delete_onePoint = copy.deepcopy(tmp_eis)
            eis_delete_onePoint.removeZByIndex(index)
            # -------------- delete one point --------------

            vogit_delete_onePoint = Vogit_3(impSpe=eis_delete_onePoint, fit_type='complex',
                                            u_optimum=0.85, add_C=True, M_max=None)
            vogit_delete_onePoint.lin_KK()
            vogit_delete_onePoint.simulate_Z()
            vogit_delete_onePoint.cal_residual()
            vogit_delete_onePointResidual_arr = vogit_delete_onePoint.residual_arr
            tmp_chi_square = vogit_delete_onePoint.cal_chiSquare()
            ny_chiSquare_list.append(tmp_chi_square)

            # ∑|▲Im|
            absIm_dif = np.sum(np.abs(residual_arr.imag)) - np.sum(np.abs(vogit_delete_onePointResidual_arr.imag))
            print('Index:', index, 'absIm_dif:', absIm_dif, 'chiSquares:', tmp_chi_square)
            ny_dif_list.append(absIm_dif)

        absZ_dif_list = []
        absZ_chiSquare_list = []
        print('-------------Index and Difference and chiSquare For absZ--------------------')
        for maxAndIndex in absZ_maxAndIndex_list:
            maximum, index = maxAndIndex

            # -------------- delete one point --------------
            eis_delete_onePoint = copy.deepcopy(tmp_eis)
            eis_delete_onePoint.removeZByIndex(index)
            # -------------- delete one point --------------

            vogit_delete_onePoint = Vogit_3(impSpe=eis_delete_onePoint, fit_type='complex',
                                            u_optimum=0.85, add_C=True, M_max=None)
            vogit_delete_onePoint.lin_KK()
            vogit_delete_onePoint.simulate_Z()
            vogit_delete_onePoint.cal_residual()
            vogit_delete_onePointResidual_arr = vogit_delete_onePoint.residual_arr
            tmp_chi_square = vogit_delete_onePoint.cal_chiSquare()
            absZ_chiSquare_list.append(tmp_chi_square)

            absResidual_dif = np.sum(np.abs(residual_arr)) - np.sum(np.abs(vogit_delete_onePointResidual_arr))
            print('Index:', index, 'absResidual_dif:', absResidual_dif, 'chiSquares:', tmp_chi_square)
            absZ_dif_list.append(absResidual_dif)

        phase_dif_list = []
        phase_chiSquare_list = []
        print('-------------Index and Difference and chiSquare For phase--------------------')
        for maxAndIndex in phase_maxAndIndex_list:
            maximum, index = maxAndIndex

            # -------------- delete one point --------------
            eis_delete_onePoint = copy.deepcopy(tmp_eis)
            eis_delete_onePoint.removeZByIndex(index)
            # -------------- delete one point --------------

            vogit_delete_onePoint = Vogit_3(impSpe=eis_delete_onePoint, fit_type='complex',
                                            u_optimum=0.85, add_C=True, M_max=None)
            vogit_delete_onePoint.lin_KK()
            vogit_delete_onePoint.simulate_Z()
            vogit_delete_onePoint.cal_residual()
            vogit_delete_onePointResidual_arr = vogit_delete_onePoint.residual_arr
            tmp_chi_square = vogit_delete_onePoint.cal_chiSquare()
            phase_chiSquare_list.append(tmp_chi_square)

            absResidual_dif = np.sum(np.abs(residual_arr)) - np.sum(np.abs(vogit_delete_onePointResidual_arr))
            print('Index:', index, 'absResidual_dif:', absResidual_dif, 'chiSquares:', tmp_chi_square)
            phase_dif_list.append(absResidual_dif)

        sorted_ny_dif_list = sorted(ny_dif_list, reverse=True)  # Descending order
        sorted_absZ_dif_list = sorted(absZ_dif_list, reverse=True)  # Descending order
        sorted_phase_dif_list = sorted(phase_dif_list, reverse=True)  # Descending order

        # ----------------------- Rank by the sum of three scores -----------------------
        """
        score_dict{
            5 (int, EIS point index): [chiSquares, nyScore, BdAbsZScore, BdPhaseScore]
        }
        """
        score_dict = {}
        for ny_dif in sorted_ny_dif_list:
            ny_dif_index = ny_dif_list.index(ny_dif)
            chiSquare = ny_chiSquare_list[ny_dif_index]
            ZIndex = ny_maxAndIndex_list[ny_dif_index][1]
            if ZIndex not in score_dict.keys():
                score_dict[ZIndex] = [chiSquare]
                # Ny score
                score_dict[ZIndex].append(len(ny_dif_list) - ny_dif_index)

        for absZ_dif in sorted_absZ_dif_list:
            absZ_index = absZ_dif_list.index(absZ_dif)
            chiSquare = absZ_chiSquare_list[absZ_index]
            ZIndex = absZ_maxAndIndex_list[absZ_index][1]
            if ZIndex not in score_dict.keys():
                score_dict[ZIndex] = [chiSquare, 0]
                # Bd-|Z| Score
                score_dict[ZIndex].append(len(absZ_dif_list) - absZ_index)
            else:
                score_dict[ZIndex].append(len(absZ_dif_list) - absZ_index)

        for phase_dif in sorted_phase_dif_list:
            phase_dif_index = phase_dif_list.index(phase_dif)
            chiSquare = phase_chiSquare_list[phase_dif_index]
            ZIndex = phase_maxAndIndex_list[phase_dif_index][1]
            if ZIndex not in score_dict.keys():
                score_dict[ZIndex] = [chiSquare, 0, 0]
                score_dict[ZIndex].append(len(phase_dif_list) - phase_dif_index)
            else:
                score_dict[ZIndex].append(len(phase_dif_list) - phase_dif_index)

        allChiSquareList = []
        scoreSumDict = {}
        for item in score_dict.items():
            ZIndex, v = item
            allChiSquareList.append(v[0])
            scoreSumDict[ZIndex] = sum(v[1:])
        sorted_ScoreSum_ZIndex_List = sorted(scoreSumDict.items(), key=lambda d:d[1], reverse=True) # Big --> Small
        sortedAllChiSquareList = sorted(allChiSquareList, reverse=False) # Small --> Big

        print('--------------- ZIndex, scoreSum, chiSquare, chiSquareOrder ---------------')
        for ZIndex, scoreSum in sorted_ScoreSum_ZIndex_List:
            chiSquare = score_dict[ZIndex][0]
            print(ZIndex, scoreSum, chiSquare, sortedAllChiSquareList.index(chiSquare))
        # ----------------------- Rank by the sum of three scores -----------------------

        # --- Wrong ---
        # find and return the index of the maximum value in dif_list
        # maxChiSquare = max(allChiSquareList)
        # maxChiSquareIndex = [item[0] for item in score_dict.items() if item[1][0] == maxChiSquare][0]
        # --- Wrong ---


        # --- Wrong ---
        # find and return the index of the maximum scoreSum in sorted_ScoreSum_ZIndex_List
        # deletedPointIndex, maxScoreSum = sorted_ScoreSum_ZIndex_List[0]
        # --- Wrong ---

        # find and return the index of the minimum value in allChiSquareList
        minChiSquare = min(allChiSquareList)
        minChiSquareIndex = [item[0] for item in score_dict.items() if item[1][0] == minChiSquare][0]
        deletedPointIndex = minChiSquareIndex

        print('--------------\nThe index of the {0} outlier is: {1}'.format(pointCount, deletedPointIndex))
        deletedPointIndex_list.append(deletedPointIndex)
        pointCount += 1

# --------------------------- One Outliers ---------------------------
# detect(eis_source='2021_09_15_R(RC)(RW)_LF_S_ecm.pkl',
#        pointNum=1, chiSquareLimit=1e-7)
# --------------------------- One Outliers ---------------------------

# --------------------------- Two Outliers ---------------------------
# detect(eis_source='2021_10_13_R(RC)(RW)_MLF_MM_ecm.pkl',
#        pointNum=2, chiSquareLimit=1e-7)
# --------------------------- Two Outliers ---------------------------

# --------------------------- Three Outliers ---------------------------
detect(eis_source='2021_10_13_R(RC)(RW)_MMM_ecm.pkl',
       pointNum=3, chiSquareLimit=1e-7)
# --------------------------- Three Outliers ---------------------------