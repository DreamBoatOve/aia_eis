import os

from a_d.ny_AD import detect as nyImDetect
from a_d.absZ_AD_1 import detect as absZDetect
from a_d.phase_AD_1 import detect as phaseDetect
from data_processor.GOA_preprocessor.goa_data_wrapper import load_Lai_EIS_data
from IS.IS import IS_0
from goa.integration.goa_intergration import goa_fitter_1
from playground.laiZhaoGui.getLaiVogitAddC import getLaiVogitAddCResDict
from playground.laiZhaoGui.goa.GOAs_fit_EIS_0 import get_para_range
from playground.laiZhaoGui.goa.GOAs_fit_EIS_1 import load_eis_ECM_dict
from data_processor.GOA_preprocessor.goa_data_wrapper import load_lai_manual_fitting_res

from utils.visualize_utils.IS_plots.ny import nyquist_plot
from utils.visualize_utils.IS_plots.bd import bode_one_plot

"""
Module Function
    1- 先去除异常点
    2- 使用一个合适的GOA拟合ECM参数
"""

# Import EIS
# Load Lai's normed(* multiply experimental area 1.01 * 1e-6 cm^2) EIS data
"""
lai_normed_eis_dict_list[
    dict0{
        'file_name': '1-1',
        'ecm_num': 9,
        'f': [100078.1, 63140.62, ..., 0.1588983, 0.1001603],
        'z_raw': [(0.005566658429999999-0.0112022736j),
                  (0.006214947129999999-0.0172324988j),
                  ...,
                  (285.52881799999994-486.4391289999999j),
                  (370.64242699999994-661.259928j)]
    }
    dict1,
    ...
]
"""
lai_normed_eis_dict_list = load_Lai_EIS_data(file_path='../../../datasets/goa_datasets/normed',
                                             file_name='2020_08_22_goa_lai_normed_dataset_pickle.file')

# --------------------------------- 去除Raw EIS中的异常点 ---------------------------------
# --------------- 先尝试 去除 一条Raw EIS 中的异常点 ---------------
def tryOneEIS_AD(eisIndex=1):
    # read an Raw-EIS
    normedRawEIS = IS_0()
    print(lai_normed_eis_dict_list[eisIndex])
    print('----------------------------------------')
    normedRawEIS.readFromLaiPickle(laiNormedEisDict=lai_normed_eis_dict_list[eisIndex], limitList=None)

    # plot Raw-EIS Nyquist and Bode for visual inspection
    nyquist_plot(z_list=normedRawEIS.z_arr,
                 grid_flag=False, fig_title='Nyquist-NormedRawEIS-i={}'.format(eisIndex))
    bode_one_plot(fre_list=normedRawEIS.fre_arr, z_list=normedRawEIS.z_arr,
                  fig_title='Bode-NormedRawEIS-i={}'.format(eisIndex))

    # Remove Outlier
    deletedPointIndex_list = nyImDetect(eis_source=normedRawEIS,
                                    vogitAddC=True,
                                    pointNum=10,
                                    chiSquareLimit=2.5*1e-2,
                                    printFlag=True)
    print(deletedPointIndex_list)

    # ------------- Check the Nyquist and Bode plots of EIS after deleted possible outliers -------------
    for dpi in deletedPointIndex_list:
        normedRawEIS.removeZByIndex(index=dpi)
    # plot Raw-EIS and AD-EIS to compare (Nyquist and Bode)
    nyquist_plot(z_list=normedRawEIS.z_arr,
                 grid_flag=False, fig_title='Nyquist-NormedRawEIS-delete:{}'.format(deletedPointIndex_list))
    bode_one_plot(fre_list=normedRawEIS.fre_arr, z_list=normedRawEIS.z_arr,
                  fig_title='Bode-NormedRawEIS-delete:{}'.format(deletedPointIndex_list))
    # ------------- Check the Nyquist and Bode plots of EIS after deleted possible outliers -------------
    return normedRawEIS
# EIS after outlier removal
# eis_AD = tryOneEIS_AD(eisIndex=17)
# --------------- 先尝试 去除 一条Raw EIS 中的异常点 ---------------
# --------------- 先尝试 去除 一条Raw EIS 中的异常点 ---------------
# --------------------------------- 去除Raw EIS中的异常点 ---------------------------------

"""
Set parameters' search range according to Lai's manual fitting results
lai_manual_fit_res_dict{
    '1-14':{
        'para': [0.01839, 0.006388, 0.8688, 1.175, 0.002783, 0.798, 1371.0],
        'limit': [[0.0001, 1], [1e-05, 0.1], [0.3, 1.0], [0.01, 100], [1e-05, 0.1], [0.3, 1.0], [10, 100000]],
        'chi_square': 0.001314
    }
    '2-13':{},
    ...
}
"""
lai_manual_fit_res_dict = load_lai_manual_fitting_res(file_path='../../../datasets/goa_datasets/Lai_manual_fitting_res',
                                                      file_names=['2020_07_22_lai_ecm2_fitting_res.CSV',
                                                                  '2020_07_22_lai_ecm9_fitting_res.CSV'])

laiVogitAddCResDict = getLaiVogitAddCResDict(fp='../', fn='laiAddVogitCRes.txt')

def packEisDict(detectType):
    global lai_normed_eis_dict_list
    global lai_manual_fit_res_dict
    eisDictList = []

    # For code test
    # for lai_normed_eis_dict in lai_normed_eis_dict_list[:3]:
    # Formal
    for lai_normed_eis_dict in lai_normed_eis_dict_list:
        eisDict = {}
        eisDict['exp_fn'] = lai_normed_eis_dict['file_name']
        print(eisDict['exp_fn'])
        eisDict['ecm_num'] = lai_normed_eis_dict['ecm_num']
        eisDict['limit'] = lai_manual_fit_res_dict[lai_normed_eis_dict['file_name']]['limit']

        # ------------ delete Outlier ------------
        normedRawEIS = IS_0()
        normedRawEIS.readFromLaiPickle(laiNormedEisDict=lai_normed_eis_dict, limitList=None)
        if detectType == 'nyIm':
            deletedPointIndex_list = nyImDetect(eis_source=normedRawEIS,
                                                # vogitAddC=True,
                                                vogitAddC=laiVogitAddCResDict[eisDict['exp_fn']],
                                                pointNum=10,
                                                chiSquareLimit=2.5 * 1e-2,
                                                printFlag=False)
        elif detectType == 'absZ':
            deletedPointIndex_list = absZDetect(eis_source=normedRawEIS,
                                                # vogitAddC=True,
                                                vogitAddC=laiVogitAddCResDict[eisDict['exp_fn']],
                                                pointNum=10,
                                                chiSquareLimit=2.5 * 1e-2,
                                                printFlag=False)
        elif detectType == 'phase':
            deletedPointIndex_list = phaseDetect(eis_source=normedRawEIS,
                                                 # vogitAddC=True,
                                                vogitAddC=laiVogitAddCResDict[eisDict['exp_fn']],
                                                pointNum=10,
                                                chiSquareLimit=2.5 * 1e-2,
                                                printFlag=False)

        for dpi in deletedPointIndex_list:
            normedRawEIS.removeZByIndex(index=dpi)
        # ------------ delete Outlier ------------

        # ------------  替换删除异常点后的fre 和 z_raw ------------
        eisDict['f'] = normedRawEIS.fre_arr.tolist()
        eisDict['z_raw'] = normedRawEIS.z_arr.tolist()
        # ------------  替换删除异常点后的fre 和 z_raw ------------

        eisDictList.append(eisDict)
    return eisDictList
# eisDictList = packEisDict(detectType='nyIm')
# eisDictList = packEisDict(detectType='absZ')
eisDictList = packEisDict(detectType='phase')

# 2- 使用一个合适的GOA拟合ECM参数
for eisDict in eisDictList:
    goa_fitter_1(ecm_para_config_dict=eisDict, repeat_time=1)