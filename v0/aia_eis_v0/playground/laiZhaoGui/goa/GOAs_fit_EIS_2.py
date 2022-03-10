import sys
sys.path.append('../../../')
import os
from multiprocessing import Pool

from data_processor.GOA_preprocessor.goa_data_wrapper import load_Lai_EIS_data
from goa.integration.goa_intergration import goa_fitter_multiProcess_Pool_0
from playground.laiZhaoGui.goa.GOAs_fit_EIS_0 import get_para_range
from playground.laiZhaoGui.goa.GOAs_fit_EIS_1 import load_eis_ECM_dict
from data_processor.GOA_preprocessor.goa_data_wrapper import load_lai_manual_fitting_res

"""
Module function
    The best five GOAs for each ECM have been picked out
    Procedure:
        1- Read raw EIS experiment
        2- Assign matched the best five GOAs to EIS according to its ECM
        3- Using python multiprocessing module to run tasks parallel, self-defined process number according 
            to hardware's condition
Version 2:
    根据Lai手动的拟合结果来为每个待拟合参数适应性分配范围[para * 1e-2, para * 1e2]，
    但是对待cpe_n时要注意特殊处理 【cpe_manual_fit_res - 0.1, 1.0】
    
Version 1:
    我统计Lai每个参数的拟合结果，得到每个参数拟合的最大和最小值作为GOA拟合的上下限范围，这个范围很大，拟合结果估计很不理想
"""
# 1- Load Lai's normed(* multiply experimental area 1.01 * 1e-6 cm^2) EIS data
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

# 2- Load each EIS's ECM type
"""
eis_ecm_dict{
    '1-1': 9,
    '1-2': 9,
    ...,
    '3-24': 2,
    ...
}
"""
eis_ecm_dict = load_eis_ECM_dict(file_path='../../../datasets/goa_datasets/Lai_manual_fitting_res',
                                 file_name='2020_07_22_lai_ecm_Num_record.txt')

# 3- Set parameters' search range according to Lai's manual fitting results
lai_manual_fit_res_dict = load_lai_manual_fitting_res(file_path='../../../datasets/goa_datasets/Lai_manual_fitting_res',
                                                      file_names=['2020_07_22_lai_ecm2_fitting_res.CSV',
                                                                  '2020_07_22_lai_ecm9_fitting_res.CSV'])

# 4- Gather all needed info for GOA fitting
def pack_ecm_para_config():
    """
    function
        Gather all the needed information for GOA's fitting, specifically:
            ecm_para_config_dict['exp_fn']: raw experimental file name, like: '1-23', '4-34'
            ecm_para_config_dict['ecm_num'] = ecm_num, int
            ecm_para_config_dict['limit']: [[para_0_low_limit, para_0_upper_limit], [para_1_low_limit, para_1_upper_limit], ...]
            ecm_para_config_dict['f']: list[float, frequency ]
            ecm_para_config_dict['z_raw']: list[complex, Impedance], but using NORMED EIS data here
    :param

    :return:
        ecm_para_config_dict_list[
            ecm_para_config_dict_0{
                'exp_fn':'1-1',
                'ecm_num': 9,
                'limit': [[1e-05, 0.1], [1e-06, 0.01], [0.3, 1.0], [1, 10000], [1e-05, 0.1], [0.3, 1.0], [10, 100000]],
                'f': [100078.1, 63140.62, ..., 0.1588983, 0.1001603],
                'z_raw': [(0.005566658429999999-0.0112022736j),
                          (0.006214947129999999-0.0172324988j),
                          ...,
                          (285.52881799999994-486.4391289999999j),
                          (370.64242699999994-661.259928j)]
            },
            ecm_para_config_dict_1, ...
        ]
    """
    global lai_normed_eis_dict_list
    global eis_ecm_dict
    global lai_manual_fit_res_dict

    ecm_para_config_dict_list = []
    for normed_eis_dict in lai_normed_eis_dict_list:
        ecm_para_config_dict = {}
        ecm_para_config_dict['exp_fn'] = normed_eis_dict['file_name']
        ecm_para_config_dict['ecm_num'] = normed_eis_dict['ecm_num']
        ecm_para_config_dict['limit'] = lai_manual_fit_res_dict[ecm_para_config_dict['exp_fn']]['limit']
        ecm_para_config_dict['f'] = normed_eis_dict['f']
        ecm_para_config_dict['z_raw'] = normed_eis_dict['z_raw']
        ecm_para_config_dict_list.append(ecm_para_config_dict)
    return ecm_para_config_dict_list
ecm_para_config_dict_list = pack_ecm_para_config()

def get_unfitted_ecm_para_config_dict_list_2nd():
    """
    Function
        在 2021-02m-07d 第2次使用GOA-Top5拟合Lai的EIS时，部分程序 因为ICA出错，程序中断，在此重跑遗漏的程序未完成的程序
            2-3，2-4，2-5
    :param
    :return:
    """
    global ecm_para_config_dict_list
    unfitted_eis_fn = ['2-3', '2-4', '2-5']
    tmp_ecm_para_config_dict_list = []
    for ecm_para_config_dict in ecm_para_config_dict_list:
        if ecm_para_config_dict['exp_fn'] in unfitted_eis_fn:
            tmp_ecm_para_config_dict_list.append(ecm_para_config_dict)
    return tmp_ecm_para_config_dict_list
# task_list = get_unfitted_ecm_para_config_dict_list_2nd()

def get_unfitted_ecm_para_config_dict_list_1st(fp):
    """
    Function
        在2020-08第一次使用GOA-Top5拟合Lai的EIS时，部分 因为ICA出错，程序中断，在此重跑遗漏的程序
    :param fp:
    :return:
    """
    global ecm_para_config_dict_list
    fn_list = os.listdir(fp)
    existed_fn_list = []
    # Check and store existed file names
    for fn in fn_list:
        if fn.endswith('.txt'):
            existed_fn_list.append(fn.split('.')[0].split('_')[0])

    """
    Code stops at 2020-08-26 because of ICA went wrong. A few training result files are not complete and needs to be rerun, 
    and their file names are: 
        File name   Index
        1-1         0
        1-8         6
        1-18        13
        1-26        20
        2-2         28
        2-10        36
        2-33        53
        3-5         56
        3-19        68
        4-2         84
        4-9         91
        4-19        96
        5-1         105
    """
    while len(existed_fn_list) > 0:
        e_fn = existed_fn_list[0]
        for i, ecm_para_config_dict in enumerate(ecm_para_config_dict_list):
            if e_fn == ecm_para_config_dict['exp_fn']:
                del ecm_para_config_dict_list[i]
                existed_fn_list.pop(0)
                break
    return ecm_para_config_dict_list
# task_list = get_unfitted_ecm_para_config_dict_list_1st(fp='./R(RC)_IS_lin-kk_res.txt/magNum=2_res')

if __name__ == '__main__':
    # ----------------------- Second time (2021-02m-07d): Try Top-5 GOAs on Lai's EIS -----------------------
    task_list = ecm_para_config_dict_list
    # task_list = get_unfitted_ecm_para_config_dict_list_2nd()
    # ----------------------- Second time (2021-02m-07d): Try Top-5 GOAs on Lai's EIS -----------------------

    # ----------------------- First time (2020-08m): Try Top-5 GOAs on Lai's EIS -----------------------
    # task_list = ecm_para_config_dict_list
    # task_list = get_unfitted_ecm_para_config_dict_list(fp='./R(RC)_IS_lin-kk_res.txt/magNum=2_res')
    # ----------------------- First time (2020-08m): Try Top-5 GOAs on Lai's EIS -----------------------

    process_num = 3
    repeat_time = 10
    input_list = [[ecm_para_config_dict, repeat_time] for ecm_para_config_dict in task_list]
    pool = Pool(processes=process_num)
    pool.map(goa_fitter_multiProcess_Pool_0, input_list)
    pool.close()
    pool.join()
"""
Second time (2021-02m-07d): Try Top-5 GOAs on Lai's EIS:
First time (2020-08m): Try Top-5 GOAs on Lai's EIS:
    Code stops at 2020-08-26 because of ICA went wrong. A few training result files are not complete and needs to be rerun, 
    and their file names are: 
        File name   Index
        1-1         0
        1-8         6
        1-18        13
        1-26        20
        2-2         28
        2-10        36
        2-33        53
        3-5         56
        3-19        68
        4-2         84
        4-9         91
        4-19        96
        5-1         105
"""