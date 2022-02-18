import os
import pickle

from utils.file_utils.filename_utils import get_date_prefix

"""
There are 10 ECMs now, creat configuration file for each ECM simulation
    1-Pickle parameters' configuration of each ECMs
    2-Load parameters' configuration of one ECM by ECM's serial
    3-The simulation process was carried in DPFC_test_3.ipynb, and results are stored at the ./ecm_sim_res
"""

# 1-Pickle parameters' configuration of each ECMs
def pickle_sim_ecm_para_config_dict(ecm_num, limit_list, para_list, fre_list, z_list):
    ecm_para_config_dict = {}
    ecm_para_config_dict['ecm_num'] = ecm_num
    ecm_para_config_dict['limit'] = limit_list
    ecm_para_config_dict['para'] = para_list
    ecm_para_config_dict['f'] = fre_list
    ecm_para_config_dict['z_sim'] = z_list

    # import datetime
    # year, month, day int
    # year = datetime.datetime.now().year
    # month = datetime.datetime.now().month
    # day = datetime.datetime.now().day
    # file_name = str(year)+'_'+str(month)+'_'+str(day)+'_sim_ecm_'+str(ecm_num)+'_pickle.file'

    file_name = get_date_prefix() + 'sim_ecm_'+str(ecm_num)+'_pickle.file'
    with open(file_name, 'wb') as file:
        pickle.dump(ecm_para_config_dict, file)

# 2-Load parameters' configuration of one ECM by ECM's serial
def load_sim_ecm_para_config_dict(ecm_num, file_path):
    if ecm_num == 0:
        file_name = ''
    elif ecm_num == 1:
        # file_name = '2020_05_19_sim_ecm_1_pickle.file'
        file_name = 'simulated/ecm_001/2020_07_04_sim_ecm_001_pickle.file'
    elif ecm_num == 2:
        # file_name = '2020_3_26_sim_ecm_2_pickle_1.file'
        file_name = 'simulated/ecm_002/2020_05_20_sim_ecm_2_pickle_2.file'
    elif ecm_num == 3:
        file_name = 'simulated/ecm_003/2020_05_20_sim_ecm_3_pickle_0.file'
    elif ecm_num == 4:
        file_name = 'simulated/ecm_004/2020_07_03_sim_ecm_004_pickle_1.file'
    elif ecm_num == 5:
        file_name = 'simulated/ecm_005/2020_07_03_sim_ecm_005_pickle_0.file'
    elif ecm_num == 6:
        file_name = 'simulated/ecm_006/2020_07_03_sim_ecm_006_pickle_1.file'
    elif ecm_num == 7:
        file_name = 'simulated/ecm_007/2020_07_03_sim_ecm_007_pickle.file'
    elif ecm_num == 8:
        file_name = 'simulated/ecm_008/2020_07_03_sim_ecm_008_pickle_0.file'
    elif ecm_num == 9:
        file_name = 'simulated/ecm_009/2020_07_03_sim_ecm_009_pickle_0.file'
    else:
        print('Currently, There is only ECM - 0 ~ 9')

    file_abs_path = os.path.join(file_path, file_name)
    with open(file_abs_path, 'rb') as file:
        ecm_para_config_dict = pickle.load(file)
    return ecm_para_config_dict
# load_sim_ecm_para_config_dict(ecm_num=9, file_path='../../datasets/goa_datasets/simulated/ecm_009')