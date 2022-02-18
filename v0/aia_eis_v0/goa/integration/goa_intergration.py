import sys

# Evolution
from goa.evolution_based.differential_evolution.de_0 import DE_EIS
from goa.evolution_based.evolution_programming.ep_0 import EP_EIS

# Human
from goa.human_based.group_search_optimizer.gso_1 import GSO_EIS
from goa.human_based.harmony_search.hs_0 import HS_EIS
from goa.human_based.imperialist_competitive_alg.ica_0 import ICA_EIS
from goa.human_based.teaching_learning_based_optimization.tlbo_0 import TLBO_EIS

# Physic
from goa.physic_based.black_hole.bh_0 import BH_EIS
from goa.physic_based.charged_system_search.css_0 import CSS_EIS
from goa.physic_based.multi_verse_optimizer.multi_verse_opt_1 import MVO_EIS

# Swarm
from goa.swarm_based.aritificial_bee_colony.abc_0 import ABC_EIS
from goa.swarm_based.grey_wolf_optimizer.gwo_0 import GWO_EIS
from goa.swarm_based.whale_optimization_algorithm.woa_0 import WOA_EIS

from utils.file_utils.filename_utils import get_date_prefix

"""
Module Function:
    After thoroughly assessments of 20 GOAs on 9 ECMs, I got their weight scores.
    When a new EIS with a certain ECM, we assign the best five GOAs of fitting that ECM to it
"""

"""
Wrong:
Obtained from dpfc_src\goa\simECM_res\\2nd\\2021_02_07_goa_weighted_score.txt
    1:['DE', 'GSO', 'TLBO', 'MVO', 'WOA'],
    2:['GSO', 'GWO', 'WOA', 'DE', 'ABC'],
    3:['GWO', 'WOA', 'HS', 'ABC', 'TLBO'],
    4:['GSO', 'WOA', 'GWO', 'CSS', 'HS'],
    5:['WOA', 'ABC', 'EP', 'DE', 'GWO'],
    6:['WOA', 'GWO', 'ABC', 'DE', 'HS'],
    7:['WOA', 'DE', 'ABC', 'GWO', 'HS'],
    8:['GWO', 'WOA', 'HS', 'ABC', 'CSS'],
    9:['DE', 'WOA', 'ABC', 'ICA', 'BH']
"""
goa_ecm_match_dict = {
    # ecm_num : [1st_GOA, 2nd_GOA, 3rd_GOA, 4th_GOA, 5th_GOA]
    # Obtained from dpfc_src\goa\simECM_res\\3rd\2021_03_07_goa_weighted_score.txt
    1: ['DE', 'GSO', 'MVO', 'TLBO', 'WOA'],
    2: ['GSO', 'GWO', 'WOA', 'DE', 'ABC'],
    3: ['GWO', 'WOA', 'HS', 'ABC', 'TLBO'],
    4: ['GSO', 'GWO', 'WOA', 'CSS', 'HS'],
    5: ['ABC', 'WOA', 'EP', 'DE', 'GWO'],
    6: ['WOA', 'GWO', 'ABC', 'DE', 'HS'],
    7: ['WOA', 'DE', 'GWO', 'ABC', 'GSO'],
    8: ['WOA', 'GWO', 'HS', 'ABC', 'CSS'],
    9: ['WOA', 'DE', 'ABC', 'GSO', 'ICA']
}

def para_limit_configer(ecm_num):
    R_limit = [1e-15, 1e15]
    Q_q_limit = [1e-15, 1]
    Q_n_limit = [0.5, 1]
    L_limit = [1e-5, 1e5]
    WB_sigma_limit = [1e-10, 1e4]
    if ecm_num == 1:
        limit_list = [[1e-10,1e10],[1e-10,1],[0.5,1],[1e3,1e30]]
    elif ecm_num == 2:
        limit_list = [[1e-18, 1e-10], [1e-10, 1.0], [0.3, 1.0], [1e-10, 1e10], [1e-5, 1.0], [0.5, 1.0], [1e-3, 1e10]]
    # ECM-3 R(QR(LR)) --> R0(Q0R1(L0R2))
    elif ecm_num == 3:
        limit_list = [R_limit, Q_q_limit, Q_n_limit, R_limit, L_limit, R_limit]
    # ECM-4 R(Q(RW)) --> R0(Q0(R1W0))
    elif ecm_num == 4:
        limit_list = [R_limit, Q_q_limit, Q_n_limit, R_limit, WB_sigma_limit]
    # ECM-5 R(QR)(QR)W --> R0(Q0R1)(Q1R2)W0
    elif ecm_num == 5:
        limit_list = [R_limit, Q_q_limit, Q_n_limit, R_limit, Q_q_limit, Q_n_limit, R_limit, WB_sigma_limit]
    # ECM-6 R(QR)(Q(RW)) --> R0(Q0R1)(Q1(R2W0))
    elif ecm_num == 6:
        limit_list = [R_limit, Q_q_limit, Q_n_limit, R_limit, Q_q_limit, Q_n_limit, R_limit, WB_sigma_limit]
    # ECM-7 R(QR)W --> R0(Q0R1)W0
    elif ecm_num == 7:
        limit_list = [R_limit, Q_q_limit, Q_n_limit, R_limit, WB_sigma_limit]
    # ECM-8 R(Q(RW))Q --> R0(Q0(R1W0))Q1
    elif ecm_num == 8:
        limit_list = [R_limit, Q_q_limit, Q_n_limit, R_limit, WB_sigma_limit, Q_q_limit, Q_n_limit]
    # ECM-9 R(Q(R(QR))) --> R0(Q0(R1(Q1R2)))
    elif ecm_num == 9:
        limit_list = [[1e-15, 1e10], [1e-10, 1.0], [0.3, 1.0], [1e-10, 1e15], [1e-5, 1.0], [0.5, 1.0], [1e-13, 1e15]]
    else:
        print('We currently do not have ecm_num = {}'.format(ecm_num))
        sys.exit(0)
    return limit_list

def pack_lai_eis(exp_fn, ecm_num, fre_list, z_list):
    ecm_para_config_dict = {}
    # the name of file Containing EIS experiment data, it does not have appendix '.txt'
    # like '1-1', '5-35'
    ecm_para_config_dict['exp_fn'] = exp_fn
    ecm_para_config_dict['ecm_num'] = ecm_num

    """
    para limits:
        ECM-2
            2020-01-04 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                3.949E-26	    0.00001363	    0.4003	    0.06686	        0.00002237	    0.6	        0.06092
                Maximum:
                0.04596	        0.5935          0.9969      120500          0.6094          0.9445      31270
                Range
                [1e-27, 1e-1],  [1e-5, 1.0],    [0.3, 1.0], [1e-3, 1e7],    [1e-5, 1.0],    [0.5, 1.0], [1e-3, 1e6]
            2020-07-22 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                6.357E-17	    0.00001363	    0.4003	    0.06686	        0.00002237	    0.6	        0.06092
                Maximum:
                0.04596	        0.5935          0.9969      120500          0.6094          0.9445      3704
                Range
                [1e-18, 1e-1],  [1e-6, 1.0],    [0.3, 1.0], [1e-3, 1e7],    [1e-5, 1.0],    [0.5, 1.0], [1e-3, 1e5]            
        ECM-9
            2020-01-04 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                2.319E-28	    3.819E-11	    0.6666	    2.595E-08	    7.379E-11	    0.5157	    82.8
                Maximum:
                0.02633	        0.946           0.9986	    26460000	    0.0123	        0.968	    41850000000
                Range
                [1e-29, 1e-1],  [1e-12, 10],    [0.6, 1.0], [1e-9, 1e9],    [1e-12, 1e-1],  [0.5, 1.0], [10, 1e12]
            2020-07-22 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                5.426E-30	    3.819E-11	    0.6666	    2.595E-08	    7.379E-11	    0.5157	    82.8
                Maximum:
                0.02633	        0.002235        0.9986	    26460000	    0.0123	        0.968	    41850000000
                Range
                [1e-31, 1e-1],  [1e-12, 10],    [0.6, 1.0], [1e-9, 1e9],    [1e-12, 1e-1],  [0.5, 1.0], [10, 1e12]            
    """
    if ecm_num == 2:
        limit_list = [[1e-18, 1e-1], [1e-6, 1.0], [0.3, 1.0], [1e-3, 1e7], [1e-5, 1.0], [0.5, 1.0], [1e-3, 1e5]]
    elif ecm_num == 9:
        limit_list = [[1e-31, 1e-1], [1e-12, 10], [0.6, 1.0], [1e-9, 1e9], [1e-12, 1e-1], [0.5, 1.0], [10, 1e12] ]
    ecm_para_config_dict['limit'] = limit_list
    ecm_para_config_dict['f'] = fre_list
    ecm_para_config_dict['z_raw'] = z_list
    return ecm_para_config_dict

def goa_fitter_1(ecm_para_config_dict, repeat_time=1):
    """
    Function
    :param
        ecm_para_config_dict:{
            'exp_fn': the name of file Containing EIS experiment data, it does not have appendix '.txt', like '1-1', '5-35'
            'ecm_num': int
            'limit': [[para_0_low_limit, para_0_upper_limit], [para_1_low_limit, para_1_upper_limit], ...]
            'f': list[float, frequency ]
            'z_raw': list[complex, Impedance]
        }
        goa_fitting_res
            str, results file name
        repeat_time:
            int, Times of a GOA repeatedly fit a EIS data
    :return:
    """
    global goa_ecm_match_dict
    ecm_num = ecm_para_config_dict['ecm_num']
    # exp_fn = ecm_para_config_dict['exp_fn']
    exp_data_dict = ecm_para_config_dict

    iter_num = 5000 # Formal

    goa_fitting_res = ecm_para_config_dict['exp_fn'] + '_AD_Fit_Res.txt'
    with open(goa_fitting_res, 'a+') as file:
        line = 'ECM-Num,GOA-Name,Repeat-Time,Iteration,Fitted-Parameters-List,Chi-Square\n'
        file.write(line)

    goa_names_list = goa_ecm_match_dict[ecm_num]
    if ecm_num == 1:
        pass
    elif ecm_num == 2:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in current_best_wolf_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 9:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num=10 * para_num, entity_num=3 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)])+ ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue

def goa_fitter_0(ecm_para_config_dict, goa_fitting_res, repeat_time=1):
    """
    :param
        ecm_para_config_dict:{
            'exp_fn': the name of file Containing EIS experiment data, it does not have appendix '.txt', like '1-1', '5-35'
            'ecm_num': int
            'limit': [[para_0_low_limit, para_0_upper_limit], [para_1_low_limit, para_1_upper_limit], ...]
            'f': list[float, frequency ]
            'z_raw': list[complex, Impedance]
        }
        goa_fitting_res
            str, results file name
        repeat_time:
            int, Times of a GOA repeatedly fit a EIS data
    :return:
    """
    global goa_ecm_match_dict
    ecm_num = ecm_para_config_dict['ecm_num']
    # exp_fn = ecm_para_config_dict['exp_fn']
    exp_data_dict = ecm_para_config_dict

    # iter_num = 100 # Quick trial
    iter_num = 5000 # Formal

    # goa_fitting_res = get_date_prefix() + exp_fn + '_fitted_res_' + 'ecmNum={}'.format(ecm_num) + '.txt'
    with open(goa_fitting_res, 'a+') as file:
        line = 'ECM-Num,GOA-Name,Repeat-Time,Iteration,Fitted-Parameters-List,Chi-Square\n'
        file.write(line)

    goa_names_list = goa_ecm_match_dict[ecm_num]
    if ecm_num == 1:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    current_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num=10 * para_num, entity_num=3 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) + ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'MVO':
                for r in range(repeat_time):
                    mvo = MVO_EIS(exp_data_dict, iter_time=iter_num, universe_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = mvo.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].uni_objs_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'TLBO':
                for r in range(repeat_time):
                    tlbo = TLBO_EIS(exp_data_dict, iter_num=10 * para_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = tlbo.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 2:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 3:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'HS':
                for r in range(repeat_time):
                    hs = HS_EIS(exp_data_dict, iter_num=iter_num, harmony_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = hs.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].x_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=10*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'TLBO':
                for r in range(repeat_time):
                    tlbo = TLBO_EIS(exp_data_dict, iter_num=10 * para_num, entity_num=10*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = tlbo.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 4:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'CSS':
                for r in range(repeat_time):
                    css = CSS_EIS(exp_data_dict, iter_num=iter_num, particle_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = css.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].x_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'HS':
                for r in range(repeat_time):
                    hs = HS_EIS(exp_data_dict, iter_num=iter_num, harmony_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = hs.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].x_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 5:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'EP':
                for r in range(repeat_time):
                    ep = EP_EIS(exp_data_dict, iter_num=iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = ep.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in cur_best_entity_list[-1].x_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 6:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'HS':
                for r in range(repeat_time):
                    hs = HS_EIS(exp_data_dict, iter_num=iter_num, harmony_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = hs.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].x_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 7:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num=10 * para_num, entity_num=3 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)])+ ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 8:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GWO':
                for r in range(repeat_time):
                    gwo = GWO_EIS(exp_data_dict, iter_time=iter_num, wolf_num=10 * para_num)
                    current_best_wolf_list, iter, chi_squared = gwo.hunt()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in current_best_wolf_list[-1].position_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'HS':
                for r in range(repeat_time):
                    hs = HS_EIS(exp_data_dict, iter_num=iter_num, harmony_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = hs.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               +',['+ ','.join([str(para) for para in global_best_entity_list[-1].x_list])+'],'\
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'CSS':
                for r in range(repeat_time):
                    css = CSS_EIS(exp_data_dict, iter_num=iter_num, particle_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = css.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
    elif ecm_num == 9:
        para_num = len(ecm_para_config_dict['limit'])
        for goa_name in goa_names_list:
            if goa_name == 'DE':
                for r in range(repeat_time):
                    de = DE_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = de.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'WOA':
                for r in range(repeat_time):
                    woa = WOA_EIS(exp_data_dict, iter_time=iter_num, whale_num=10 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = woa.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'ABC':
                for r in range(repeat_time):
                    abc = ABC_EIS(exp_data_dict, iter_num, bee_num=10 * para_num, tabu_num=3*para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = abc.forage()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
                               + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].position_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue
            elif goa_name == 'GSO':
                for r in range(repeat_time):
                    gso = GSO_EIS(exp_data_dict, iter_num=10 * para_num, entity_num=3 * para_num)
                    cur_best_entity_list, global_best_entity_list, iter, chi_squared = gso.search()
                    with open(goa_fitting_res, 'a+') as file:
                        line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)])+ ',[' + ','.join(
                            [str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                               + str(chi_squared) + '\n'
                        file.write(line)
                continue

            # ICA的代码老是报错，先忽略，之后再去调试
            # elif goa_name == 'ICA':
            #     for r in range(repeat_time):
            #         ica = ICA_EIS(exp_data_dict, iter_num, entity_num=10 * para_num)
            #         cur_best_entity_list, global_best_entity_list, iter, chi_squared = ica.search()
            #         with open(goa_fitting_res, 'a+') as file:
            #             line = ','.join([str(ecm_num), goa_name, str(r), str(iter - 1)]) \
            #                    + ',[' + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
            #                    + str(chi_squared) + '\n'
            #             file.write(line)
            #     continue

def goa_fitter_multiProcess_0(que, p_name, ecm_para_config_dict, repeat_time):
    goa_fitter_0(ecm_para_config_dict, repeat_time)
    que.put(p_name)

def goa_fitter_multiProcess_Pool_0(input):
    ecm_para_config_dict, repeat_time = input
    goa_fitter_0(ecm_para_config_dict, repeat_time)