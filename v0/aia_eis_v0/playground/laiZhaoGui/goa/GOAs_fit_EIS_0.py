import pickle

from utils.file_utils.filename_utils import get_date_prefix

# 1- Read Lai's normed EIS data
def load_eis_normed_data(normed_file_path):
    """
    :param
        file_folder:
    :return:
        eis_data_dict:
             dict{
                    '1-1'  : {'fre': fre_list[float], 'z':normed_z_list[complex]}
                    '1-13' : {'fre': fre_list[float], 'z':normed_z_list[complex]}
                    '3-15' : {'fre': fre_list[float], 'z':normed_z_list[complex]}
                }
    """
    with open(normed_file_path, 'r') as file:
        eis_normed_data_dict = pickle.load(file)
    return eis_normed_data_dict

# 2- Load the corresponding ECM_Num of eis-data file
def load_ecm_Num(file_path):
    """
    :param
        file_folder:
    :return:
         ecm_record_dict:
            dict{
                    2:['2-13','2-15','2-17',...],
                    9:['1-1','1-6','3-1',...]
                }
    """
    ecm_record_dict = {2:[], 9:[]}
    with open(file_path, 'r') as file:
        for line in file.readlines():
            fn, ecm_num_str = line.strip().split(',')
            ecm_record_dict[int(ecm_num_str)].append(fn)
    return ecm_record_dict

def get_para_range(num, mag_num=2):
    """
    Function:
        Get the magnitude m of the input number
        Take 1e^(m - 3) ~ 1e^(m + 3)
    :param
        num: float
        mag_num: int
    :return:
        range list, [min, max]
    refer:
        python基础-python3-格式化浮点数方法-%e、%f、%g: https://www.cnblogs.com/summer1019/p/11387889.html
            %e是用科学记数法计数
            %f是按指定精确格式化浮点数（默认保留6位）
            %g是根据数值的大小采用e或%f
    """
    num_str = '%e'%num
    mag_str = num_str.split('e')[-1]
    mag = int(mag_str)

    # p_min = 10 ** (mag - 3)
    # p_max = 10 ** (mag + 3)
    p_min = 10 ** (mag - mag_num)
    p_max = 10 ** (mag + mag_num)
    return [p_min, p_max]

# 3- Generate limit for each EIS-data, for CPE_n, its range is set as [CPE_n_min - 0.1, 1.0]
def load_lai_eis_fitting_res(lai_ecm2_file_path, lai_ecm9_file_path):
    """
    Function:
        1- Take the Lai's fitting results as references to set the range of parameters
        2- There are three kinds of circuit elements: R, CPE_q, CPE_n.
        3- For R and CPE_q, if Lai's fitting result for parameter p is A * 1eB (A: float < 10, B: int),
            then the range of p is 1e(B-2) ~ 1e(B+2)
           For CPE_n, its range is 0.35 ~ 1.0
    :param
        folder:
    :return:
        limit_dict {
            '1-1': [[p0_min, p0_max], [p1_min, p1_max],...],
            '1-3': [[p0_min, p0_max], [p1_min, p1_max],...],
        }
    """
    lai_fit_eis_res_dict = {}
    # ECM-2 R0(Q0 R1)(Q1 R2)
    with open(lai_ecm2_file_path, 'r') as file:
        ecm_num = 2
        for line in file.readlines():
            line_str_list = line.strip().split(',')
            # file name, '2-29', '5-17'
            fn = line_str_list[1]

            R0 = float(line_str_list[2])
            R0_range = get_para_range(R0)

            CPE0_q = float(line_str_list[3])
            CPE0_q_range = get_para_range(CPE0_q)
            # CPE0_n = float(line_str_list[])
            CPE0_n_range = [0.35, 1]

            R1 = float(line_str_list[5])
            R1_range = get_para_range(R1)

            CPE1_q = float(line_str_list[6])
            CPE1_q_range = get_para_range(CPE1_q)
            # CPE1_n = float(line_str_list[])
            CPE1_n_range = [0.35, 1]

            R2 = float(line_str_list[8])
            R2_range = get_para_range(R2)

            lai_chi_squared = float(line_str_list[-1])

            # R0, Q0_pair_q, Q0_pair_n, R1, Q1_pair_q, Q1_pair_n, R2
            limit_list = [R0_range,
                          CPE0_q_range, CPE0_n_range,
                          R1_range,
                          CPE1_q_range, CPE1_n_range,
                          R2_range]

            lai_fit_eis_res_dict[fn] = {'ecm_num': ecm_num,
                                 'limit':limit_list,
                                 'lai_Chi-Squared': lai_chi_squared}

    # ECM-9 R0(Q0(R1(Q1R2)))
    with open(lai_ecm9_file_path, 'r') as file:
        ecm_num = 9
        for line in file.readlines():
            line_str_list = line.strip().split(',')
            fn = line_str_list[1]

            R0 = float(line_str_list[2])
            R0_range = get_para_range(R0)

            CPE0_q = float(line_str_list[3])
            CPE0_q_range = get_para_range(CPE0_q)
            # CPE0_n = float(line_str_list[])
            CPE0_n_range = [0.35, 1]

            R1 = float(line_str_list[5])
            R1_range = get_para_range(R1)

            CPE1_q = float(line_str_list[6])
            CPE1_q_range = get_para_range(CPE1_q)
            # CPE1_n = float(line_str_list[])
            CPE1_n_range = [0.35, 1]

            R2 = float(line_str_list[8])
            R2_range = get_para_range(R2)

            # In Lai's fitting results, there are few places where they Chi-Squared are missing,
            # I fill 'Null' in the cell. At here, i use None to fill the blank
            lai_chi_squared = None
            if line_str_list[-1] != 'Null':
                lai_chi_squared = float(line_str_list[-1])

            # R0, Q0_pair_q, Q0_pair_n, R1, Q1_pair_q, Q1_pair_n, R2
            limit_list = [R0_range,
                          CPE0_q_range, CPE0_n_range,
                          R1_range,
                          CPE1_q_range, CPE1_n_range,
                          R2_range]

            lai_fit_eis_res_dict[fn] = {'ecm_num': ecm_num,
                                 'limit': limit_list,
                                 'lai_Chi-Squared': lai_chi_squared}
    return lai_fit_eis_res_dict

def fit_lai_eis_data(normed_file_path, lai_ecm2_file_path, lai_ecm9_file_path):
    # gather all the requirements: ecm_num, limit, fre_list, z_list, Chi-Squared
    eis_full_data_dict = {}
    eis_normed_data_dict = load_eis_normed_data(normed_file_path)
    lai_fit_eis_res_dict = load_lai_eis_fitting_res(lai_ecm2_file_path, lai_ecm9_file_path)

    for k, v in eis_normed_data_dict.items():
        eis_full_data_dict[k] = {'fre': v['fre'],
                                 'z_normed': v['z'],
                                 'ecm_num': lai_fit_eis_res_dict['ecm_num'],
                                 'limit': lai_fit_eis_res_dict['limit'],
                                 'lai_Chi-Squared': lai_fit_eis_res_dict['lai_Chi-Squared']
                                 }

    res_fn = get_date_prefix() + 'goa_fit_lai_eis_res.txt'
    for k, v in eis_full_data_dict.items():
        fn = k
        ecm_num = v['ecm_num']

        # GOA casual setting:
        iter_num = 10000
        entity_num = 10 * len(v['limit'])
        if ecm_num == 2:
            # DE? not Known yet
            cur_best_entity_list, global_best_entity_list, iter_num, chi_squared = goa.search()
            # line = raw-eis-filename + ecm_num + fitted_para_list + chi-Squared
            line = ','.join([fn]
                            + [str(para) for para in global_best_entity_list.x_list]
                            + [str(iter_num), str(chi_squared)]) + '\n'
            pass
        elif ecm_num == 9:
            # DE? not Known yet
            pass
        else:
            print('Lai only has ECM 2 and 9, {0} comes out from no where, check it')
        with open(res_fn, 'a+') as file:
            file.write(line)

        # 3- Fit each EIS by a GOA (iter_num = 10000, entity_num = 10 * para_len, the same as the accessment)