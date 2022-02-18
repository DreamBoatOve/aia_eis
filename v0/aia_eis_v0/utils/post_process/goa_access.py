import copy
import math
import os
import sys
sys.path.append('../../')

from utils.file_utils.filename_utils import get_date_prefix

class goa_accessor:
    """
    Function
        Get a GOA's averaged Chi-Squared, Averaged Running Time and RMSE of Chi-Squared on 9 ECMs, respectively
    Input:
        res_folder: Folder of results (900 files) of a GOA trained on 9 ECMs:
        goa_name:
            like:
                res_folder = '../../../../large_files/dpfc/goa_res/evolution/de'
                goa_name = 'de'
    Output:
        a txt file
            file name: date_goaName_result.txt
            ecm_No, accuracy_score, avg_time(s),    rmse(Chi-Squared)
            1,      10,             5369.8,         50.5
            2,      5,              465.6,          12.38
            3,      7,              8553.8,         89.36
    """
    def __init__(self, res_folder, goa_name):
        self.res_folder = res_folder
        self.goa_name = goa_name
        self.fn_dict = self.divide_file_by_ecm_Num()
        self.selected_data_dict = self.exclude_extreme_res()
        self.res_dict = self.get_result()

    def divide_file_by_ecm_Num(self):
        """
        Function:
            In a GOA trained results folder, firstly divide all the file names in the folder by ecm_No(ECM kind),
            and store the file names in different dict tree, like:
                fn_dict = {
                            1 (ECM1): [goa_ecm1_00.txt, goa_ecm1_01.txt, goa_ecm1_02.txt](len=100)
                            2 (ECM2): [goa_ecm2_00.txt, goa_ecm1_02.txt, goa_ecm2_02.txt](len=100)
                            ...
                            9 (ECM9): [goa_ecm9_00.txt, goa_ecm9_01.txt, goa_ecm9_02.txt](len=100)
                            }
        :return:
        """
        fn_dict = {}

        fn_list = os.listdir(self.res_folder)
        for fn in fn_list:
            if self.goa_name == 'bb_bc':
                ecm_No = int(fn.split('_')[2][-1])
            else:
                ecm_No = int(fn.split('_')[1][-1])
            if ecm_No not in fn_dict.keys():
                fn_dict[ecm_No] = [fn]
            else:
                fn_dict[ecm_No].append(fn)
        return fn_dict

    def exclude_extreme_res(self):
        """
        Function:
            Delete the best 5% and the worst 5% (the lower Chi-Squared, the better),
            the left 90% data is taken into further consideration and stored in selected_fn_dict
        :return:
        """
        selected_data_dict = {}
        for k, v in self.fn_dict.items():
            # data_list = [[chi-squared, time](from file1), [chi-squared, time](from file2)]
            data_list = []
            for fn in v:
                file_path = os.path.join(self.res_folder, fn)
                with open(file_path, 'r') as file:
                    # Take the last/bottom line in the result files directly
                    for line in reversed(file.readlines()):
                        """
                        Old Line:
                            这里的Chi-Square是不正确的，没有除以v（自由度v = 测试点数N - ECM待拟合参数数量M）
                            Iteration,  para_list[float, ...],                                                  Chi-Square(错误版本),     Cumulated Running Time
                            9984,       [0.02012484780316051,0.0001799896579117755, ..., 52972.38087848809],    4.433244077492469e-06,  124.94990979999966
                        New Line:
                            这里的Chi-Square是正确的，除以v（自由度v = 测试点数N - ECM待拟合参数数量M）
                            Iteration,  para_list[float, ...],                                                  Chi-Square(错误版本),       Chi-Square(Right Version),    Cumulated Running Time
                            9984,       [0.02012484780316051,0.0001799896579117755, ..., 52972.38087848809],    4.433244077492469e-06,      4.433244077492469e-06 / v,  124.94990979999966
                        """
                        if len(line.strip()) > 0:
                            # x == Chi-Squared
                            line_str_list = line.strip().split(',')
                            x = float(line_str_list[-2])
                            # t = training time
                            t = float(line_str_list[-1])
                            data_list.append([x, t])
                            break
            data_list.sort(key=lambda d:d[0], reverse=False)
            """
            Remove the first 5% and last 5% extreme data
                Considering the training time of ACA is too long, I just train it for 2000 times, sometimes a little 
                bit more than 2000 times. Until 2020-08-13, I do not finish all the training of ACA on nine ECMs, but I think 
                the training results is enough, can give it a go, Something like, I have 40 results on ECM-6's fitting,
                'selected_data_list = copy.deepcopy(data_list[5:95])' will be wrong, have to change it to the following: 
            """
            # selected_data_list = copy.deepcopy(data_list[5:95])
            selected_data_list = copy.deepcopy(data_list[int(0.05 * len(data_list)) : int(0.95 * len(data_list))])
            selected_data_dict[k] = selected_data_list
        return selected_data_dict

    def get_result(self):
        """
        Function:
            Score the accuracy, get averaged running time and score later, get RMSE of Chi-Squared and score later
                Get the averaged accuracy
                    The smaller magnitude of Chi-Squared, the higher score
                    The samllest Chi-Squared    -->     The biggest Chi-Squared
                    10                          -->     0
                Get the averaged running time
                Get RMSE of Chi-Squared
        :return:
        """
        res_dict = {}
        for k, v in self.selected_data_dict.items():
            x_sum = 0.0
            t_sum = 0.0
            for data in v:
                x, t = data
                x_sum += x
                t_sum += t
            x_mean = x_sum / len(v)
            t_mean = t_sum / len(v)

            x_rmse = math.sqrt(sum([(data[0] - x_mean) ** 2 for data in v]) / len(v))

            # Give Accuracy Score by the magnitude of chi-squared. the smaller magnitude, the higher score
            # acc_score = None
            # if x_mean < 1e-34:
            #     acc_score = 10
            # elif x_mean < 1e-30:
            #     acc_score = 9
            # elif x_mean < 1e-26:
            #     acc_score = 8
            # elif x_mean < 1e-22:
            #     acc_score = 7
            # elif x_mean < 1e-18:
            #     acc_score = 6
            # elif x_mean < 1e-14:
            #     acc_score = 5
            # elif x_mean < 1e-10:
            #     acc_score = 4
            # elif x_mean < 1e-6:
            #     acc_score = 3
            # elif x_mean < 1e-2:
            #     acc_score = 2
            # elif x_mean < 1e2:
            #     acc_score = 1
            # else:
            #     acc_score = 0

            res_dict[k] = [x_mean, t_mean, x_rmse]
        return res_dict

    def output_res_2_txt(self):
        fn = get_date_prefix() + self.goa_name + '_res.txt'
        for k, v in self.res_dict.items():
            with open(fn, 'a+') as file:
                # line = ecm_No + averaged_Chi-Squared + averaged_train-time + RMSE_Chi-Squared
                # line = ','.join([str(k)] + [str(d) for d in v]) + '\n'

                # line = ecm_No + Chi-Squared-Score + averaged_train-time + RMSE_Chi-Squared

                # line = ecm_No + averaged Chi-Squared + averaged_train-time + RMSE_Chi-Squared
                line = ','.join([str(k)] + [str(d) for d in v]) + '\n'
                file.write(line)
#--------------------- Evolution ---------------------
# ------------ DE ------------
# res_folder 在台式机上存着
# de_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/evolution/de',
#                            goa_name='de')
# de_accessor.output_res_2_txt()
# txt R(RC)_IS_lin-kk_res.txt is stored at src\global_optimizations\evolution_based\fit_simECM_res\2st\2021_02_06_de_res.txt
# txt R(RC)_IS_lin-kk_res.txt is stored at src\global_optimizations\evolution_based\fit_simECM_res\1st\2020_07_14_de_res.txt
# ------------ DE ------------

# ------------ EDA ------------
# eda_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/evolution/eda',
#                            goa_name='eda')
# eda_accessor.output_res_2_txt()
# txt R(RC)_IS_lin-kk_res.txt is stored at src\global_optimizations\evolution_based\fit_simECM_res\1st\2020_07_14_eda_res.txt
# ------------ EDA ------------

# ------------ EP ------------
# ep_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/evolution/ep',
#                            goa_name='ep')
# ep_accessor.output_res_2_txt()
# txt R(RC)_IS_lin-kk_res.txt is stored at src\global_optimizations\evolution_based\fit_simECM_res\1st\2020_07_14_ep_res.txt
# ------------ EP ------------

# ------------ ES ------------
# es_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/evolution/es',
#                            goa_name='es')
# es_accessor.output_res_2_txt()
# txt R(RC)_IS_lin-kk_res.txt is stored at src\global_optimizations\evolution_based\fit_simECM_res\1st\2020_07_14_es_res.txt
# ------------ ES ------------

# ------------ GA ------------
# ga_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/evolution/ga',
#                            goa_name='ga')
# ga_accessor.output_res_2_txt()
# ------------ GA ------------
#--------------------- Evolution ---------------------

#--------------------- Human ---------------------
# ------------ GSO ------------
# gso_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/human/gso',
#                            goa_name='gso')
# gso_accessor.output_res_2_txt()
# ------------ GSO ------------

# ------------ HS ------------
# hs_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/human/hs',
#                            goa_name='hs')
# hs_accessor.output_res_2_txt()
# ------------ HS ------------

# ------------ ICA ------------
# ica_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/human/ica',
#                            goa_name='ica')
# ica_accessor.output_res_2_txt()
# ------------ ICA ------------

# ------------ ISA ------------
# isa_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/human/isa',
#                            goa_name='isa')
# isa_accessor.output_res_2_txt()
# ------------ ISA ------------

# ------------ TLBO ------------
# tlbo_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/human/tlbo',
#                            goa_name='tlbo')
# tlbo_accessor.output_res_2_txt()
# ------------ TLBO ------------
#--------------------- Human ---------------------

#--------------------- Physic ---------------------
# ------------ BB_BC ------------
# bb_bc_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/physic/bb_bc',
#                               goa_name='bb_bc')
# bb_bc_accessor.output_res_2_txt()
# ------------ BB_BC ------------

# ------------ BH ------------
# bh_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/physic/bh',
#                               goa_name='bh')
# bh_accessor.output_res_2_txt()
# ------------ BH ------------

# ------------ CSS ------------
# css_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/physic/css',
#                               goa_name='css')
# css_accessor.output_res_2_txt()
# ------------ CSS ------------

# ------------ GSA ------------
# gsa_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/physic/gsa',
#                               goa_name='gsa')
# gsa_accessor.output_res_2_txt()
# ------------ GSA ------------

# ------------ MVO ------------
# mwo_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/physic/mwo',
#                               goa_name='mwo')
# mwo_accessor.output_res_2_txt()
# ------------ MVO ------------
#--------------------- Physic ---------------------

#--------------------- Swarm ---------------------
# ------------ ACA ------------
# aca_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/swarm/aca',
#                               goa_name='aca')
# aca_accessor.output_res_2_txt()
# ------------ ACA ------------

# ------------ ABC ------------
# abc_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/swarm/abc',
#                               goa_name='abc')
# abc_accessor.output_res_2_txt()
# ------------ ABC ------------

# ------------ GWO ------------
# gwo_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/swarm/gwo',
#                               goa_name='gwo')
# gwo_accessor.output_res_2_txt()
# ------------ GWO ------------

# ------------ PSO ------------
# pso_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/swarm/pso',
#                               goa_name='pso')
# pso_accessor.output_res_2_txt()
# ------------ PSO ------------

# ------------ WOA ------------
# woa_accessor = goa_accessor(res_folder='../../../../dpfc_large_files/goa_res/swarm/woa',
#                               goa_name='woa')
# woa_accessor.output_res_2_txt()
# ------------ WOA ------------
#--------------------- Swarm ---------------------