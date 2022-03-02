import os
import sys
import time
import math
from utils.file_utils.filename_utils import get_date_prefix

from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from ml_sl.logistic.lrc_0 import LRC

# 1-Import dataset (Training, validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
tr_va_dataset, test_dataset = get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)

tr_va_label_list, tr_va_data_list = split_labeled_dataset_list(tr_va_dataset)
te_label_list, te_data_list = split_labeled_dataset_list(test_dataset)
label_list = [2,4,5,6,7,8,9]

# 2-组合不同的OVR和OVO，并测试AK
# OVR的参数组合有5种，重复10次，故有50个模型；OVO的参数组合有5种，重复10次，故有50个模型；OVR-OVO一共有50*50=2500种组合
# 2.1 在Validation上测试
def lrc_vali_ovr_ovo_linear(ovr_linear_final_models_folder, ovr_model_name, ovo_linear_final_models_folder, ovo_model_name):
    global ml_dataset_pickle_file_path
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
    vali_label_list, vali_dataset_list = split_labeled_dataset_list(validation_dataset)
    lrc = LRC(alpha=0, max_iter=0, unlabeled_dataset_list=vali_dataset_list, \
              labeled_dataset_list=None, label_list=label_list)

    lrc_ovr_classifer_dict_pickle_filename = os.path.join(ovr_linear_final_models_folder, ovr_model_name)
    lrc_ovo_linear_classifer_dict_pickle_filename = os.path.join(ovo_linear_final_models_folder, ovo_model_name)

    sample_label_prob_dict_list, ovo_vote_rate = lrc.classify_ovr_ovo(lrc_ovr_classifer_dict_pickle_filename, \
                                                                      lrc_ovo_linear_classifer_dict_pickle_filename)
    lrc_acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    lrc_kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('On vali: Acc={0}, kappa={1}, ovo_vote_rate={2}'.format(lrc_acc, lrc_kappa, ovo_vote_rate))
ovr_linear_final_models_folder = 'ovr_res/linear_final/models'
ovo_linear_final_models_folder = 'ovo_res/linear_final/models'
ovr_model_name = '2020_05_08_lrc_ovr_linear_final_iter=3000_alpha_init=1_classifer_dict_pickle_5.file'
ovo_model_name = '2020_04_22_lrc_ovo_linear_final_iter=7000_alpha_init=1000_classifer_dict_pickle_3.file'
# lrc_vali_ovr_ovo_linear(ovr_linear_final_models_folder, ovr_model_name, ovo_linear_final_models_folder, ovo_model_name)
# R(RC)_IS_lin-kk_res.txt: On vali: Acc=0.7446808510638298, kappa=0.6899395272127542, ovo_vote_rate=0.8404255319148937, AK = 1.434620378276584

# 2.2 在Test上测试
def lrc_test_ovr_ovo_linear(ovr_linear_final_models_folder, ovo_linear_final_models_folder):
    global te_label_list, te_dataset_list, label_list
    # Load OVR model
    ovr_model_names = os.listdir(ovr_linear_final_models_folder)
    ovo_model_names = os.listdir(ovo_linear_final_models_folder)
    for i in range(len(ovr_model_names)):
        for j in range(len(ovo_model_names)):
            lrc = LRC(alpha=0, max_iter=0, unlabeled_dataset_list=te_dataset_list,\
                      labeled_dataset_list=None, label_list=label_list)

            ovr_model_name = ovr_model_names[i]
            lrc_ovr_classifer_dict_pickle_filename = os.path.join(ovr_linear_final_models_folder, ovr_model_name)
            ovo_model_name = ovo_model_names[j]
            lrc_ovo_linear_classifer_dict_pickle_filename = os.path.join(ovo_linear_final_models_folder, ovo_model_name)

            sample_label_prob_dict_list, ovo_vote_rate = lrc.classify_ovr_ovo(lrc_ovr_classifer_dict_pickle_filename,\
                                                                              lrc_ovo_linear_classifer_dict_pickle_filename)
            lrc_acc = cal_accuracy(sample_label_prob_dict_list, te_label_list)
            lrc_kappa = cal_kappa(sample_label_prob_dict_list, te_label_list)

            res_file_name = get_date_prefix() + 'ovr_ovo_linear_final.txt'
            with open(res_file_name, 'a+') as file:
                line_str = ','.join([ovr_model_name, ovo_model_name, str(lrc_acc), str(lrc_kappa), str(ovo_vote_rate)]) + '\n'
                print(line_str)
                file.write(line_str)
# ovr_linear_final_models_folder = 'ovr_res/linear_final/models'
# ovo_linear_final_models_folder = 'ovo_res/linear_final/models'
# lrc_test_ovr_ovo_linear(ovr_linear_final_models_folder, ovo_linear_final_models_folder)
"""
Final best R(RC)_IS_lin-kk_res.txt (Test):
    OvR model: 2020_05_08_lrc_ovr_linear_final_iter=3000_alpha_init=1_classifer_dict_pickle_5.file
   +OvO model: 2020_04_22_lrc_ovo_linear_final_iter=7000_alpha_init=1000_classifer_dict_pickle_3.file
   accuracy=0.494505495;
   kappa=0.380585972;
   ovo usage rate=0.791208791;
   AK=0.875091467
"""

# 3- Access the accuracy of the first one, two, and three predictions
def get_acc_on_first_3_predictions():
    global te_label_list, te_dataset_list, label_list
    ovr_linear_final_models_folder = 'ovr_res/linear_final/models'
    ovo_linear_final_models_folder = 'ovo_res/linear_final/models'

    ovr_model_name = '2020_05_08_lrc_ovr_linear_final_iter=3000_alpha_init=1_classifer_dict_pickle_5.file'
    ovo_model_name = '2020_04_22_lrc_ovo_linear_final_iter=7000_alpha_init=1000_classifer_dict_pickle_3.file'

    lrc_ovr_classifer_dict_pickle_filename = os.path.join(ovr_linear_final_models_folder, ovr_model_name)
    lrc_ovo_linear_classifer_dict_pickle_filename = os.path.join(ovo_linear_final_models_folder, ovo_model_name)

    lrc = LRC(alpha=0, max_iter=0, unlabeled_dataset_list=te_dataset_list, \
              labeled_dataset_list=None, label_list=label_list)
    sample_label_prob_dict_list, ovo_vote_rate = lrc.classify_ovr_ovo(lrc_ovr_classifer_dict_pickle_filename, \
                                                                      lrc_ovo_linear_classifer_dict_pickle_filename)
    lrc_acc = cal_accuracy(sample_label_prob_dict_list, te_label_list)
    lrc_acc_on_2 = cal_accuracy_on_2(sample_label_prob_dict_list, te_label_list)
    lrc_acc_on_3 = cal_accuracy_on_3(sample_label_prob_dict_list, te_label_list)
    lrc_kappa = cal_kappa(sample_label_prob_dict_list, te_label_list)

    print('Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(\
          lrc_acc, lrc_acc_on_2, lrc_acc_on_3, lrc_kappa))
# get_acc_on_first_3_predictions()
# Accuracy on 1 = 0.4945054945054945, Accuracy on 2 = 0.6593406593406593, Accuracy on 3 = 0.7472527472527473, Kappa=0.38058597218111867

# 4-将Final最佳的模型 OvR-OvO 在 trVa 和 te 上的 测试结果 计算出来
def finalModelRes():
    ovrModelFp = os.path.join('ovr_res/linear_final/models',
                              '2020_05_08_lrc_ovr_linear_final_iter=3000_alpha_init=1_classifer_dict_pickle_5.file')
    ovoModelFp = os.path.join('ovo_res/linear_final/models',
                              '2020_04_22_lrc_ovo_linear_final_iter=7000_alpha_init=1000_classifer_dict_pickle_3.file')
    lrcFinalTrVa = LRC(alpha=0, max_iter=0,
                       unlabeled_dataset_list=tr_va_data_list,
                       labeled_dataset_list=None, label_list=label_list)

    lrcFinalTrVaSample_label_prob_dict_list, lrcFinalTrVaOvo_vote_rate = lrcFinalTrVa.classify_ovr_ovo(ovrModelFp, ovoModelFp)
    lrcFinalTrVaAcc = cal_accuracy(lrcFinalTrVaSample_label_prob_dict_list, tr_va_label_list)
    lrcFinalTrVaKappa = cal_kappa(lrcFinalTrVaSample_label_prob_dict_list, tr_va_label_list)
    print('lrcFinalTrVa: ACC={0}, Kappa={1}'.format(lrcFinalTrVaAcc, lrcFinalTrVaKappa))
    # lrcFinalTrVa: ACC=0.7323420074349443, Kappa=0.6710164805999431, AK=1.4033584880348875
    
    lrcFinalTe = LRC(alpha=0, max_iter=0,
                     unlabeled_dataset_list=te_data_list,
                     labeled_dataset_list=None, label_list=label_list)

    lrcFinalTeSample_label_prob_dict_list, lrcFinalTeOvo_vote_rate = lrcFinalTe.classify_ovr_ovo(ovrModelFp, ovoModelFp)
    lrcFinalTeAcc = cal_accuracy(lrcFinalTeSample_label_prob_dict_list, te_label_list)
    lrcFinalTeKappa = cal_kappa(lrcFinalTeSample_label_prob_dict_list, te_label_list)
    print('lrcFinalTe: ACC={0}, Kappa={1}'.format(lrcFinalTeAcc, lrcFinalTeKappa))
    # lrcFinalTe: ACC=0.4945054945054945, Kappa=0.38058597218111867, AK=0.8750914666866132
# finalModelRes()