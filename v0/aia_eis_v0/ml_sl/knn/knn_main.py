import datetime
import math

from ml_sl.knn.knn_0 import KNN
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from utils.file_utils.filename_utils import get_date_prefix
# from utils.visualize_utils.ml_heatmap_utils import knn_heatmap
# -------------------------------------- General routine test of KNN --------------------------------------
"""
Plan-1: Steps
    1-Import dataset (Training, validation, Test)
    2-Training
        Use Training+Validation dataset to train KNN
    3-Test and measure its performance (Accuracy, Kappa)
        Use Test dataset to train KNN
    4-Use grid search to find the best parameters(Distance measure + K) of KNN
"""

# 1-Import dataset (Training, validation, Test)
# ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
# tr_va_dataset, test_dataset = get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)

# 2-Training (Use Training + Validation dataset to train KNN)
# 2.1 parameters setting
# Separate the test_dataset into label and data
# tr_va_label_list, tr_va_data_list = split_labeled_dataset_list(tr_va_dataset)
# test_label_list, test_dataset_list = split_labeled_dataset_list(test_dataset)
# unlabeled_dataset_list = test_dataset_list

# labeled_dataset_list = tr_va_dataset
#
# distance_mode = 'bc_d'
#
# label_list = [2,4,5,6,7,8,9]
# K = 1
#
# # 2.2 KNN Training
# knn = KNN(K, unlabeled_dataset_list, labeled_dataset_list, distance_mode, label_list)
# knn_sample_label_prob_dict_list = knn.classify()
#
# # 3-Test and measure its performance (Accuracy, Kappa)
# # Accuracy
# knn_acc = cal_accuracy(knn_sample_label_prob_dict_list, test_label_list)
# print('KNN Accuracy', knn_acc)
# # Kappa
# knn_kappa = cal_kappa(knn_sample_label_prob_dict_list, test_label_list)
# print('KNN Kappa', knn_kappa)
# -------------------------------------- General routine test of KNN --------------------------------------

# -------------------------------------- Standard routine test of KNN --------------------------------------
"""
Plan-2 in MindMap: 
    Steps
        1-Import dataset (Training, validation, Test)
        2-Training model with training-dataset and access their performance(Accuracy, Kappa) on validation-dataset
            Use grid search to find the best parameters(Distance measure + K) of KNN
        3-Access the performance of KNN under the best parameters(Distance measure + K) on Training+Validation, Test datasets, seperately.
"""
# 1-Import dataset (Training, validation, Test)
# ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
# training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)

# 2-Training model with training-dataset and assess their performance(Accuracy, Kappa) on validation-dataset
# 2.1 Combine all the distance measures and K-candidate
# distance_mode_list = ['bc_d','cheb_d','cos_d','dtw_d','e_d','em_d','jsd_d','maha_d','manha_d','pcc_d','se_d']
# K_list = [i for i in range(1, int(math.sqrt(len(training_dataset) + len(validation_dataset) + len(test_dataset))))]
# for distance_mode in distance_mode_list:
#     for K in K_list:
#         # 2.2 Use training-dataset and the distance_mode+K to get a KNN
#         validation_label_list, validation_dataset_list = split_labeled_dataset_list(validation_dataset)
#         label_list = [2, 4, 5, 6, 7, 8, 9]
#         knn = KNN(K=K,  unlabeled_dataset_list=validation_dataset_list,\
#                         labeled_dataset_list=training_dataset,\
#                         distance_mode=distance_mode, label_list=label_list)
#         knn_sample_label_prob_dict_list = knn.classify()
#         # 2.3 Access the performance of the KNN by (Accuracy and Kappa)
#         knn_acc = cal_accuracy(knn_sample_label_prob_dict_list, validation_label_list)
#         knn_kappa = cal_kappa(knn_sample_label_prob_dict_list, validation_label_list)
#         # 2.4 Serialize the result into a file(Distance_mode, K, Accuracy, Kappa)
#         year = str(datetime.datetime.now().year)
#         month = str(datetime.datetime.now().month)
#         day = str(datetime.datetime.now().day)
#         file_name = year+'_'+month+'_'+day+'_KNN_T_V.txt'
#         with open(file_name, 'a+') as file:
#             line_str = distance_mode+','+str(K)+','+str(knn_acc)+','+str(knn_kappa)+'\n'
#             file.write(line_str)
# 2.5 Manually select the best parameters(Distance_mode, K) by the average of Accuracy and Kappa

# 3 Visualize the results of grid search for finding optimal parameter setting
# knn_heatmap(txt_file_path='./plots/2020_11_23_KNN_GS_res.txt',
#             axis_title_list=['Distance Measure', 'Nearest Neighbors Number (K)', 'AK'])
"""
Top 5 records
    Distance measure	K	Accuracy(validation)	Kappa(validation)	Accuracy + kappa	Rank
    se_d	            6	    0.553191489	        0.420859616	        0.974051105	        1
    manha_d	            14	    0.542553191	        0.395272292	        0.937825484	        2
    e_d	                14      0.542553191	        0.393912131	        0.936465322	        3
    se_d	            5	    0.531914894	        0.396820767	        0.928735661	        4
    jsd_d	            2       0.531914894	        0.389790499	        0.921705392	        5
"""
# top5_para_pairs = [['se_d', 6],['manha_d', 14],['e_d', 14],['se_d', 5],['jsd_d', 2]]
# 4- Use the best parameters(Distance_mode, K) and training+validation_dataset to build the final KNN
# Merge training and validation datasets into tr_va_dataset
# training_dataset.extend(validation_dataset)
# tr_va_dataset = training_dataset
#
# test_label_list, test_dataset_list = split_labeled_dataset_list(test_dataset)
# label_list = [2, 4, 5, 6, 7, 8, 9]
#
# for para_pair in top5_para_pairs:
#     distance_mode = para_pair[0]
#     K = para_pair[1]
#     knn = KNN(K = K, unlabeled_dataset_list = test_dataset_list, labeled_dataset_list = tr_va_dataset,\
#               distance_mode = distance_mode, label_list = label_list)
#     knn_sample_label_prob_dict_list = knn.classify()
#     # 4-Access the performance of the final KNN on training+validation_dataset and test_dataset, respectively
#     knn_acc = cal_accuracy(knn_sample_label_prob_dict_list, test_label_list)
#     knn_kappa = cal_kappa(knn_sample_label_prob_dict_list, test_label_list)
#     print('Distance mode={0}, K={1}, Accuracy={2}, Kappa={3}'.format(distance_mode, K, knn_acc, knn_kappa))
"""
Distance mode=se_d, K=6, Accuracy=0.5274725274725275, Kappa=0.37301714468835123
Distance mode=manha_d, K=14, Accuracy=0.5384615384615384, Kappa=0.3810526315789473
Distance mode=e_d, K=14, Accuracy=0.5384615384615384, Kappa=0.3840451248992747
Distance mode=se_d, K=5, Accuracy=0.5274725274725275, Kappa=0.3764143426294821
Distance mode=jsd_d, K=2, Accuracy=0.4945054945054945, Kappa=0.33555555555555555
"""

# 5- Access top 5 KNN para settings' accuracy on the first one, two, and three predictions, the procedure is similar with step 4
# 英语可能表达的不清楚，中文再解释一下：计算预测的前第一，二，三个的正确率，必然会越来越高，但是能搞到哪里，还得实验看一下
# top5_para_pairs = [['se_d', 6],['manha_d', 14],['e_d', 14],['se_d', 5],['jsd_d', 2]]
#
# test_label_list, test_dataset_list = split_labeled_dataset_list(test_dataset)
# label_list = [2, 4, 5, 6, 7, 8, 9]
#
# for para_pair in top5_para_pairs:
#     distance_mode = para_pair[0]
#     K = para_pair[1]
#     knn = KNN(K = K, unlabeled_dataset_list = test_dataset_list, labeled_dataset_list = tr_va_dataset,\
#               distance_mode = distance_mode, label_list = label_list)
#     knn_sample_label_prob_dict_list = knn.classify()
#     # 4-Access the performance of the final KNN on training+validation_dataset and test_dataset, respectively
#     knn_acc = cal_accuracy(knn_sample_label_prob_dict_list, test_label_list)
#     knn_acc_on_2 = cal_accuracy_on_2(knn_sample_label_prob_dict_list, test_label_list)
#     knn_acc_on_3 = cal_accuracy_on_3(knn_sample_label_prob_dict_list, test_label_list)
#     knn_kappa = cal_kappa(knn_sample_label_prob_dict_list, test_label_list)
#     print('Distance mode = {0}, K = {1}, Accuracy on 1 = {2}, Accuracy on 2 = {3}, Accuracy on 3 = {4}, Kappa={5}'.format(
#         distance_mode, K, knn_acc, knn_acc_on_2, knn_acc_on_3, knn_kappa))

"""
Trained on tr+va, Tested on te
Results:
    Distance mode = se_d, K = 6, 
        Accuracy on 1 = 0.5274725274725275, 
        Accuracy on 2 = 0.7032967032967034, 
        Accuracy on 3 = 0.8021978021978022, 
        Kappa=0.37301714468835123
    Distance mode = manha_d, K = 14, 
        Accuracy on 1 = 0.5384615384615384, 
        Accuracy on 2 = 0.7362637362637363, 
        Accuracy on 3 = 0.9010989010989011, 
        Kappa=0.38503620273531775
    Distance mode = e_d, K = 14, 
        Accuracy on 1 = 0.5384615384615384, 
        Accuracy on 2 = 0.7252747252747253, 
        Accuracy on 3 = 0.9010989010989011, 
        Kappa=0.388088376560999
    Distance mode = se_d, K = 5, 
        Accuracy on 1 = 0.5274725274725275, 
        Accuracy on 2 = 0.7362637362637363, 
        Accuracy on 3 = 0.8021978021978022, 
        Kappa=0.380952380952381
    Distance mode = jsd_d, K = 2, 
        Accuracy on 1 = 0.4945054945054945, 
        Accuracy on 2 = 0.6373626373626373, 
        Accuracy on 3 = 0.6593406593406593, 
        Kappa=0.33555555555555555
"""
# -------------------------------------- Standard routine test of KNN --------------------------------------

# -------------------------------------- Assess the best KNN model --------------------------------------
# 1-Import dataset (Training, validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
tr_va_dataset, test_dataset = get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)
# Tr+Va == 538 samples
tr_va_label_list, tr_va_data_list = split_labeled_dataset_list(tr_va_dataset)
# Te == 91 samples
test_label_list, test_data_list = split_labeled_dataset_list(test_dataset)
label_list = [2,4,5,6,7,8,9]

# ---------------- Assess the best KNN model on Tr+Va dataset -----------------
# Create KNN Model
trVa_knn = KNN(K = 14,
               unlabeled_dataset_list = tr_va_data_list,
               labeled_dataset_list = tr_va_dataset,
               distance_mode = 'e_d',
               label_list = label_list)
trVa_sample_label_prob_dict_list = trVa_knn.classify()
finalTrVaAcc = cal_accuracy(trVa_sample_label_prob_dict_list, tr_va_label_list)
finalTrVaKappa = cal_kappa(trVa_sample_label_prob_dict_list, tr_va_label_list)
# print('Final-TrVa-K={0}-D={1}: Acc={2}, Kappa={3}'.format(14, 'e_d', finalTrVaAcc, finalTrVaKappa))
# Final-TrVa-K=14-D=e_d: Acc=0.570631970260223, Kappa=0.43823312705953615

# --------------------- Output the results into a txt ---------------------
# KNN_TrVa_txt_fn = get_date_prefix()+'KNN_trVa_classify_res.txt'
# KNN_TrVa_f = open(KNN_TrVa_txt_fn, 'a+')
# KNN_TrVa_f_header = 'True_Label,Predict_Label\n'
# KNN_TrVa_f.write(KNN_TrVa_f_header)
# for true_label, label_prob_dict in zip(tr_va_label_list, trVa_sample_label_prob_dict_list):
#     max_prob_k_v_pair = max(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1])
#     pre_label = max_prob_k_v_pair[0]
#     line_str = str(true_label)+','+str(pre_label)+'\n'
#     KNN_TrVa_f.write(line_str)
#
# KNN_TrVa_f.close()
# --------------------- Output the results into a txt ---------------------
# Res: dpfc_src\ml_sl\knn\2021_11_07_KNN_trVa_classify_res.txt
# ---------------- Assess the best KNN model on Tr+Va dataset -----------------

# ---------------- Assess the best KNN model on Test dataset -----------------
# Create KNN Model
finalTe_knn = KNN(K = 14,
                unlabeled_dataset_list = test_data_list,
                labeled_dataset_list = tr_va_dataset,
                distance_mode = 'e_d',
                label_list = label_list)
finalTe_sample_label_prob_dict_list = finalTe_knn.classify()
finalTeAcc = cal_accuracy(finalTe_sample_label_prob_dict_list, test_label_list)
finalTeKappa = cal_kappa(finalTe_sample_label_prob_dict_list, test_label_list)
# print('Final-Te-K={0}-D={1}: Acc={2}, Kappa={3}'.format(14, 'e_d', finalTeAcc, finalTeKappa))
# Final-Te-K=14-D=e_d: Acc=0.5384615384615384, Kappa=0.388088376560999

# --------------------- Output the results into a txt ---------------------
# KNN_finalTe_txt_fn = get_date_prefix()+'KNN_finalTe_classify_res.txt'
# KNN_finalTe_f = open(KNN_finalTe_txt_fn, 'a+')
# KNN_finalTe_f_header = 'True_Label,Predict_Label\n'
# KNN_finalTe_f.write(KNN_finalTe_f_header)
# for true_label, label_prob_dict in zip(test_label_list, finalTe_sample_label_prob_dict_list):
#     max_prob_k_v_pair = max(label_prob_dict.items(), key=lambda k_v_pair: k_v_pair[1])
#     pre_label = max_prob_k_v_pair[0]
#     line_str = str(true_label)+','+str(pre_label)+'\n'
#     KNN_finalTe_f.write(line_str)
#
# KNN_finalTe_f.close()
# --------------------- Output the results into a txt ---------------------
# Res: dpfc_src\ml_sl\knn\2021_11_07_KNN_finalTe_classify_res.txt
# ---------------- Assess the best KNN model on Test dataset -----------------
# -------------------------------------- Assess the best KNN model --------------------------------------
