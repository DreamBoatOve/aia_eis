import os
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from ml_sl.svm.multiclass_svm_0 import Multiclass_SVM, load_Multiclass_SVM_model

tr_va_dataset, test_dataset = get_TV_T_dataset(file_path='../../datasets/ml_datasets/normed')
def svm_final_OvR_OvO(svm_final_ovr_model_path, svm_final_ovo_model_path):
    global tr_va_dataset, test_dataset
    TV_dataset = tr_va_dataset
    te_data_list = []
    te_label_list = []
    for te in test_dataset:
        te_label_list.append(te[0])
        te_data_list.append(te[1])

    # svm_para_dict, kernel_para_dict can be set as any format but it has some content, in case of causing error during the initialization of multi_SVM
    svm_para_dict = {'C': 0.1, 'max_iter': 5000}
    kernel_para_dict = {'type':'linear', 'paras':None}
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict, \
                               unlabeled_dataset_list=te_data_list, labeled_dataset_list=TV_dataset, \
                               label_list=[2, 4, 5, 6, 7, 8, 9])
    sample_label_prob_dict_list = multi_svm.classify_ovr_ovo(svm_ovr_model_pickle_name=svm_final_ovr_model_path,\
                                                             svm_ovo_model_pickle_name=svm_final_ovo_model_path)
    acc = cal_kappa(sample_label_prob_dict_list, te_label_list)
    acc_on_2 = cal_accuracy_on_2(sample_label_prob_dict_list, te_label_list)
    acc_on_3 = cal_accuracy_on_3(sample_label_prob_dict_list, te_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, te_label_list)

    print('Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(acc, acc_on_2, acc_on_3, kappa))

# if __name__ == '__main__':
#     ovr_file_name = '2020_07_25_svm_ovr_final_poly_C=1_iter=3000_q=1_pickle_1.file'
#     ovr_path = 'ovr_models/final'
#     ovo_file_name = '2020_06_26_svm_ovo_linear_final_C=0.001_iter=5000_3_pickle.file'
#     ovo_path = 'ovo_models/final'
#     svm_final_ovr_model_path = os.path.join(ovr_path, ovr_file_name)
#     svm_final_ovo_model_path = os.path.join(ovo_path, ovo_file_name)
#     svm_final_OvR_OvO(svm_final_ovr_model_path, svm_final_ovo_model_path)
# Accuracy on 1 = 0.23683327742368335, Accuracy on 2 = 0.5934065934065934,
# Accuracy on 3 = 0.7032967032967034, Kappa=0.23683327742368335
# OvR再经OvO改进后，其Acc并无显著改善，说明OvR对Test-dataset中的样本大都做出一个标签的投票，OvO并没有被充分利用