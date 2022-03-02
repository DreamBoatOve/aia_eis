from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from ml_sl.nbc.nbc_0 import NBC
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3

# -------------------------------------- Standard routine test of NBC --------------------------------------
label_list = [2,4,5,6,7,8,9]
# 1- Load dataset (Training+validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
tr_dataset, va_dataset, te_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
tr_va_dataset, test_dataset = get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)

tr_label_list, tr_data_list = split_labeled_dataset_list(tr_dataset)
va_label_list, va_data_list = split_labeled_dataset_list(va_dataset)
tr_va_label_list, tr_va_data_list = split_labeled_dataset_list(tr_va_dataset)
te_label_list, te_data_list = split_labeled_dataset_list(te_dataset)

nbc = NBC(unlabeled_dataset_list=tr_va_data_list,
          labeled_dataset_list=tr_va_dataset,
          label_list=label_list)
nbcTrVaSample_label_prob_dict_list = nbc.classify()

# 2- Build a Naive Bayesian Classifer with training_validation-dataset
# Separate the test_dataset into label and data
# test_label_list, test_dataset_list = split_labeled_dataset_list(test_dataset)
# nbc = NBC(unlabeled_dataset_list=test_dataset_list,
#           labeled_dataset_list=tr_va_dataset,
#           label_list=label_list)
# 2.1 Use NBC to make classification of test_dataset
# nbc_test_sample_label_prob_dict_list = nbc.classify()

# 3- Assess the performance of NBC on tr_va_dataset and test_dataset, respectively
# 3.1- Assess the performance of NBC on test_dataset
# nbc_test_acc = cal_accuracy(nbc_test_sample_label_prob_dict_list, test_label_list)
# nbc_test_kappa = cal_kappa(nbc_test_sample_label_prob_dict_list, test_label_list)
# print('NBC on Test dataset: Accuracy =',nbc_test_acc, 'Kappa =', nbc_test_kappa)
# NBC on Test dataset: Accuracy = 0.4945054945054945 Kappa = 0.3751306165099269

# 3.2- Assess the performance of NBC on tr_va_dataset
# tr_va_label_list, tr_va_dataset_list = split_labeled_dataset_list(tr_va_dataset)
# nbc.unlabeled_dataset_list = tr_va_dataset_list
# nbc_tr_va_sample_label_prob_dict_list = nbc.classify()
# nbc_tr_va_acc = cal_accuracy(nbc_tr_va_sample_label_prob_dict_list, tr_va_label_list)
# nbc_tr_va_kappa = cal_kappa(nbc_tr_va_sample_label_prob_dict_list, tr_va_label_list)
# print('NBC on Training+Validation dataset: Accuracy =',nbc_tr_va_acc, 'Kappa =', nbc_tr_va_kappa)
# NBC on Training+Validation dataset: Accuracy = 0.5092936802973977 Kappa = 0.39062982666895485

# 4- Access the accuracy of the first one, two, and three predictions
# nbc_test_acc_on_2 = cal_accuracy_on_2(nbc_test_sample_label_prob_dict_list, test_label_list)
# nbc_test_acc_on_3 = cal_accuracy_on_3(nbc_test_sample_label_prob_dict_list, test_label_list)
# print('Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(
#     nbc_test_acc, nbc_test_acc_on_2, nbc_test_acc_on_3, nbc_test_kappa))
"""
Results:
    NBC on Test dataset: Accuracy = 0.4945054945054945 Kappa = 0.3751306165099269
    Accuracy on 1 = 0.4945054945054945, Accuracy on 2 = 0.7032967032967034, Accuracy on 3 = 0.8681318681318682, Kappa=0.3751306165099269
"""
# -------------------------------------- Standard routine test of NBC --------------------------------------