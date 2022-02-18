import copy
import pickle

from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from ml_sl.rf.dt_0 import Node, save_node, load_node
from ml_sl.ml_data_wrapper import pack_list_2_list, single_point_list_2_list, reform_labeled_dataset_list
from utils.file_utils.filename_utils import get_date_prefix
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3

# --------------------- 1-No Pruning ---------------------
def dt_no_pruning(training_dataset, validation_dataset, test_dataset, label_list=[2,4,5,6,7,8,9]):
    tr_va_dataset = training_dataset + validation_dataset
    reformed_tr_va_dataset = reform_labeled_dataset_list(tr_va_dataset)

    # 1.1- Use [training+validation]-dataset to train a Decision Tree (DT), DT0,
    dt = Node(reformed_labeled_dataset_list = reformed_tr_va_dataset, level = 0)
    dt.create_child_node()

    # 1.2- save DT
    dt_file_name = get_date_prefix() + 'dt_no_pruning_pickle.file'
    save_node(node=dt, file_name=dt_file_name)

    # 1.3- Test the performance(accuracy, kappa) of DT0 on test-dataset
    test_label_list = [t[0] for t in test_dataset]
    sample_label_prob_dict_list = []

    empty_sample_label_prob_dict = {}
    for label in label_list:
        empty_sample_label_prob_dict[label] = 0.0

    for t_d in test_dataset:
        t_d = single_point_list_2_list(t_d[1])
        pre = dt.classify(unlabeled_data_list=t_d)
        sample_label_prob_dict = copy.deepcopy(empty_sample_label_prob_dict)
        sample_label_prob_dict[pre] += 1
        sample_label_prob_dict_list.append(sample_label_prob_dict)

    acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)
    return acc, kappa

# if __name__ == '__main__':
#     training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')

    # Running condition-1
    # acc, kappa = dt_no_pruning(training_dataset, validation_dataset, test_dataset)
    # print('Accuracy: {0}, Kappa: {1}'.format(acc, kappa))

    # Running condition-2
    # acc, kappa = dt_no_pruning(training_dataset, validation_dataset=[], test_dataset=validation_dataset)
    # print('Accuracy: {0}, Kappa: {1}'.format(acc, kappa))
"""
Running condition-1
    Train on [Training+validation]-dataset
    Test on test-dataset
    1-Accuracy: 0.45054945054945056, Kappa: 0.3173293323330833
    2-Accuracy: 0.45054945054945056, Kappa: 0.3173293323330833
Running condition-2
    Train on [Training]-dataset
    Test on validation-dataset
    1-Accuracy: 0.5319148936170213, Kappa: 0.42762247439800716
    2-Accuracy: 0.5319148936170213, Kappa: 0.42762247439800716
"""

def load_dt_no_pruning(training_dataset, validation_dataset, test_dataset, label_list=[2,4,5,6,7,8,9]):
    dt = load_node(file_name='2020_04_11_dt_no_pruning_pickle.file', file_path='dt_res')

    # 1.3- Test the performance(accuracy, kappa) of DT0 on test-dataset
    test_label_list = [t[0] for t in test_dataset]
    sample_label_prob_dict_list = []

    empty_sample_label_prob_dict = {}
    for label in label_list:
        empty_sample_label_prob_dict[label] = 0.0

    for t_d in test_dataset:
        t_d = single_point_list_2_list(t_d[1])
        pre = dt.classify(unlabeled_data_list=t_d)
        sample_label_prob_dict = copy.deepcopy(empty_sample_label_prob_dict)
        sample_label_prob_dict[pre] += 1
        sample_label_prob_dict_list.append(sample_label_prob_dict)

    acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
    acc_on_2 = cal_accuracy_on_2(sample_label_prob_dict_list, test_label_list)
    acc_on_3 = cal_accuracy_on_3(sample_label_prob_dict_list, test_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

    print('Decision Tree with no pruning: Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(
        acc, acc_on_2, acc_on_3, kappa))

# training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
# load_dt_no_pruning(training_dataset, validation_dataset, test_dataset, label_list=[2,4,5,6,7,8,9])
# Decision Tree with no pruning: Accuracy on 1 = 0.4945054945054945, Accuracy on 2 = 0.5164835164835165,
#                                Accuracy on 3 = 0.6923076923076923, Kappa=0.3706209592542475
# --------------------- 1-No Pruning ---------------------

# --------------------- 2-Pruning ---------------------
def dt_pruning(training_dataset, validation_dataset, test_dataset, label_list=[2,4,5,6,7,8,9]):
    reformed_tr_dataset_list = reform_labeled_dataset_list(training_dataset)

    # 2.1- Use training-dataset to train a Decision Tree, DT
    dt = Node(reformed_labeled_dataset_list=reformed_tr_dataset_list, level=0)
    dt.create_child_node()

    # 2.2- Use validation-dataset to prune DT1
    dt.post_pruning_1(reform_labeled_dataset_list(validation_dataset))

    # 2.3- save model
    dt_file_name = get_date_prefix() + 'dt_pruning_pickle.file'
    save_node(node=dt, file_name=dt_file_name)

    # 2.4- Test the performance(accuracy, kappa) of DT on test-dataset
    test_label_list = [t[0] for t in test_dataset]
    sample_label_prob_dict_list = []
    empty_sample_label_prob_dict = {}
    for label in label_list:
        empty_sample_label_prob_dict[label] = 0.0

    for t_d in test_dataset:
        t_d = single_point_list_2_list(t_d[1])
        pre = dt.classify(unlabeled_data_list=t_d)
        sample_label_prob_dict = copy.deepcopy(empty_sample_label_prob_dict)
        sample_label_prob_dict[pre] += 1
        sample_label_prob_dict_list.append(sample_label_prob_dict)

    acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)
    return acc, kappa

# if __name__ == '__main__':
#     training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
#     acc, kappa = dt_pruning(training_dataset, validation_dataset, test_dataset, label_list=[2, 4, 5, 6, 7, 8, 9])
#     print('Accuracy: {0}, Kappa: {1}'.format(acc, kappa))
"""
    1- Accuracy: 0.4835164835164835, Kappa: 0.3591549295774648
    2- Accuracy: 0.4835164835164835, Kappa: 0.3591549295774648
"""

def load_dt_pruning(test_dataset, label_list=[2,4,5,6,7,8,9]):
    dt = load_node(file_name='2020_04_11_dt_pruning_pickle_1.file', file_path='dt_res')

    # 2.4- Test the performance(accuracy, kappa) of DT on test-dataset
    test_label_list = [t[0] for t in test_dataset]
    sample_label_prob_dict_list = []
    empty_sample_label_prob_dict = {}
    for label in label_list:
        empty_sample_label_prob_dict[label] = 0.0

    for t_d in test_dataset:
        t_d = single_point_list_2_list(t_d[1])
        pre = dt.classify(unlabeled_data_list=t_d)
        sample_label_prob_dict = copy.deepcopy(empty_sample_label_prob_dict)
        sample_label_prob_dict[pre] += 1
        sample_label_prob_dict_list.append(sample_label_prob_dict)

    acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
    acc_on_2 = cal_accuracy_on_2(sample_label_prob_dict_list, test_label_list)
    acc_on_3 = cal_accuracy_on_3(sample_label_prob_dict_list, test_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

    print('Decision Tree with pruning: Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(
            acc, acc_on_2, acc_on_3, kappa))
# training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
# load_dt_pruning(test_dataset, label_list=[2,4,5,6,7,8,9])
# Decision Tree with pruning: Accuracy on 1 = 0.4835164835164835, Accuracy on 2 = 0.5054945054945055,
#                             Accuracy on 3 = 0.6703296703296703, Kappa = 0.3591549295774648
# --------------------- 2-Pruning ---------------------