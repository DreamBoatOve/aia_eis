import numpy as np
import matplotlib.pyplot as plt
from sklearn.tree import DecisionTreeClassifier

from ml_sl.test.load_testSet import read_testSet, read_testSet_with_label
from ml_sl.rf.dt_0 import Node
from ml_sl.ml_data_wrapper import pack_list_2_list, single_point_list_2_list, reform_labeled_dataset_list
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset

#----------------------- Comparsion on testSet (Binary Classification) -----------------------
def SK_learn_DT_on_testSet(data_i_list, data_j_list):
    # 2-Convert data structure
    label_list = [0 for i in range(len(data_i_list))] + [1 for i in range(len(data_j_list))]
    data_i_list.extend(data_j_list)
    data_list = [[d[0][0], d[0][1]] for d in data_i_list]

    label_arr = np.array(label_list)
    data_arr = np.array(data_list)

    # 3-Train Decision Tree(DT)
    sk_dt = DecisionTreeClassifier(criterion="entropy")
    sk_dt.fit(X = data_arr, y = label_arr)

    # 4-Use DT to make prediction
    x0_min, x0_max = data_arr[:, 0].min(), data_arr[:, 0].max()
    x1_min, x1_max = data_arr[:, 1].min(), data_arr[:, 1].max()
    step_size = 0.02
    xx, yy = np.meshgrid(np.arange(x0_min - 1, x0_max + 1, step_size), np.arange(x1_min - 1 , x1_max + 1, step_size))
    Z = sk_dt.predict(np.c_[xx.ravel(), yy.ravel()])

    # 5-Draw plot / Calculate Accuracy
    Z = Z.reshape(xx.shape)
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(data_arr[:, 0], data_arr[:, 1], c=label_arr, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x0')
    plt.ylabel('x1')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.title('SK-Learn-DecisionTree')
    plt.show()

def my_dt_on_testSet():
    # 2-Convert data structure
    labeled_data_i_list, labeled_data_j_list = read_testSet_with_label(filepath='../../../test')
    label_list = [0 for i in range(len(labeled_data_i_list))] + [1 for i in range(len(labeled_data_j_list))]
    reformed_data_list = reform_labeled_dataset_list(labeled_data_i_list + labeled_data_j_list)

    # 3-Train Decision Tree (DT)
    dt = Node(reformed_labeled_dataset_list = reformed_data_list, level = 0)
    dt.create_child_node()

    # 4-Use DT to make prediction
    d0_list = [d[1][0] for d in reformed_data_list]
    d1_list = [d[1][1] for d in reformed_data_list]
    x0_min, x0_max = min(d0_list), max(d0_list)
    x1_min, x1_max = min(d1_list), max(d1_list)
    step_size = 0.02
    xx0, xx1 = np.meshgrid(np.arange(x0_min - 1, x0_max + 1, step_size), np.arange(x1_min - 1, x1_max + 1, step_size))

    Z_list = []
    for x0, x1 in zip(xx0.flatten().tolist(), xx1.flatten().tolist()):
        Z_list.append(dt.classify([x0, x1]))
    Z_arr = np.array(Z_list).reshape(xx0.shape)

    # 5-Draw plot / Calculate Accuracy
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx0, xx1, Z_arr, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(d0_list, d1_list, c=label_list, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x0')
    plt.ylabel('x1')

    plt.xlim(xx0.min()-1, xx0.max()+1)
    plt.ylim(xx1.min()-1, xx1.max()+1)
    plt.xticks(())
    plt.yticks(())

    plt.title('My-DecisionTree')
    plt.show()

# if __name__ == '__main__':
#     # 1-Load data
#     data_i_list, data_j_list = read_testSet(filepath='../../../test')
#     SK_learn_DT_on_testSet(data_i_list, data_j_list)
#     my_dt_on_testSet()
#----------------------- Comparsion on testSet (Binary Classification) -----------------------

#----------------------- Comparsion on EIS (Multi Classification) -----------------------
def SK_learn_DT_on_EIS(training_dataset, validation_dataset):
    """
    (training/validation)_dataset
        list[
                [label (int), [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)]]
            ]
    """
    tr_label_list = [tr[0] for tr in training_dataset]
    va_label_list = [va[0] for va in validation_dataset]

    tr_data_list = [single_point_list_2_list(tr[1]) for tr in training_dataset]
    va_data_list = [single_point_list_2_list(va[1]) for va in validation_dataset]

    tr_label_arr = np.array(tr_label_list)
    tr_data_arr = np.array(tr_data_list)

    sk_dt = DecisionTreeClassifier(criterion="entropy", max_features=1.0)
    sk_dt.fit(X = tr_data_arr, y = tr_label_arr)

    # Accuracy on validation-dataset
    va_data_arr = np.array(va_data_list)
    va_pre_arr = sk_dt.predict(X = va_data_arr)

    correct_count = 0
    for v_label, v_pre in zip(va_label_list, va_pre_arr):
        if v_label - v_pre == 0.0:
            correct_count += 1
    acc = correct_count / len(va_data_list)
    return acc

def my_DT_on_EIS(training_dataset, validation_dataset, post_pruning_flag=False):
    reformed_tr_dataset_list = reform_labeled_dataset_list(training_dataset)

    dt = Node(reformed_labeled_dataset_list = reformed_tr_dataset_list, level = 0)
    dt.create_child_node()

    if post_pruning_flag:
        dt.post_pruning_1(reform_labeled_dataset_list(validation_dataset))

    pre_list = []
    va_label_list = [va[0] for va in validation_dataset]
    for va_list in validation_dataset:
        va_list = single_point_list_2_list(va_list[1])
        pre = dt.classify(unlabeled_data_list = va_list)
        pre_list.append(pre)

    # Accuracy
    acc = sum([1 for p, va in zip(pre_list, va_label_list) if p - va == 0.0]) / len(va_label_list)
    return acc

# if __name__ == '__main__':
    # 1-Load two kinds of EIS data according to ecm_num_pair
    # training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../../../datasets/ml_datasets/normed')

    # sk_acc = SK_learn_DT_on_EIS(training_dataset, validation_dataset)
    # print(sk_acc)

    # my_acc = my_DT_on_EIS(training_dataset, validation_dataset, post_pruning_flag=True)
    # print(my_acc)

    """
    SK-Learn-DecisionTreeClassifier
        Accuracy on validation-dataset (max_features = 'auto' = sqrt(n_features))
            1- 0.4787234042553192
            2- 0.5
            3- 0.46808510638297873
            4- 0.4787234042553192
            5- 0.5106382978723404
        Accuracy on validation-dataset (max_features = float 1.0 = n_features * 1.0)
            1- 0.5106382978723404
            2- 0.4787234042553192
    My Decision Tree
        No post-pruning
            1- 0.5319148936170213, the result does not change and is a constant for a certain (training,validation)-datasets
        Post-pruning
            1- 0.5319148936170213, the result also does not change and is a constant for a certain (training,validation)-datasets
    """
#----------------------- Comparsion on EIS (Multi Classification) -----------------------