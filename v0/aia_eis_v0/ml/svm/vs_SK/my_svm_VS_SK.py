import numpy as np
import matplotlib.pyplot as plt
from sklearn import svm

from ml_sl.test.load_testSet import read_testSet, read_testSet_with_label
from ml_sl.svm.svm_0 import SVM

# 1-Check SK-Learn-SVM
# 2-Test SK-Learn-SVM on testSet (Linear/Poly/RBF)
def SK_SVM_on_testSet(data_i_list, data_j_list, kernel):
    label_list = [0 for i in range(len(data_i_list))] + [1 for i in range(len(data_j_list))]

    data_i_list.extend(data_j_list)
    data_list = [[d[0][0], d[0][1]] for d in data_i_list]

    sk_svm = svm.SVC(kernel=kernel, gamma=10)

    # List ==> np.array
    x_arr = np.array(data_list)
    y_arr = np.array(label_list)
    sk_svm.fit(X=x_arr, y=y_arr)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x0_min, x0_max], [x1_min, x1_max].
    x0_min, x0_max = x_arr[:, 0].min(), x_arr[:, 0].max()
    x1_min, x1_max = x_arr[:, 1].min(), x_arr[:, 1].max()

    # step size in the mesh
    step_size = 0.02
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, step_size), np.arange(x1_min, x1_max, step_size))
    Z = sk_svm.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(1, figsize=(4, 3))
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x_arr[:, 0], x_arr[:, 1], c=y_arr, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.title('SK-Learn-SVM with {0} kernel'.format(kernel))
    plt.show()

# 3-Test my SVM on testSet
def my_SVM_on_testSet(labeled_data_i_list, labeled_data_j_list, kernel_paras_dict):
    label_list = [-1 for i in range(len(labeled_data_i_list))] + [1 for i in range(len(labeled_data_j_list))]
    labeled_data_list = []
    for data in labeled_data_i_list + labeled_data_j_list:
        if data[0] == 0:
            data[0] = -1
        labeled_data_list.append(data)

    d0_list = [d[1][0][0] for d in labeled_data_list]
    d1_list = [d[1][0][1] for d in labeled_data_list]
    x0_min, x0_max = min(d0_list), max(d0_list)
    x1_min, x1_max = min(d1_list), max(d1_list)
    step_size = 0.02
    xx0, xx1 = np.meshgrid(np.arange(x0_min - 1, x0_max + 1, step_size), np.arange(x1_min - 1, x1_max + 1, step_size))
    # unlabeled_data_list = [[(x0, x1)] for x0, x1 in zip(xx0.flatten().tolist(), xx1.flatten().tolist())]
    a = xx0.flatten().T
    # unlabeled_data_arr = np.concatenate((xx0.flatten().T, xx1.flatten().T), axis=1)
    unlabeled_data_arr = np.vstack((xx0.flatten(), xx1.flatten())).T
    global C, tol, max_iter
    svm = SVM(labeled_dataset_list=labeled_data_list, kernel_paras_dict=kernel_paras_dict, C=C, tol=tol, max_iter=max_iter)
    svm.smo_1()
    pre_list = []
    for data_arr in unlabeled_data_arr:
        pre = svm.classify(data_arr)
        pre_list.append(pre)
    pre_arr = np.array(pre_list).reshape(xx0.shape)

    # 5-Draw plot / Calculate Accuracy
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx0, xx1, pre_arr, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(d0_list, d1_list, c=label_list, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x0')
    plt.ylabel('x1')

    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())
    plt.xticks(())
    plt.yticks(())

    if kernel_paras_dict['type'] == 'linear':
        plt.title('My-SVM with {0} kernel, C={1}, tol={2}, iter={3}'.format(kernel_paras_dict['type'], C, tol, max_iter))
    elif kernel_paras_dict['type'] == 'poly':
        power, constant, qua_coe = kernel_paras_dict['paras']
        plt.title('My-SVM with {0} kernel, C={1}, tol={2}, iter={3}, power={4}, constant={5}, qua_coe={6}'.format(\
                                                kernel_paras_dict['type'], C, tol, max_iter, power, constant, qua_coe))
    elif kernel_paras_dict['type'] == 'rbf':
        plt.title('My-SVM with {0} kernel, C={1}, tol={2}, iter={3}, sigma={4}'.format(\
                                            kernel_paras_dict['type'], C, tol, max_iter, kernel_paras_dict['paras']))
    plt.show()

if __name__ == '__main__':
    # data_i_list, data_j_list = read_testSet(filepath='../../test')
    # kernel = 'linear', 'rbf', 'poly'
    # kernel = 'rbf'
    # SK_SVM_on_testSet(data_i_list, data_j_list, kernel)

    labeled_data_i_list, labeled_data_j_list = read_testSet_with_label(filepath='../../test')
    C = 1
    tol = 0.01
    max_iter = 100
    linear_kernel_paras_dict = {'type':'linear','paras':None}
    # poly_paras_list: [power, constant, qua_coe]
    poly_kernel_paras_dict = {'type':'poly','paras':[2, 1.0, 100]}
    # rbf_para = sigma
    rbf_kernel_paras_dict = {'type':'rbf','paras':1}
    my_SVM_on_testSet(labeled_data_i_list, labeled_data_j_list, kernel_paras_dict=rbf_kernel_paras_dict)