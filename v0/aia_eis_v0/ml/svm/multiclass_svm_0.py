import os
import sys
import random
import numpy as np
import pickle

from ml_sl.svm.svm_0 import SVM
from ml_sl.ml_data_wrapper import single_point_list_2_list

class Multiclass_SVM:
    """
    Multiclass_SVM
        Support_Vector_Machine for multi classification
        one vs one
        SMO (Sequential minimal optimization)
    """
    def __init__(self, svm_para_dict, kernel_para_dict, unlabeled_dataset_list, labeled_dataset_list, label_list):
        """
        :param
            svm_para_dict
                {
                    'C':
                        C, float
                        balance the W.T * W / 2 and Violation
                        惩罚参数, 惩罚因子
                        支持向量的权重与越界数据之间的平衡因子
                        理解为调节优化方向中两个指标（间隔大小，分类准确度）偏好的权重
                    'tol':
                        tol, float, default 0.01
                        the tolerance of the error, prediction_f - real_target_y
                        计算数据点与超平面之间的距离时，允许有微小的误差tol
                        tol 在Platt的SMO论文中建议取值10^(-2) ~ 10^(-3)，取值越小，算法收敛越慢
                    'max_iter':
                        int, maximum iteration of SVM
                }
            kernel_paras_dict
                {
                    'type'(kernel type):
                        'linear'
                        'poly'
                        'rbf'
                    'paras':
                        if poly:
                            (refer 203_handout.pdf, page 7)
                            Kn(X, X') = (ζ + r * X.T * X')^n
                            Kn = Φn * Φn'
                                K2(X, X') = (1 + r * X.T * X')^2
                                Φ2(X)    = (1, sqrt(2r) * x1, sqrt(2r) * x2, ... , sqrt(2r) * xd, r * x1^2, r * x2^2, ... , r * xn^2)
                            [power (int), constant (ζ， float), Quadratic coefficient (r, float)]
                            power
                                多项式的指数，几次方的多项式
                                K2 is commonly used
                            constant,
                                常数项
                            qua_coe
                                coe, short for coefficient(系数)
                                X.T * X前的系数
                        if rbf:
                            参照 台湾 林轩田 203_handout, page 12-13/22
                            exp(- sigma * ||X1 - X2||^2 )
                            sigma (float)
                }
            unlabeled_dataset_list:
                a group of unlabeled samples
                    [
                        [points list]
                        [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                        [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                        [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                        ...
                    ]
            labeled_dataset_list:
                [
                    [label number, points list]
                    [1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    [3, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    [4, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    ...
                ]
            label_list
                record the types of ECMs, this parameter should be manually set
                [0,1,2,3, ... , 7, 8, 9]
        """
        self.C, self.max_iter = svm_para_dict['C'], svm_para_dict['max_iter']
        self.tol = 0.01
        self.kernel_para_dict = kernel_para_dict
        self.unlabeled_dataset_list = unlabeled_dataset_list
        self.labeled_dataset_list = labeled_dataset_list
        self.label_list = label_list

    def create_svm_ovo_classifer(self, svm_model_pickle_name, store_model=True):
        label_sorted_data_dcit = {}
        for label in self.label_list:
            label_sorted_data_dcit[label] = [data for data in self.labeled_dataset_list if data[0] == label]

        # 每次抽取不同的两类样本，构建分类器
        svm_classifer_dict = {}
        for index, label_i in enumerate(self.label_list):
            for label_j in self.label_list[index + 1:]:
                label_i_data_list = label_sorted_data_dcit[label_i]
                label_j_data_list = label_sorted_data_dcit[label_j]
                svm = SVM(labeled_dataset_list=label_i_data_list+label_j_data_list,\
                          kernel_paras_dict=self.kernel_para_dict, C=self.C, tol=self.tol, max_iter=self.max_iter)
                svm.smo_1()
                # remove the original training data from SVM to store an smaller model
                svm.labeled_dataset_list = None
                svm_classifer_dict[(label_i, label_j)] = svm
        if store_model:
            with open(svm_model_pickle_name, 'wb') as file:
                pickle.dump(svm_classifer_dict, file)

    def create_svm_ovr_classifer(self, svm_model_pickle_name, store_model=True):
        # 每次抽取一类样本和剩余的样本，构建分类器
        svm_classifer_dict = {}
        for label in self.label_list:
            # 当前类的标签为1，其余数据标签为-1
            label_i_data_list = [] # 标签为-1
            label_j_data_list = [] # 标签为1
            # 在SVM class中会将标签数值大的自动转为标签1，标签数值小的自动转为标签-1
            for data in self.labeled_dataset_list:
                if data[0] == label:
                    label_j_data_list.append(data)
                else:
                    data[0] = -1
                    label_i_data_list.append(data)

            svm = SVM(labeled_dataset_list=label_i_data_list + label_j_data_list, kernel_paras_dict=self.kernel_para_dict,\
                      C = self.C, tol = self.tol, max_iter = self.max_iter)
            svm.smo_1()
            # remove the original training data from SVM to store an smaller model
            svm.labeled_dataset_list = None
            svm_classifer_dict[label] = svm
        if store_model:
            with open(svm_model_pickle_name, 'wb') as file:
                pickle.dump(svm_classifer_dict, file)

    def classify_ovo(self, svm_ovo_model_pickle_name=None):
        with open(svm_ovo_model_pickle_name, 'rb') as file:
            svm_ovo_classifer_dict = pickle.load(file)

        sample_label_prob_dict_list = []
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0
            x_arr = np.array(single_point_list_2_list(unlabeled_data_list))
            for key, value in svm_ovo_classifer_dict.items():
                label_i, label_j = key
                svm = value
                # pre_label is -1 or 1, not 2,3,4,... 9, ...
                pre_label = svm.classify(x_arr)
                if pre_label == -1:
                    pre_label = label_i
                elif pre_label == 1:
                    pre_label = label_j
                else:
                    print('SVM prediction={0} is not -1 or 1, check it'.format(pre_label))
                    sys.exit(0)
                label_vote_dict[pre_label] += 1

            count_num = len(svm_ovo_classifer_dict.keys())
            label_prob_dict = {}
            for key, value in label_vote_dict.items():
                label_prob_dict[key] = value / count_num
            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

    def classify_ovr(self, svm_ovr_model_pickle_name=None):
        with open(svm_ovr_model_pickle_name, 'rb') as file:
            svm_ovr_classifer_dict = pickle.load(file)

        sample_label_prob_dict_list = []
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            label_prob_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0
                label_prob_dict[label] = 0

            x_arr = np.array(single_point_list_2_list(unlabeled_data_list))
            for label, value in svm_ovr_classifer_dict.items():
                svm = value
                # pre_label is -1 or 1, not 2,3,4,... 9, ...
                pre_label = svm.classify(x_arr)
                if pre_label == 1:
                    pre_label = label
                    label_vote_dict[pre_label] += 1

            # 预测可能出现的三种情况：1-只有一个标签获得投票；2-有两个及以上的标签获得投票；3-没有标签被投票
            vote_count = sum(label_vote_dict.values())
            # 1-只有一个标签获得投票
            if vote_count == 1:
                label_prob_dict = label_vote_dict
            # 2-有两个及以上的标签获得投票
            elif vote_count >= 2:
                voted_label_list = [k for k, v in label_vote_dict.items() if v == 1 ]
                random_voted_label = random.choice(voted_label_list)
                label_prob_dict[random_voted_label] += 1
            # 3-没有标签被投票
            elif vote_count == 0:
                random_voted_label = random.choice(self.label_list)
                label_prob_dict[random_voted_label] += 1
            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

    def classify_ovr_ovo(self, svm_ovr_model_pickle_name=None, svm_ovo_model_pickle_name=None):
        with open(svm_ovr_model_pickle_name, 'rb') as file0:
            svm_ovr_classifer_dict = pickle.load(file0)
        with open(svm_ovo_model_pickle_name, 'rb') as file1:
            svm_ovo_classifer_dict = pickle.load(file1)

        sample_label_prob_dict_list = []
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            label_prob_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0
                label_prob_dict[label] = 0

            x_arr = np.array(single_point_list_2_list(unlabeled_data_list))
            for label, value in svm_ovr_classifer_dict.items():
                svm = value
                # pre_label is -1 or 1, not 2,3,4,... 9, ...
                pre_label = svm.classify(x_arr)
                if pre_label == 1:
                    pre_label = label
                    label_vote_dict[pre_label] += 1

            # 预测可能出现的三种情况：1-只有一个标签获得投票；2-有两个及以上的标签获得投票；3-没有标签被投票
            vote_count = sum(label_vote_dict.values())
            # 1- vote_count = 1： 只有一个标签获得投票,OvR的决策结果作为最终结果
            if vote_count == 1:
                label_prob_dict = label_vote_dict

            # 2- vote_count = 2 ~ 8: 有两个及以上的标签获得投票，如获得投票的标签为2，3，4,
            # 则从OvO分类其中抽出(2,3),(2,4) and (3,4)这三个分类器，分别再做一次分类
            elif (vote_count >= 2) and (vote_count < 9):
                voted_label_list = [k for k, v in label_vote_dict.items() if v == 1]
                # sort list in ascending order
                voted_label_list.sort()
                for index, label_i in enumerate(voted_label_list):
                    for label_j in voted_label_list[index + 1 : ]:
                        svm_ovo_classifer = svm_ovo_classifer_dict[(label_i, label_j)]
                        # pre_label is -1 or 1, not 2,3,4,... 9, ...
                        pre_label = svm_ovo_classifer.classify(x_arr)
                        if pre_label == -1:
                            pre_label = label_i
                        elif pre_label == 1:
                            pre_label = label_j
                        label_vote_dict[pre_label] += 1

                count_sum = sum(label_vote_dict.values())
                for k, v in label_vote_dict.items():
                    label_prob_dict[k] = v / count_sum

            # 3- vote_count = 0 or 9 : 没有标签被投票或者所有标签都被投票，重新用OvO重新进行预测
            elif (vote_count == 0) or (vote_count == 9):
                for index, label_i in enumerate(self.label_list):
                    for label_j in self.label_list[index + 1 :]:
                        svm_ovo_classifer = svm_ovo_classifer_dict[(label_i, label_j)]
                        # pre_label is -1 or 1, not 2,3,4,... 9, ...
                        pre_label = svm_ovo_classifer.classify(x_arr)
                        if pre_label == -1:
                            pre_label = label_i
                        elif pre_label == 1:
                            pre_label = label_j
                        label_vote_dict[pre_label] += 1

                count_sum = sum(label_vote_dict.values())
                for k, v in label_vote_dict.items():
                    label_prob_dict[k] = v / count_sum

            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

def load_Multiclass_SVM_model(file_name, path):
    file_path = os.path.join(path, file_name)
    with open(file_path, 'rb') as file:
        svm_classifer_dict = pickle.load(file)
    return svm_classifer_dict