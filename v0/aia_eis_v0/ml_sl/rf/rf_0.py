import copy
import os
import pickle
import random
import numpy as np

from ml_sl.ml_data_wrapper import reform_labeled_dataset_list, pack_list_2_list
from ml_sl.rf.dt_0 import Random_Tree

class RF():
    """
    RF
        Random_forest
        此算法主要参照
            周志华 《机器学习》
                第4章 决策树
                第8章 集成学习-8.3.2 随机森林
            paper <<Random Forest>>
    """
    def __init__(self, unlabeled_dataset_list, labeled_dataset_list, label_list, tree_num=100):
        """
        :param
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
            Tree_num
                Tree Number in the forest
                default as 100
        """
        self.unlabeled_dataset_list = unlabeled_dataset_list
        self.labeled_dataset_list = labeled_dataset_list
        self.label_list = label_list
        self.tree_num = tree_num

    def create_forest(self):
        """
        create forest
            create decision trees
                Attribute selection
                    At each node in a tree, it randomly select K attributes (K = int(log(M, 2) + 1), M is the total number of attributes)
                No Tree pruning
        :return:
        """
        # 训练数据的数量 N
        N = len(self.labeled_dataset_list)
        self.forest = []
        # 1-按照要建立树的数量确定循环的次数（二者相等）
        for t_i in range(self.tree_num):
            # 1.1-从数据集Dataset（数量为N）中随机有放回的抽取N个样本组成一个临时的数据集tmp_dataset
            tmp_labeled_dataset_list = [copy.deepcopy(self.labeled_dataset_list[random.randint(0, N-1)]) for i in range(N)]

            # 1.2-用tmp_dataset训练一颗随机树
            tmp_reform_labeled_dataset_list = reform_labeled_dataset_list(tmp_labeled_dataset_list)
            rt = Random_Tree(tmp_reform_labeled_dataset_list, leaf_label=None)
            rt.create_child_node()
            self.forest.append(rt)

    def classify(self):
        sample_label_prob_dict_list = []
        reformed_unlabeled_dataset_list = pack_list_2_list(self.unlabeled_dataset_list)
        for unlabeled_data_list in reformed_unlabeled_dataset_list:
            label_num_dict = {}
            for label in self.label_list:
                label_num_dict[label] = 0

            for random_tree in self.forest:
                pre_label = random_tree.classify(unlabeled_data_list)
                # KeyError: None
                label_num_dict[pre_label] += 1

            label_proba_dict = {}
            for key, value in label_num_dict.items():
                label_proba_dict[key] = value/self.tree_num
            sample_label_prob_dict_list.append(label_proba_dict)
        return sample_label_prob_dict_list

    def predict_proba(self, data_arr):
        """
        Function
            为了使用LIME对RF的结果进行解释，需要一个能接受数组数据的函数，并输出概率预测的数组
        :param
            data_arr
                np.array = (n_Samples, n_Dimensions)
        :return:
            proba_arr
                np.array = (n_Samples, n_labels_num)
        """
        sample_label_prob_list = []
        # Transform numpy array to list
        data_list = data_arr.tolist()
        for unlabeled_data_list in data_list:
            label_num_dict = {}
            for label in self.label_list:
                label_num_dict[label] = 0

            for random_tree in self.forest:
                pre_label = random_tree.classify(unlabeled_data_list)
                # KeyError: None
                label_num_dict[pre_label] += 1

            label_proba_list = []
            for label in self.label_list:
                label_proba_list.append(label_num_dict[label] / self.tree_num)
            sample_label_prob_list.append(label_proba_list)
        sample_label_prob_arr = np.array(sample_label_prob_list)
        return sample_label_prob_arr

def save_random_forest(random_forest, filename, filepath):
    file_abs_path = os.path.join(filepath, filename)
    with open(file_abs_path, 'wb') as file:
        pickle.dump(random_forest, file)

def load_random_forest(filename, filepath):
    file_abs_path = os.path.join(filepath, filename)
    with open(file_abs_path, 'rb') as file:
        random_forest = pickle.load(file)
    return random_forest