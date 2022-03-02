import sys
import math
from ml_sl.test.load_testSet import read_testSet, read_testSet_with_label
from ml_sl.ml_data_wrapper import pack_list_2_list, split_labeled_dataset_list, single_point_list_2_list

# sigma == 标准差（Standard Deviation）
def gauss(x, u, sigma):
    a = 1 / (math.sqrt(2 * math.pi) * sigma)
    # When The value of math.exp(Num) is too small, math.exp(Num) will return 0.0
    b = math.exp(- ((x - u) ** 2) / (2 * sigma * sigma) )
    if b == 0.0:
        b = 1e-30
    return a * b

def get_u_sigma(data_list):
    # data_list =
    data_list = pack_list_2_list(data_list)
    u_list = []
    sigma_list = []
    for i in range(len(data_list[0])):
        num_list = [data[i] for data in data_list]
        u = sum(num_list) / len(num_list)
        u_list.append(u)
        sigma = math.sqrt(sum([(num - u) ** 2 for num in num_list]) / len(num_list))
        sigma_list.append(sigma)
    return u_list, sigma_list

class NBC():
    """
    NBC
        Naive Bayes Classifer
        此算法主要参照 周志华 《机器学习》 第七章 贝叶斯分类器，下面注释中的公式编号均与书中公式编号一致
        在样本某一属性的概率计算时，以及某类样本出现频率的计算时，均不存在样本数量为0的情况，因此无需采用拉格朗日修正
    """
    def __init__(self, unlabeled_dataset_list, labeled_dataset_list, label_list):
        """
        :param
            unlabeled_dataset_list:
                Possibility 1
                    a group of unlabeled samples
                        [
                            [points list]
                            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                            ...
                        ]
                Possibility 2
                    a sample
                        [
                            [points list]
                            [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
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
        self.unlabeled_dataset_list = unlabeled_dataset_list
        self.labeled_dataset_list = labeled_dataset_list
        self.label_list = label_list

    def classify(self):
        # 计算各类样本在数据集(训练+验证)中的频率p(C)， eq 7.19
        label_dict = {}
        for label in self.label_list:
            label_dict[label] = sum([label == data[0] for data in self.labeled_dataset_list])

        self.label_fre_dict = {}
        dataset_len = len(self.labeled_dataset_list)
        for label in self.label_list:
            # 有拉格朗日修正
            # numerator = label_dict[label] / dataset_len + 1
            # denominator = dataset_len + len(self.label_list)

            # 无拉格朗日修正
            numerator = label_dict[label]
            denominator = dataset_len
            self.label_fre_dict[label] = numerator / denominator

        # 计算p(X|c) = p(x0|c) * p(x1|c) * p(x2|c) * ... * p(xn-2|c) * p(xn-1|c) eq 7.20
        # 为防止太多小数相乘导致计算下溢，上式eq 7.20 采用对数化，将乘法转化为加法
        u_sigma_dict = {}
        for label in self.label_list:
            tmp_data_list = []
            for data in self.labeled_dataset_list:
                if data[0] == label:
                    tmp_data_list.append(data[1])
            # tmp_data_list = pack_list_2_list(tmp_data_list)
            u_list, sigma_list = get_u_sigma(tmp_data_list)
            u_sigma_dict[label] = {'u':u_list, 'sigma':sigma_list}

        sample_label_prob_dict_list = []
        # 遍历每个无标记样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            unlabeled_data_list = single_point_list_2_list(unlabeled_data_list)
            label_proba_dict = {}
            label_p_list = []
            for label in self.label_list:
                log_p_multi = 0.0
                u_list = u_sigma_dict[label]['u']
                sigma_list = u_sigma_dict[label]['sigma']
                for x, u, sigma in zip(unlabeled_data_list, u_list, sigma_list):
                    # 目标：计算p(c|X) = p(X|c) * p(C) / p(X)
                    try:
                        log_p_multi += math.log(gauss(x, u, sigma), 2)
                    except ValueError as e:
                        print('x={0},u={1},sigma={2},Gauss={3}'.format(x, u, sigma, gauss(x,u,sigma)))
                        print(e)
                        sys.exit(1)

                label_p_list.append(pow(2, log_p_multi) * self.label_fre_dict[label])

            for index, label in enumerate(self.label_list):
                label_proba_dict[label] = label_p_list[index] / sum(label_p_list)
            sample_label_prob_dict_list.append(label_proba_dict)
        return sample_label_prob_dict_list

# -------------Test Naive Bayesian classifier on testSet (Binary Classification)-------------
# if __name__ == '__main__':
#     unlabeled_dataset_list = [[(-0.017612,	14.053064)],
#                               [(-1.395634,	4.662541)],
#                               [(-0.752157,	6.538620)],
#                               [(-1.322371,	7.152853)],
#                               [(0.423363,	11.054677)]]
#
#     labeled_data_i_list, labeled_data_j_list = read_testSet_with_label(filename='../test/testSet.txt')
#     labeled_data_i_list.extend(labeled_data_j_list)
#
#     label_list = [0, 1]
#     nbc = NBC(unlabeled_dataset_list, labeled_data_i_list, label_list)
#     sample_label_prob_dict_list = nbc.classify()
#     print(sample_label_prob_dict_list)
    """
    Right result
         [
            {0: 1, 1: 0}, 
            {0: 0, 1: 1}, 
            {0: 1, 1: 0}, 
            {0: 1, 1: 0},
            {0: 1, 1: 0}
        ]
    Results of NBC(the result will always be the same with the same dataset)
        [
            {0: 0.9967002861247554, 1: 0.003299713875244549}, 
            {0: 0.008576110297285585, 1: 0.9914238897027144}, 
            {0: 0.23084397705238166, 1: 0.7691560229476183}, 
            {0: 0.3444622818343295, 1: 0.6555377181656706}, 
            {0: 0.9847430356230701, 1: 0.015256964376929794}
        ] accuracy: 60%
    """
# -------------Test Naive Bayesian classifier on testSet (Binary Classification)-------------