import sys
import math
import pickle
import random

from ml_sl.ml_data_wrapper import single_point_list_2_list, pack_list_2_list
from ml_sl.test.load_testSet import read_testSet

def sigmoid(x):
    try:
        if -x < -100:
            z = 1
            return z
        elif -100 <= -x <= 100:
            z = 1 / (1 + math.exp(-x))
            return z
        elif -x > 100:
            z = 0
            return z
    except OverflowError as e:
        print('x=',x) # x= -909.7916515971691, when alpha_init is equals to or bigger than 10
        print('math.exp(x)=',math.exp(-x))
        print(e)
        sys.exit(1)

def gradient_descent(data_i_list, data_j_list, stable_alpha, iter_time=1000, alpha_init=None):
    m = len(data_i_list) + len(data_j_list)
    # W_list = w_list + b
    W_list = [random.random() for i in range(len(data_i_list[0]) + 1)]
    for iter_index in range(iter_time):
        # alpha is similar to the learning rate in NN, its value can be stable or decrease along with iteration
        if stable_alpha:
            # Stable alpha
            alpha = 1 / iter_time
        else:
            # alpha decreases along with iteration， alpha的初始值设置要慎重，大了（=1），效果很差
            # alpha = 0.01 * (iter_time - iter_index) / iter_time
            alpha = alpha_init * (iter_time - iter_index) / iter_time

        d_J_list = []
        for data_i in data_i_list:
            d = data_i + [1]
            z = sum([W * x for W, x in zip(W_list, d)])
            pre_y = sigmoid(z)
            tmp_d_J_list = [(pre_y - 0) * x for x in d]
            d_J_list.append(tmp_d_J_list)
        for data_j in data_j_list:
            d = data_j + [1]
            z = sum([W * x for W, x in zip(W_list, d)])
            pre_y = sigmoid(z)
            tmp_d_J_list = [(pre_y - 1) * x for x in d]
            d_J_list.append(tmp_d_J_list)
        avg_d_J_list = []
        for i in range(len(d_J_list[0])):
            avg_d_J = sum([d_j[i] for d_j in d_J_list]) / m
            avg_d_J_list.append(avg_d_J)
        W_list = [W - alpha * d_j for W, d_j in zip(W_list, avg_d_J_list)]
    return W_list

def create_lrc(data_i_list, data_j_list, stable_alpha=True, max_iter = 1000, alpha_init=None):
    """
    策略
        一对一
        再缩放的策略
            直接基于原始训练集进行学习，但在用训练好的分类器进行预测时，将式(3.48)嵌入到其决策过程中，称为"阈值移动" (threshold-moving)
    :param
        data_i_list:
            label 0
            [
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                ...
            ]
        data_j_list:
            label 1
            [
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
                ...
            ]
    :return:
        ‘w’：权重列表 【float】
         'w': w_list [w0, w1, w2, ..., w5]
         ‘b’：偏置 float
         'b': b
         ratio:
            data_i中样本数目与data_j中样本数目的比值
    """
    # ratio = number of J / number of I
    ratio = len(data_j_list) / len(data_i_list)
    data_i_list = pack_list_2_list(data_i_list)
    data_j_list = pack_list_2_list(data_j_list)

    # W_list = w_list + b
    W_list = gradient_descent(data_i_list, data_j_list, stable_alpha, iter_time=max_iter, alpha_init=alpha_init)
    w_list = W_list[: len(W_list) - 1]
    b = W_list[-1]
    return w_list, b, ratio

def binary_lrc_classify(w_list, b, ratio, x_list):
    z = sum([w * x for w, x in zip(w_list, x_list)]) + b
    pre = sigmoid(z)

    if pre == 1.0:
        scaled_pre_y = 1.0
    else:
        # ratio = number of J(m+) / number of I(m-)
        scale_facotr = pre / (1 - pre) / ratio
        scaled_pre_y = scale_facotr / (1 + scale_facotr)
    if scaled_pre_y <= 0.5:
        lable = 0
    elif scaled_pre_y <= 1:
        lable = 1
    else:
        print('Prediction:', scaled_pre_y)
        print('Prediction of logisticRegresion is above 1')
        import sys
        sys.exit(1)
    return lable

# ---------------------------- Test of function create_lrc ----------------------------
# if __name__ == '__main__':
    # --------- linear test points ---------
    # data_i_list = [
    #                 [(0, 4)],
    #                 [(1, 5)],
    #                 [(2, 6)],
    #                 [(3, 7)],
    #
    #                 [(4, 8)],
    #                 [(5, 9)],
    #                 [(6, 10)],
    #                 [(7, 11)],
    #               ]
    # data_j_list = [
    #                 [(0, -4)],
    #                 [(1, -5)],
    #                 [(2, -6)],
    #                 [(3, -7)],
    #
    #                 [(4, -8)],
    #                 [(5, -9)],
    #                 [(6, -10)],
    #                 [(7, -11)],
    #               ]
    # x_list = [2, 9]
    # x1_list = [2, -9]
    # --------- linear test points ---------
    # --------- linear test points ---------

    # --------- nonlinear test points ---------
    # data_i_list = [
    #                 [(1,0)],
    #                 [(-1,0)],
    #                 [(0,1)],
    #                 [(0,-1)],
    #
    #                 [(1,1)],
    #                 [(1,-1)],
    #                 [(-1,1)],
    #                 [(-1,-1)],
    #               ]
    # data_j_list = [
    #                 [(2,0)],
    #                 [(-2,0)],
    #                 [(0,2)],
    #                 [(0,-2)],
    #
    #                 [(2,2)],
    #                 [(2,-2)],
    #                 [(-2,2)],
    #                 [(-2,-2)]
    #               ]
    # --------- nonlinear test points ---------

    # x_list = [0, 0]

    # --------- linear test points from testSet.txt (Machine learning in Action chapter 5)---------
    # data_i_list, data_j_list = read_testSet('../test/testSet.txt')
    # x_list = [0, 10]
    # x1_list = [0, 0]
    # --------- linear test points from testSet.txt (Machine learning in Action chapter 5)---------

    # max_iter = 500
    # print('Iteration', max_iter)

    # stable_alpha = True
    # w_list, b, ratio = create_lrc(data_i_list, data_j_list, stable_alpha, max_iter)
    # print('w:{0}, b:{1}'.format(w_list, b))
    # f = binary_lrc_classify(w_list, b, ratio, x_list)
    # print('For unseen data {0}, after {1} iteration with stable alpha, the prediction is {2}'.format(x_list, max_iter ,f))
    # f1 = binary_lrc_classify(w_list, b, ratio, x1_list)
    # print('For unseen data {0}, after {1} iteration with stable alpha, the prediction is {2}'.format(x1_list, max_iter, f1))

    # def lrc_classify_testSet(data_i_list, data_j_list):
    #     right_label_count = 0
    #     for d_i in data_i_list:
    #         f = binary_lrc_classify(w_list, b, ratio, d_i[0])
    #         if f <= 0.5:
    #             right_label_count += 1
    #     for d_j in data_j_list:
    #         f = binary_lrc_classify(w_list, b, ratio, d_j[0])
    #         if f > 0.5 and f < 1:
    #             right_label_count += 1
    #     return right_label_count
    # right_label_count = lrc_classify_testSet(data_i_list, data_j_list)
    # print('Correct prediction: {0}/{1}'.format(right_label_count, len(data_i_list)+len(data_j_list))) # Iteration 500, Correct prediction: 77/100

    # stable_alpha = False
    # w_list, b, ratio = create_lrc(data_i_list, data_j_list, stable_alpha, max_iter)
    # print('w:{0}, b:{1}'.format(w_list, b))
    # f = binary_lrc_classify(w_list, b, ratio, x_list)
    # print('For unseen data {0}, after {1} iteration with linear decreasing alpha, the prediction is {2}'.format(x_list, max_iter ,f))
    # f1 = binary_lrc_classify(w_list, b, ratio, x1_list)
    # print('For unseen data {0}, after {1} iteration with linear decreasing alpha, the prediction is {2}'.format(x1_list, max_iter ,f1))
# ---------------------------- Test of function create_lrc ----------------------------

def lrc_ovo_linear_classify_on_one_slice(voted_label_list, x_list, lrc_ovo_linear_classifer_dict_pickle_filename):
    """
    Function
        在采取One Vs Rest 策略时，可能出现1-多个分类器投票；2-无分类器投票两种现象
            1-多个分类器投票
                假设投票的分类器为1-rest， 2-rest 和3-rest分类器
                则从ovo得到的最优分类器集合中抽取1-2，1-3和2-3三个二分类分类器再次进行分类 ==》 最终分类结果
            2-无分类器投票
                完全重新按照OVO的策略再次投票
    :param
        voted_label_list:
        x_list
        lrc_ovo_classifer_dict_pickle_filename
    :return:
        label_prob_dict
    """
    with open(lrc_ovo_linear_classifer_dict_pickle_filename, 'rb') as file:
        lrc_ovo_linear_classifer_dict = pickle.load(file)

    label_vote_dict = {}
    for label in voted_label_list:
        label_vote_dict[label] = 0

    vote_count = 0
    voted_label_list.sort()
    for index, label_i in enumerate(voted_label_list):
        for label_j in voted_label_list[index + 1 : ]:
            binary_lrc_classifier_dict = lrc_ovo_linear_classifer_dict[(label_i, label_j)]
            w_list = binary_lrc_classifier_dict['w']
            b = binary_lrc_classifier_dict['b']
            ratio = binary_lrc_classifier_dict['r']

            z = sum([w * x for w, x in zip(w_list, x_list)]) + b
            pre_y = sigmoid(z)

            # pre_y might be 1
            if pre_y == 1.0:
                scaled_pre_y = 1.0
            else:
                scale_facotr = pre_y / (1 - pre_y) / ratio
                scaled_pre_y = scale_facotr / (1 + scale_facotr)
            if scaled_pre_y <= 0.5:
                label_vote_dict[label_i] += 1
            elif scaled_pre_y <= 1:
                label_vote_dict[label_j] += 1
            vote_count += 1

    label_prob_dict = {}
    for key, value in label_vote_dict.items():
        label_prob_dict[key] = value / vote_count
    return label_prob_dict

class LRC():
    """
    LRC
        logistic regression classifer
        此算法主要参照 周志华 《机器学习》 第3章 线性模型，下面注释中的公式编号均与书中公式编号一致
        具体实现主要参照 Python机器学习算法 — 逻辑回归（Logistic Regression）
            https://www.cnblogs.com/lsqin/p/9342935.html
        多分类学习的策略
            一对一
            需构建的分类器数量 N*(N-1)/2 （N：ECMs的种类数目）
        再缩放的策略
            直接基于原始训练集进行学习，但在用训练好的分类器进行预测时，将式(3.48)嵌入到其决策过程中，称为"阈值移动" (threshold-moving)
    """
    def __init__(self, alpha, max_iter, unlabeled_dataset_list, labeled_dataset_list, label_list):
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
        """
        self.alpha = alpha
        self.max_iter = max_iter

        self.unlabeled_dataset_list = unlabeled_dataset_list
        self.labeled_dataset_list = labeled_dataset_list
        self.label_list = label_list

    # ovo_res: One Vs One
    def create_lrc_classifer_dict_ovo(self, lrc_classifer_dict_pickle_filename):
        """
        :return:
            lrc_classifer_dict
            {
                (类别a， 类别b)：{‘w’：权重列表 【float】， ‘b’：偏置 float}
                (0, 1): {'w': w_list [w0, w1, w2, ..., w5], 'b': b}
            }
        """
        label_sorted_data_dcit = {}
        for label in self.label_list:
            label_sorted_data_dcit[label] = [data[1] for data in self.labeled_dataset_list if data[0] == label]

        # 每次抽取不同的两类样本，构建分类器
        lrc_classifer_dict = {}
        for index, label_i in enumerate(self.label_list):
            for label_j in self.label_list[index + 1:]:
                label_i_data_list = label_sorted_data_dcit[label_i]
                label_j_data_list = label_sorted_data_dcit[label_j]
                if self.alpha == None:
                    # Gradient descent with stable learning rate
                    w_list, b, ratio = create_lrc(label_i_data_list, label_j_data_list,\
                                                  stable_alpha=True, max_iter = self.max_iter, alpha_init=self.alpha)
                elif (type(self.alpha) == float) or (type(self.alpha) == int):
                    # Gradient descent with linear decreasing learning rate
                    w_list, b, ratio = create_lrc(label_i_data_list, label_j_data_list,\
                                                  stable_alpha=False, max_iter=self.max_iter, alpha_init=self.alpha)
                else:
                    print('Current para: max_iter = {0}, alpha = {1}'.format(self.max_iter, self.alpha))
                    print('Choose a stable or linear decreasing Alpha')
                    import sys
                    sys.exit(1)
                lrc_classifer_dict[(label_i, label_j)] = {'w': w_list, 'b': b, 'r': ratio}
        # 使用pickle将训练好的LRC分类器保存成文件，存储起来，文件名包含日期，便于之后更新，区分
        with open(lrc_classifer_dict_pickle_filename, 'wb') as file:
            pickle.dump(lrc_classifer_dict, file)

    # ovo_res: One Vs One
    def create_wlrc_classifer_dict_ovo(self):
        """
        :return:
            lrc_classifer_dict
            {
                (类别a， 类别b)：{‘w’：权重列表 【float】， ‘b’：偏置 float}
                (0, 1): {'w': w_list [w0, w1, w2, ..., w5], 'b': b}
            }
        """
        label_sorted_data_dcit = {}
        for label in self.label_list:
            label_sorted_data_dcit[label] = [data[1] for data in self.labeled_dataset_list if data[0] == label]

        # 每次抽取不同的两类样本，构建分类器
        lrc_classifer_dict = {}
        for index, label_i in enumerate(self.label_list):
            for label_j in self.label_list[index + 1:]:
                label_i_data_list = label_sorted_data_dcit[label_i]
                label_j_data_list = label_sorted_data_dcit[label_j]
                # Gradient descent with linear decreasing learning rate
                w_list, b, ratio = create_lrc(label_i_data_list, label_j_data_list,\
                                              stable_alpha=False, max_iter=self.max_iter, alpha_init=self.alpha)
                lrc_classifer_dict[(label_i, label_j)] = {'w': w_list, 'b': b, 'r': ratio}
        return lrc_classifer_dict

    # ovr: One Vs Rest
    def create_lrc_classifer_dict_ovr(self, lrc_classifer_dict_pickle_filename):
        """
        算法
            按照OVR的策略，一共有K 类样本，只需构建K 个LR，分别是1-rest，2-rest，。。。，K-rest
            分类时，每个分类器投一票，可能的投票情况如下：
                1- 理想情况下只有一个分类器会投票，投票的分类器即为标签
                    投票结果：[1,0,0,0,0,0,0]
                2- 多个分类器投一票
                    投票结果：
                        [1,1,0,0,0,0,0]
                        [1,1,1,0,0,0,0]
                        [1,0,1,0,0,1,0]
                    2.1- 任选一个标签
                    2.2- 投票的分类器之间再采用OVO的方式进行判断
                3- 没有分类器投票
                    投票结果：[0,0,0,0,0,0,0]
                    采用OVO的方式进行判断
            统一：
                rest标签：0
                目标标签：1
        :param lrc_classifer_dict_pickle_filename:
        """
        # Extract data by ecm_num
        lrc_classifer_dict = {}
        for label in self.label_list:
            label_i_data_list = [data[1] for data in self.labeled_dataset_list if data[0] != label]
            label_j_data_list = [data[1] for data in self.labeled_dataset_list if data[0] == label]

            if self.alpha == None:
                # Gradient descent with stable learning rate
                w_list, b, ratio = create_lrc(label_i_data_list, label_j_data_list, \
                                              stable_alpha=True, max_iter=self.max_iter, alpha_init=self.alpha)
            elif (type(self.alpha) == float) or (type(self.alpha) == int):
                # Gradient descent with linear decreasing learning rate
                w_list, b, ratio = create_lrc(label_i_data_list, label_j_data_list, \
                                              stable_alpha=False, max_iter=self.max_iter, alpha_init=self.alpha)
            else:
                print('Current para: max_iter = {0}, alpha = {1}'.format(self.max_iter, self.alpha))
                print('Choose a stable or linear decreasing Alpha')
                import sys
                sys.exit(1)
            lrc_classifer_dict[label] = {'w': w_list, 'b': b, 'r': ratio}
        with open(lrc_classifer_dict_pickle_filename, 'wb') as file:
            pickle.dump(lrc_classifer_dict, file)

    def classify_ovo(self, lrc_classifer_dict_pickle_filename):
        # 使用pickle加载训练好的lrc分类器
        with open(lrc_classifer_dict_pickle_filename, 'rb') as file:
            lrc_classifer_dict = pickle.load(file)

        sample_label_prob_dict_list = []
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0

            # Transform the data type : [(x0, y0), (x1, y1)] ==> [x0, y0, x1, y1]
            x_list = single_point_list_2_list(unlabeled_data_list)

            for key, value_dict in lrc_classifer_dict.items():
                label_i, label_j = key
                w_list = value_dict['w']
                b = value_dict['b']
                ratio = value_dict['r']

                z = sum([w * x for w, x in zip(w_list, x_list)]) + b
                pre_y = sigmoid(z)
                # print('z1=',z, 'pre_y1=',pre_y)
                # import time
                # time.sleep(20)

                # pre_y might be 1
                if pre_y == 1.0:
                    scaled_pre_y = 1.0
                else:
                    # ratio = number of J(m+) / number of I(m-)
                    # scale_facotr = pre_y * ratio / (1 - pre_y)
                    scale_facotr = pre_y / (1 - pre_y) / ratio
                    scaled_pre_y = scale_facotr / (1 + scale_facotr)
                if scaled_pre_y <= 0.5:
                    label_vote_dict[label_i] += 1
                elif scaled_pre_y <= 1:
                    label_vote_dict[label_j] += 1

            # count_num == vote for label time == classifier number (each classifier vote a label one time)
            # label_num = len(self.label_list)
            # count_num = int(0.5 * label_num * (label_num - 1)) <==> the following line of code
            count_num = len(lrc_classifer_dict.keys())

            label_prob_dict = {}
            for key, value in label_vote_dict.items():
                label_prob_dict[key] = value / count_num
            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

    def classify_wlrc_ovo(self, lrc_classifer_dict):
        sample_label_prob_dict_list = []
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0

            # Transform the data type : [(x0, y0), (x1, y1)] ==> [x0, y0, x1, y1]
            x_list = single_point_list_2_list(unlabeled_data_list)

            for key, value_dict in lrc_classifer_dict.items():
                label_i, label_j = key
                w_list = value_dict['w']
                b = value_dict['b']
                ratio = value_dict['r']

                z = sum([w * x for w, x in zip(w_list, x_list)]) + b
                pre_y = sigmoid(z)
                # print('z1=',z, 'pre_y1=',pre_y)
                # import time
                # time.sleep(20)

                # pre_y might be 1
                if pre_y == 1.0:
                    scaled_pre_y = 1.0
                else:
                    # ratio = number of J(m+) / number of I(m-)
                    # scale_facotr = pre_y * ratio / (1 - pre_y)
                    scale_facotr = pre_y / (1 - pre_y) / ratio
                    scaled_pre_y = scale_facotr / (1 + scale_facotr)
                if scaled_pre_y <= 0.5:
                    label_vote_dict[label_i] += 1
                elif scaled_pre_y <= 1:
                    label_vote_dict[label_j] += 1

            # count_num == vote for label time == classifier number (each classifier vote a label one time)
            # label_num = len(self.label_list)
            # count_num = int(0.5 * label_num * (label_num - 1)) <==> the following line of code
            count_num = len(lrc_classifer_dict.keys())

            label_prob_dict = {}
            for key, value in label_vote_dict.items():
                label_prob_dict[key] = value / count_num
            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

    def classify_ovr(self, lrc_ovr_classifer_dict_pickle_filename):
        """
        按照OVR的策略，一共有K 类样本，只需构建K 个LR，分别是1-rest，2-rest，。。。，K-rest
            分类时，每个分类器投一票，可能的投票情况如下：
                1- 理想情况下只有一个分类器会投票，投票的分类器即为标签
                    投票结果：[1,0,0,0,0,0,0]
                2- 多个分类器投一票
                    投票结果：[1,0,0,1,0,1,0]
                    在获得投票的标签中任选一个
                3- 没有分类器投票
                    投票结果：[0,0,0,0,0,0,0]
                    在所有的标签中任选一个
        """
        # 使用pickle加载训练好的lrc分类器
        with open(lrc_ovr_classifer_dict_pickle_filename, 'rb') as file:
            lrc_ovr_classifer_dict = pickle.load(file)

        sample_label_prob_dict_list = []
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            label_prob_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0

            # Transform the data type : [(x0, y0), (x1, y1)] ==> [x0, y0, x1, y1]
            x_list = single_point_list_2_list(unlabeled_data_list)

            for label, value_dict in lrc_ovr_classifer_dict.items():
                w_list = value_dict['w']
                b = value_dict['b']
                ratio = value_dict['r']

                z = sum([w * x for w, x in zip(w_list, x_list)]) + b
                pre_y = sigmoid(z)

                if pre_y == 1.0:
                    scaled_pre_y = 1.0
                else:
                    scale_facotr = pre_y / (1 - pre_y) / ratio
                    scaled_pre_y = scale_facotr / (1 + scale_facotr)

                if 0.5 < scaled_pre_y <= 1+0.01:
                    label_vote_dict[label] += 1
                elif scaled_pre_y > 1:
                    print('Prediction:', scaled_pre_y)
                    print('Prediction of logistic Regression is above 1')
                    import sys
                    sys.exit(1)

            # 1 - 理想情况下只有一个分类器会投票，投票的分类器即为标签,投票结果：[1,0,0,0,0,0,0]
            if sum(label_vote_dict.values()) == 1:
                for k, v in label_vote_dict.items():
                    if v == 1:
                        label_prob_dict[k] = 1
                    else:
                        label_prob_dict[k] = 0

            # 2 - 多个分类器投一票, 投票结果：[1,0,1,0,0,1,0]
            elif sum(label_vote_dict.values()) > 1:
                print('Multi labels are voted')
                voted_label_list = [k for k, v in label_vote_dict.items() if v == 1]
                random_choiced_label = random.choice(voted_label_list)
                for k, v in label_vote_dict.items():
                    if k == random_choiced_label:
                        label_prob_dict[k] = 1
                    else:
                        label_prob_dict[k] = 0

            # 3- 没有分类器投票, 投票结果：[0,0,0,0,0,0,0]
            elif sum(label_vote_dict.values()) == 0:
                print('No label is voted')
                random_choiced_label = random.choice(self.label_list)
                for k, v in label_vote_dict.items():
                    if k == random_choiced_label:
                        label_prob_dict[k] = 1
                    else:
                        label_prob_dict[k] = 0
            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

    def classify_ovr_ovo(self, lrc_ovr_classifer_dict_pickle_filename, lrc_ovo_linear_classifer_dict_pickle_filename):
        """
        按照OVR的策略，一共有K 类样本，只需构建K 个LR，分别是1-rest，2-rest，。。。，K-rest
            分类时，每个分类器投一票，可能的投票情况如下：
                1- 理想情况下只有一个分类器会投票，投票的分类器即为标签
                    投票结果：[1,0,0,0,0,0,0]
                2- 多个分类器投一票
                    投票结果：
                        [1,1,0,0,0,0,0]
                        [1,1,1,0,0,0,0]
                        [1,0,1,0,0,1,0]
                    2.1- 任选一个标签
                    2.2- 投票的分类器之间再采用OVO的方式进行判断
                3- 没有分类器投票
                    投票结果：[0,0,0,0,0,0,0]
                    采用OVO的方式进行判断
        """
        # 使用pickle加载训练好的lrc分类器
        with open(lrc_ovr_classifer_dict_pickle_filename, 'rb') as file:
            lrc_ovr_classifer_dict = pickle.load(file)

        sample_label_prob_dict_list = []
        ovo_vote_rate = 0.0
        ovo_vote_count = 0
        # 遍历每一条无标签的样本
        for unlabeled_data_list in self.unlabeled_dataset_list:
            label_vote_dict = {}
            label_prob_dict = {}
            for label in self.label_list:
                label_vote_dict[label] = 0

            # Transform the data type : [(x0, y0), (x1, y1)] ==> [x0, y0, x1, y1]
            x_list = single_point_list_2_list(unlabeled_data_list)

            for label, value_dict in lrc_ovr_classifer_dict.items():
                w_list = value_dict['w']
                b = value_dict['b']
                ratio = value_dict['r']

                z = sum([w * x for w, x in zip(w_list, x_list)]) + b
                pre_y = sigmoid(z)

                if pre_y == 1.0:
                    scaled_pre_y = 1.0
                else:
                    scale_facotr = pre_y / (1 - pre_y) / ratio
                    scaled_pre_y = scale_facotr / (1 + scale_facotr)

                if 0.5 < scaled_pre_y <= 1:
                    label_vote_dict[label] += 1
                elif scaled_pre_y > 1:
                    print('Prediction:', scaled_pre_y)
                    print('Prediction of logistic Regression is above 1')
                    import sys
                    sys.exit(1)

            # 1 - 理想情况下只有一个分类器会投票，投票的分类器即为标签,投票结果：[1,0,0,0,0,0,0]
            if sum(label_vote_dict.values()) == 1:
                for k, v in label_vote_dict.items():
                    if v == 1:
                        label_prob_dict[k] = 1
                    else:
                        label_prob_dict[k] = 0

            # 2 - 多个分类器投一票, 投票结果：[1,0,1,0,0,1,0]
            elif sum(label_vote_dict.values()) > 1:
                print('Multi labels are voted')
                ovo_vote_count += 1
                voted_label_list = [k for k, v in label_vote_dict.items() if v == 1]
                voted_label_prob_dict = lrc_ovo_linear_classify_on_one_slice(voted_label_list, x_list, lrc_ovo_linear_classifer_dict_pickle_filename)
                for k, v in label_vote_dict.items():
                    if k in voted_label_prob_dict.keys():
                        label_prob_dict[k] = voted_label_prob_dict[k]
                    else:
                        label_prob_dict[k] = 0

            # 3- 没有分类器投票, 投票结果：[0,0,0,0,0,0,0]
            elif sum(label_vote_dict.values()) == 0:
                print('No label is voted')
                ovo_vote_count += 1
                voted_label_prob_dict = lrc_ovo_linear_classify_on_one_slice(self.label_list, x_list, lrc_ovo_linear_classifer_dict_pickle_filename)
                label_prob_dict = voted_label_prob_dict
            sample_label_prob_dict_list.append(label_prob_dict)
        ovo_vote_rate = ovo_vote_count / len(self.unlabeled_dataset_list)
        return sample_label_prob_dict_list, ovo_vote_rate

# ---------------------------- Test of Class LRC ----------------------------
# --------- Test of Class LRC (linear data test) ---------
# if __name__ == '__main__':
#     unlabeled_dataset_list = [
#         [(1, 0)],
#         [(1, 1)],
#         [(1, 2)],
#         [(1, 3)]
#     ]
#     labeled_dataset_list = [
#         [1, [(1, 0)]],
#         [1, [(2, 1)]],
#         [1, [(3, 2)]],
#         [1, [(4, 3)]],
#
#         [2, [(0, 0)]],
#         [2, [(1, 1)]],
#         [2, [(2, 2)]],
#         [2, [(3, 3)]],
#
#         [3, [(0, 1)]],
#         [3, [(1, 2)]],
#         [3, [(2, 3)]],
#         [3, [(3, 4)]],
#
#         [4, [(0, 2)]],
#         [4, [(1, 3)]],
#         [4, [(2, 4)]],
#         [4, [(3, 5)]]
#     ]
#     label_list = [1, 2, 3, 4]
#     lrc = LRC(unlabeled_dataset_list, labeled_dataset_list, label_list)
#     lrc_classifer_dict_pickle_filename = '2020_03_18_lrc_classifer_dict_pickle_linear_points_test_3.file'
#
#     lrc.create_lrc_classifer_dict(lrc_classifer_dict_pickle_filename)
#
#     sample_label_prob_dict_list = lrc.classify(lrc_classifer_dict_pickle_filename)
#     print(sample_label_prob_dict_list)
    """
    Right result
         [
            {1: 1.0, 2: 0, 3: 0, 4: 0}, 
            {1: 0.0, 2: 1, 3: 0, 4: 0}, 
            {1: 0.0, 2: 0, 3: 1, 4: 0}, 
            {1: 0.0, 2: 0, 3: 0, 4: 1}
        ]
    Results for four time (alpha = 1/iter_time, iter_time = 1000)
        [
            {1: 0.3333333333333333, 2: 0.3333333333333333, 3: 0.16666666666666666, 4: 0.16666666666666666}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}
        ] accuracy: 50%
        [
            {1: 0.16666666666666666, 2: 0.3333333333333333, 3: 0.16666666666666666, 4: 0.3333333333333333}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}
        ] accuracy: 25%
        [
            {1: 0.16666666666666666, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.3333333333333333}, 
            {1: 0.16666666666666666, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.3333333333333333}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}, 
            {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}
        ] accuracy: 25%
    """

# --------- Test of Class LRC (linear data test) ---------
# --------- Test of Class LRC (nonlinear data test) ---------
# if __name__ == '__main__':
#     unlabeled_dataset_list = [
#         [(1, 1)],
#         [(1, -1)],
#         [(-1, -1)],
#         [(-1, 1)]
#     ]
#     labeled_dataset_list = [
#         [1, [(1, 1)]],
#         [1, [(1, 2)]],
#         [1, [(2, 1)]],
#         [1, [(2, 2)]],
#
#         [2, [(1, -1)]],
#         [2, [(1, -2)]],
#         [2, [(2, -2)]],
#         [2, [(2, -1)]],
#
#         [3, [(-1, 1)]],
#         [3, [(-1, 2)]],
#         [3, [(-2, 1)]],
#         [3, [(-2, 2)]],
#
#         [4, [(-2, -1)]],
#         [4, [(-2, -2)]],
#         [4, [(-1, -1)]],
#         [4, [(-1, -2)]]
#     ]
#     label_list = [1, 2, 3, 4]
#     lrc = LRC(unlabeled_dataset_list, labeled_dataset_list, label_list)
#     lrc_classifer_dict_pickle_filename = '2020_03_18_lrc_classifer_dict_pickle_nonlinear_points_test_3.file'
#
#     lrc.create_lrc_classifer_dict(lrc_classifer_dict_pickle_filename)
#
#     sample_label_prob_dict_list = lrc.classify(lrc_classifer_dict_pickle_filename)
#     print(sample_label_prob_dict_list)
# Right result
#      [
#         {1: 1.0, 2: 0, 3: 0, 4: 0},
#         {1: 0.0, 2: 1, 3: 0, 4: 0},
#         {1: 0.0, 2: 0, 3: 1, 4: 0},
#         {1: 0.0, 2: 0, 3: 0, 4: 1}
#     ]
# Results for four time (alpha = 1/iter_time, iter_time = 1000)
#     [
#         {1: 0.16666666666666666, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.3333333333333333},
#         {1: 0.0, 2: 0.3333333333333333, 3: 0.16666666666666666, 4: 0.5},
#         {1: 0.0, 2: 0.3333333333333333, 3: 0.16666666666666666, 4: 0.5},
#         {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5}
#     ] accuracy: 25%
#     [
#         {1: 0.3333333333333333, 2: 0.0, 3: 0.3333333333333333, 4: 0.3333333333333333},
#         {1: 0.16666666666666666, 2: 0.5, 3: 0.0, 4: 0.3333333333333333},
#         {1: 0.16666666666666666, 2: 0.3333333333333333, 3: 0.16666666666666666, 4: 0.3333333333333333},
#         {1: 0.16666666666666666, 2: 0.0, 3: 0.5, 4: 0.3333333333333333}
#     ] accuracy: 50%
#     [
#         {1: 0.3333333333333333, 2: 0.0, 3: 0.3333333333333333, 4: 0.3333333333333333},
#         {1: 0.16666666666666666, 2: 0.5, 3: 0.0, 4: 0.3333333333333333},
#         {1: 0.0, 2: 0.16666666666666666, 3: 0.3333333333333333, 4: 0.5},
#         {1: 0.16666666666666666, 2: 0.0, 3: 0.3333333333333333, 4: 0.5}
#     ] accuracy: 75%
#     [
#         {1: 0.3333333333333333, 2: 0.0, 3: 0.3333333333333333, 4: 0.3333333333333333},
#         {1: 0.3333333333333333, 2: 0.5, 3: 0.0, 4: 0.16666666666666666},
#         {1: 0.16666666666666666, 2: 0.3333333333333333, 3: 0.0, 4: 0.5},
#         {1: 0.0, 2: 0.16666666666666666, 3: 0.5, 4: 0.3333333333333333}
#     ] accuracy: 50%
# --------- Test of Class LRC (nonlinear data test) ---------
# ---------------------------- Test of Class LRC ---------------------------