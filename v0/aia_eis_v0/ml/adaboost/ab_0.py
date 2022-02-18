import sys
import pickle
import math
import random

from ml.logistic.lrc_0 import LRC

def cal_resample_score(w):
    """
    :param
        w:
            the weight of a sample
    :return:
        s
            score for the sample with the weight (w)
    refer:
        机器学习-加权采样算法简介
            https://blog.csdn.net/xiao2cai3niao/article/details/80587471
    """
    R = random.uniform(0, 1)
    s = R ** (1/w)
    return s

class AB:
    """
    AdaBoost
        Weak learner
            Weighted Logistic regression classifier
        weight
            resampling
                机器学习-加权采样算法简介
                https://blog.csdn.net/xiao2cai3niao/article/details/80587471
        Linear alpha
        Multiclassification strategy:
            One VS One
        Refer
            paper:
                <A Short Introduction to Boosting>
                <SPECIAL INVITED PAPER ADDITIVE LOGISTIC REGRESSION: A STATISTICAL VIEW OF BOOSTING>
    """

    def __init__(self, boost_num, resample_num, alpha_init, max_iter, unlabeled_dataset_list, labeled_dataset_list, label_list):
        """
        :param
            boost_num
                Number of weak learner
            resample_num
                During resampling, pick out resample_num samples from dataset and add into the resampled_dataset
            w_list
                list[weight_0(float), weight_1(float), weight_2(float), ...]
                    weight_0(float): the weight of the first sample in labeled_dataset_list
                    weight_1(float): the weight of the second sample in labeled_dataset_list
            alpha_init
                Learning rate (Step size) in gradient descent
            max_iter
                Maximum iteration of gradient descent
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
        self.boost_num = boost_num
        self.resample_num = resample_num
        self.alpha_init = alpha_init
        self.max_iter = max_iter

        self.unlabeled_dataset_list = unlabeled_dataset_list
        self.labeled_dataset_list = labeled_dataset_list
        self.label_list = label_list

        # self.w_list = [1 / len(self.labeled_dataset_list) for i in range(len(self.labeled_dataset_list))]
        self.w_list = [1 / len(self.labeled_dataset_list)] * len(self.labeled_dataset_list)

    def resample(self):
        """
        During resample by weight, some class might not be choosen,
        Ensure that there is at least one sample of one class in the resample_data_list
        :return:
        """
        def all_kind_resample_check(resample_data_list):
            existed_label_set = set([d[0] for d in resample_data_list])
            left_label_list = list(set(self.label_list) - existed_label_set)
            if len(left_label_list) == 0:
                return True
            else:
                return left_label_list

        resample_data_list = []
        while len(resample_data_list) <= len(self.labeled_dataset_list):
            s_list = [cal_resample_score(w) for w in self.w_list]
            # Descending order
            sorted_s_list = sorted(s_list, reverse=True)
            first_big_s_index_list = list(map(s_list.index, sorted_s_list[:self.resample_num]))
            resample_data_list.extend([self.labeled_dataset_list[i] for i in first_big_s_index_list])
        resample_data_list = resample_data_list[: len(self.labeled_dataset_list)]

        while True:
            check_res = all_kind_resample_check(resample_data_list)
            if type(check_res) == bool:
                # all_kind_resample_check_flag = True
                return resample_data_list
            elif type(check_res) == list:
                insert_data_list = []
                for left_label in check_res:
                    data_list = [d for d in self.labeled_dataset_list if d[0] == left_label]
                    insert_data_list.append(data_list[random.randint(0, len(data_list)-1)])
                resample_data_list = insert_data_list + resample_data_list[len(insert_data_list) :]

    def create_ab_classifer(self, ab_model_name):
        ab_dict_list = []
        for b_i in range(self.boost_num):
            # 1-Resample data
            resample_data_list = self.resample()

            # 2-Build LRC
            lrc = LRC(alpha=self.alpha_init, max_iter=self.max_iter, unlabeled_dataset_list=None,\
                      labeled_dataset_list=resample_data_list, label_list=self.label_list)

            lrc_classifer_dict = lrc.create_wlrc_classifer_dict_ovo()
            lrc_label_list = [d[0] for d in resample_data_list]
            lrc.unlabeled_dataset_list = [d[1] for d in resample_data_list]
            sample_label_prob_dict_list = lrc.classify_wlrc_ovo(lrc_classifer_dict)

            # 3-calculate err_w(Ew) --> c
            E = 0
            correct_predict_flag_list = []
            for sample_label_prob_dict, label in zip(sample_label_prob_dict_list, lrc_label_list):
                max_prob = max(sample_label_prob_dict.values())
                key_list = []
                for k, v in sample_label_prob_dict.items():
                    if v == max_prob:
                        key_list.append(k)
                if label in key_list:
                    correct_predict_flag_list.append(True)
                    # continue
                else:
                    correct_predict_flag_list.append(False)
                    E += 1
            err_w = E / len(lrc_label_list)
            # When all the classification is correct, the err_w is 0.0, (1-err_w)/err_w will cause ZeroDivisionError
            try:
                if err_w > 0.0:
                    c = math.log((1-err_w)/err_w, math.e)
                elif err_w == 0.0:
                    err_w = 0.1 / len(lrc_label_list)
                    c = math.log((1 - err_w) / err_w, math.e)
            except ZeroDivisionError as e:
                print(e)
                sys.exit(1)
            ab_dict_list.append({'model': lrc_classifer_dict, 'c': c})

            # 4-Update w_list
            new_w_list = []
            for w, correct_predict_flag in zip(self.w_list, correct_predict_flag_list):
                if correct_predict_flag:
                    new_w_list.append(w)
                else:
                    new_w = w * math.exp(c)
                    new_w_list.append(new_w)
            self.w_list = [w / sum(new_w_list) for w in new_w_list]

        # Store AB model
        with open(ab_model_name, 'wb') as file:
            pickle.dump(ab_dict_list, file)

    def classify(self, ab_model_name):
        with open(ab_model_name, 'rb') as file:
            ab_dict_list = pickle.load(file)

        ab_sample_label_prob_dict_list = []
        # for label in self.label_list:
        #     ab_sample_label_prob_dict_list[]
        lrc = LRC(alpha=None, max_iter=None, unlabeled_dataset_list=self.unlabeled_dataset_list,\
                  labeled_dataset_list=None, label_list=self.label_list)
        c_list = []
        lrc_sample_label_prob_dict_list = []
        for model_dict in ab_dict_list:
            lrc_classifer_dict = model_dict['model']
            c_list.append(model_dict['c'])
            sample_label_prob_dict_list = lrc.classify_wlrc_ovo(lrc_classifer_dict)
            lrc_sample_label_prob_dict_list.append(sample_label_prob_dict_list)

        for i in range(len(self.unlabeled_dataset_list)):
            ab_sample_label_prob_dict = {}
            for label in self.label_list:
                ab_sample_label_prob_dict[label] = 0.0
                for sample_label_prob_dict_list, c in zip(lrc_sample_label_prob_dict_list, c_list):
                    ab_sample_label_prob_dict[label] += c * sample_label_prob_dict_list[i][label]
            ab_sample_label_prob_dict_list.append(ab_sample_label_prob_dict)
        return ab_sample_label_prob_dict_list