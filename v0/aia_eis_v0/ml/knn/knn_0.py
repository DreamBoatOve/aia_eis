from ml.knn.distance_pack.distance_measures import distance_measure_0
import heapq

class KNN():
    def __init__(self, K, unlabeled_dataset_list, labeled_dataset_list, distance_mode, label_list):
        """
        :param
            K:
                K nearest neighbors
                K >= the classification kinds + 1
                (Default) K = the classification kinds + 1
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
            distance_mode (d_type_str):
                'bc_d'
                    bray_curtis_distance
                    measure distance between numbers
                'cheb_d'
                    chebyshev_distance
                    measure distance between points
                'cos_d'
                    cosine_distance
                    measure distance between numbers
                'dtw_d'
                    Dynamic time warping distance
                    measure distance between points
                'em_d'
                    earth_mover_distance
                    measure distance between points
                'e_d'
                    euclidean_distance
                    measure distance between points
                'jsd_d'
                    Jensen-Shanon Divergence
                    measure distance between norm of impedance
                    Why do not measure the distance between numbers or points?
                        KLD calculate the difference between two distribution that all the number are above
                        if use numbers or points directly, the calculation will involve negative values
                'maha_d'
                    mahalanobis_distance
                    measure distance between numbers
                'manha_d'
                    manhattan_distance
                    measure distance between points == measure distance between numbers
                'pcc_d'
                    Pearson correlation coefficient
                    measure distance between norm of impedance
                'se_d'
                    standardized_euclidean_distance
                    measure distance between points
            label_list
                record the types of ECMs, this parameter should be manually set
                [0,1,2,3, ... , 7, 8, 9]
        """
        self.K = K
        self.unlabeled_dataset_list = unlabeled_dataset_list
        self.labeled_dataset_list = labeled_dataset_list
        self.distance_mode = distance_mode
        self.label_list = label_list

    def classify(self):
        sample_label_prob_dict_list = []
        # 按照某种距离测量方式，计算无标签样本与数据集中的每个有标签样本的距离
        for unlabeled_data_list in self.unlabeled_dataset_list:
            d_list = distance_measure_0(x_list=unlabeled_data_list, data_list=self.labeled_dataset_list, d_type_str=self.distance_mode)

            # --------------------- 找最近的K个近邻的索引 ---------------------
            # 找最近的K个近邻的索引-Solution 1
            # If the first K small distances are the same, this solution will always find the index of the first one small distance
            # neighbor_index_list = list(map(d_list.index, heapq.nsmallest(self.K, d_list)))

            # 找最近的K个近邻的索引-Solution 2 == Solution 1
            # If the first K small distances are the same, this solution will always find the index of the first one small distance
            # sorted_d_list = sorted(d_list)
            # neighbor_index_list = [d_list.index(d) for d in sorted_d_list[:self.K]

            # 找最近的K个近邻的索引-Solution 3
            d_and_index_list = [[d, i] for i, d in enumerate(d_list)]
            sorted_d_and_index_list = sorted(d_and_index_list, key=lambda d_i : d_i[0])
            neighbor_index_list = [d_i[1] for d_i in sorted_d_and_index_list[: self.K]]
            # --------------------- 找最近的K个近邻的索引 ---------------------

            # 初始化样本的标签字典
            label_dict = {}
            for label in self.label_list:
                label_dict[label] = 0
            # 最近的K个近邻进行投票
            for neighbor_index in neighbor_index_list:
                neighbor_label = self.labeled_dataset_list[neighbor_index][0]
                label_dict[neighbor_label] += 1
            # 每个标签的投票数 / 投票的邻居数量 = 每个标签的概率
            label_prob_dict = {}
            for label in self.label_list:
                label_prob_dict[label] = label_dict[label] / self.K
            sample_label_prob_dict_list.append(label_prob_dict)
        return sample_label_prob_dict_list

# --------------------------- Use manual made data to test KNN ---------------------------
# if __name__ == '__main__':
#     unlabeled_dataset_list = [
#                                 [(1, 1)],
#                                 [(1, -1)],
#                                 [(-1, -1)],
#                                 [(-1, 1)]
#                             ]
#     labeled_dataset_list = [
#                             # [1, [(0.1, 0.1)]],
#                             [1, [(1,1)]],
#                             [1, [(1,2)]],
#                             [1, [(2,1)]],
#                             [1, [(2,2)]],
#
#                             [2, [(1,-1)]],
#                             [2, [(1,-2)]],
#                             [2, [(2,-2)]],
#                             [2, [(2,-1)]],
#
#                             [3, [(-1,1)]],
#                             [3, [(-1,2)]],
#                             [3, [(-2,1)]],
#                             [3, [(-2,2)]],
#
#                             [4, [(-2,-1)]],
#                             [4, [(-2,-2)]],
#                             [4, [(-1,-1)]],
#                             [4, [(-1,-2)]]
#                             ]
#     label_list = [1,2,3,4]
#     distance_mode_list = ['bc_d','cheb_d','cos_d','dtw_d','e_d','em_d','jsd_d','maha_d','manha_d','pcc_d','se_d']
#     for d_m in distance_mode_list:
#         print('Current distance mode {0}'.format(d_m))
#         knn = KNN(K = 4, unlabeled_dataset_list = unlabeled_dataset_list,\
#                   labeled_dataset_list = labeled_dataset_list, distance_mode = d_m, label_list = label_list)
#         sample_label_prob_dict_list = knn.classify()
#         print(sample_label_prob_dict_list)
#         print('---------------------------------')
#         import time
#         time.sleep(600)
# --------------------------- Use manual made data to test KNN ---------------------------