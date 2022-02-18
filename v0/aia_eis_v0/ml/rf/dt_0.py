import sys
import copy
import os
import pickle
import random
import math

from ml.ml_data_wrapper import pack_list_2_list, reform_labeled_dataset_list

def cal_entropy(reformed_labeled_dataset_list):
    """
    :param
        label_count_dict:
            {'label 0' : 8, 'label 1': 3, ...}
    :return:
        the entropy of this node (before any division)
    """
    label_count_dict = {}
    for reformed_labeled_data_list in reformed_labeled_dataset_list:
        if reformed_labeled_data_list[0] not in label_count_dict.keys():
            label_count_dict[reformed_labeled_data_list[0]] = 1
        else:
            label_count_dict[reformed_labeled_data_list[0]] += 1

    entropy = 0.0
    data_amount = len(reformed_labeled_dataset_list)
    for value in label_count_dict.values():
        p = value / data_amount
        entropy += - p * math.log(p, 2)
    return entropy, label_count_dict

def cal_node_accuracy(col_index, T, left_label, right_label, reformed_vali_data_list):
    accuracy = 0
    for v_d_list in reformed_vali_data_list:
        x = v_d_list[1][col_index]
        if (x <= T) and (left_label == v_d_list[0]):
            accuracy += 1
        elif (x > T) and (right_label == v_d_list[0]):
            accuracy += 1
    return accuracy

class Node:
    def __init__(self, reformed_labeled_dataset_list, level, leaf_label=None):
        """
        :param
            reformed_labeled_dataset_list
                [
                    [label1, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                    [label3, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                    [label4, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                    ...
                ]
            level
                int
                record the level in the tree, start from the root (level 0)
            prune_flag
                boolean
                mark the status of this node, if it is pruned, flag = True, else flag = False
            t
                因为属性值为连续属性，需要寻找阈值T，对该属性的范围进行划分
            child_left_node
                属性值 小于 阈值T 的数据 归属到 左侧子分支
            child_right_node
                属性值 大于 阈值T 的数据 归属到 右侧子分支
        """
        self.reformed_labeled_dataset_list = reformed_labeled_dataset_list

        self.prune_flag = False

        self.level = level
        self.leaf_label = leaf_label

        self.child_left_node = None
        self.child_right_node = None

        self.col_index = None
        self.T = None
        self.gain = None

        self.entropy, self.label_count_dict = cal_entropy(self.reformed_labeled_dataset_list)

        # 1-当此节点为叶节点时，无需计算分割数据产生的增益
        # 2-当此节点为叶节点 and 叶节点只有一条数据时，计算分割产生的增益时，会因无处分割而报错
        if type(self.leaf_label) != int:
            self.cal_gain()

    def cal_gain(self):
        # all_col_max_gain_list = [(column_index, threshold T in this column, maximum Gain with this T), ...]
        all_col_max_gain_list = []

        # 遍历样本的每一个属性（列）
        for col_index in range(len(self.reformed_labeled_dataset_list[0][1])):
            col_list = [data_list[1][col_index] for data_list in self.reformed_labeled_dataset_list]
            # 对属性进行排序(reverse = False ==> Ascending)，计算相邻两点间的平均值（可能的阈值T）
            col_list.sort(reverse=False)
            # 找出该连续属性所有可能的分割点（阈值）T
            T_candidate_list = [(col_list[i] + col_list[i+1]) / 2 for i in range(len(col_list) - 1)]
            gain_list = []
            # 遍历每个可能的阈值T
            for T in T_candidate_list:
                left_dataset_list = [data_list for data_list in self.reformed_labeled_dataset_list\
                                                    if data_list[1][col_index] < T]
                right_dataset_list = [data_list for data_list in self.reformed_labeled_dataset_list\
                                                    if data_list[1][col_index] >= T]
                left_entropy, left_label_count_dict = cal_entropy(left_dataset_list)
                right_entropy, right_label_count_dict = cal_entropy(right_dataset_list)
                # 根据阈值T分割后的结果计算增益Gain
                gain = self.entropy \
                                    - len(left_dataset_list) * left_entropy / len(self.reformed_labeled_dataset_list) \
                                    - len(right_dataset_list) * right_entropy / len(self.reformed_labeled_dataset_list)
                gain_list.append(gain)
            max_gain_index = gain_list.index(max(gain_list))
            all_col_max_gain_list.append((col_index, T_candidate_list[max_gain_index], max(gain_list)))
        # 对增益gain list进行排序找出最大的gain及其对应的属性（列）
        # reverse = True 降序， reverse = False 升序（默认）
        all_col_max_gain_list.sort(key=lambda data: data[2], reverse=True)
        self.col_index, self.T, self.gain = all_col_max_gain_list[0]

    def create_child_node(self):
        # 按照T分割数据
        # x < T
        left_dataset_list = [data for data in self.reformed_labeled_dataset_list if data[1][self.col_index] < self.T]
        # x >= T
        right_dataset_list = [data for data in self.reformed_labeled_dataset_list if data[1][self.col_index] >= self.T]

        """
        如果分割所得的子集中：1-子集中数据标签种类是否相同；2-子集样本数量；按照这两条标准区分可得一下四种情况，两种结果：
            label同异         样本数量>1(>=2)         结果
            同(label_num=1)      是(data_amount>1)    叶节点
            同                   否                   叶节点
            -------------------------------------------
            否(label_num>1)      是                   子树的根节点
            否                   否(data_amount=1)    【不存在这种情况】
        """
        left_label_num = len(set([data[0] for data in left_dataset_list]))
        if left_label_num == 1:
            self.child_left_node = Node(left_dataset_list, self.level+1, leaf_label = left_dataset_list[0][0]) # an int
        elif left_label_num > 1:
            self.child_left_node = Node(left_dataset_list, self.level+1)
            self.child_left_node.create_child_node()

        right_label_num = len(set([data[0] for data in right_dataset_list]))
        if right_label_num == 1:
            self.child_right_node = Node(right_dataset_list, self.level+1, leaf_label = right_dataset_list[0][0])
        elif right_label_num > 1:
            """
            当出现数据相同，标签不同的错误数据时，这些数据会被分到right_dataset_list,此时把他们分到一个叶节点里
                训练集中，标签不同，数据相同
                    [4, [0.010820171959196951, 1.0, 0.01630376465973397, 0.9887005649439885, 0.02175795455276722, 0.9830508474904379, 0.04348650644694115, 0.9774011298879769, 0.07609403546858824, 0.9661016948319653, 0.10327677740470584, 0.9548022599248643, 0.1413090938097033, 0.9435028248688527, 0.17391662298026075, 0.9322033898128411, 0.1793708128732941, 0.9265536723592906, 0.18489850941281064, 0.9067796610484979, 0.21208125120001783, 0.8954802259924864, 0.27738451751691284, 0.8559322033709013, 0.23924929180683874, 0.8870056497377053, 0.28279460342205615, 0.8587570621721318, 0.3046995711188796, 0.8192090395505465, 0.31571086035893314, 0.7881355931837424, 0.33211753417483336, 0.7627118644193992, 0.3376011267264599, 0.7514124293633876, 0.26106605152788226, 0.8644067796256824, 0.3594472891060968, 0.723163841797814, 0.36513670026787665, 0.6723163841202172, 0.40340423779271034, 0.6158192089890699, 0.40356595226606323, 0.5847457627711762, 0.4363204947295874, 0.545197740149591, 0.4364822092029403, 0.5141242937827868, 0.46928085550544396, 0.46610169490642056, 0.4748673573621472, 0.4350282485396164, 0.4966988184124874, 0.40960451977527323, 0.49142104432210365, 0.3813559322096996, 0.5024323334132468, 0.35028248584289545, 0.5188831112170371, 0.3163841808237712, 0.5244549117444437, 0.28813559325819754, 0.5245725223788169, 0.2655367231461745, 0.5356426167871468, 0.22316384187226915, 0.54665390587829, 0.19209039550546497, 0.54674211385407, 0.17514124299590292, 0.5577681042745097, 0.14124293782786826, 0.5579298187478627, 0.11016949160997452, 0.5526961486453689, 0.07344632778961978, 0.5582973518313689, 0.03954802262158512, 0.569352744910402, 0.0, 0.585568301594356, 0.011299435056011466, 0.5355838114699601, 0.23446327692828073, 0.5079012247844873, 0.34180790958811436, 0.46935436215192733, 0.45197740119808893, 0.4365557158494236, 0.5000000000744552, 0.4253974134653138, 0.5593220338579226, 0.4143567217155773, 0.5960451976782773, 0.3870122653061068, 0.6384180791010929, 0.37603037887355695, 0.6638418078654361, 0.37578045642442404, 0.7118644067418025, 0.34839189602706355, 0.7627118644193992, 0.32642812301305346, 0.8135593219480854, 0.2882193908054065, 0.8587570621721318, 0.2991571732500663, 0.8418079095136592, 0.2719156261456725, 0.8644067796256824, 0.25012826908322233, 0.8813559321352443, 0.20661235997768793, 0.9039548022472674, 0.1631258535307468, 0.9209039547568296, 0.14676328370273656, 0.9378531072663917, 0.0870024154035652, 0.9548022599248643, 0.07612343812718159, 0.9604519773784149, 0.010937782593570287, 0.9774011298879769, 2.9402658593325923e-05, 0.9887005649439885, 0.0, 0.994350282397539, 0.0326222304998542, 0.9802259886892074, 0.07610873679788491, 0.9632768361796452, 0.14127969115110997, 0.9491525423224033, 0.19022038749108433, 0.9265536723592906, 0.2174325319368848, 0.909604519700818, 0.2610366488692889, 0.8700564972281432, 0.2938499965010893, 0.8192090395505465, 0.3429377061340303, 0.7683615818729498, 0.35935908127922717, 0.7401129943073761, 0.3702527597359971, 0.731638418052595, 0.38673294019838056, 0.6920903954310099, 0.4362175854245107, 0.5649717514603836, 0.4691779462003674, 0.48587570621721315, 0.5023882295742672, 0.35875706209767655, 0.547271361410929, 0.07344632778961978]]
                    [8, [0.010820171959196951, 1.0, 0.01630376465973397, 0.9887005649439885, 0.02175795455276722, 0.9830508474904379, 0.04348650644694115, 0.9774011298879769, 0.07609403546858824, 0.9661016948319653, 0.10327677740470584, 0.9548022599248643, 0.1413090938097033, 0.9435028248688527, 0.17391662298026075, 0.9322033898128411, 0.1793708128732941, 0.9265536723592906, 0.18489850941281064, 0.9067796610484979, 0.21208125120001783, 0.8954802259924864, 0.27738451751691284, 0.8559322033709013, 0.23924929180683874, 0.8870056497377053, 0.28279460342205615, 0.8587570621721318, 0.3046995711188796, 0.8192090395505465, 0.31571086035893314, 0.7881355931837424, 0.33211753417483336, 0.7627118644193992, 0.3376011267264599, 0.7514124293633876, 0.26106605152788226, 0.8644067796256824, 0.3594472891060968, 0.723163841797814, 0.36513670026787665, 0.6723163841202172, 0.40340423779271034, 0.6158192089890699, 0.40356595226606323, 0.5847457627711762, 0.4363204947295874, 0.545197740149591, 0.4364822092029403, 0.5141242937827868, 0.46928085550544396, 0.46610169490642056, 0.4748673573621472, 0.4350282485396164, 0.4966988184124874, 0.40960451977527323, 0.49142104432210365, 0.3813559322096996, 0.5024323334132468, 0.35028248584289545, 0.5188831112170371, 0.3163841808237712, 0.5244549117444437, 0.28813559325819754, 0.5245725223788169, 0.2655367231461745, 0.5356426167871468, 0.22316384187226915, 0.54665390587829, 0.19209039550546497, 0.54674211385407, 0.17514124299590292, 0.5577681042745097, 0.14124293782786826, 0.5579298187478627, 0.11016949160997452, 0.5526961486453689, 0.07344632778961978, 0.5582973518313689, 0.03954802262158512, 0.569352744910402, 0.0, 0.585568301594356, 0.011299435056011466, 0.5355838114699601, 0.23446327692828073, 0.5079012247844873, 0.34180790958811436, 0.46935436215192733, 0.45197740119808893, 0.4365557158494236, 0.5000000000744552, 0.4253974134653138, 0.5593220338579226, 0.4143567217155773, 0.5960451976782773, 0.3870122653061068, 0.6384180791010929, 0.37603037887355695, 0.6638418078654361, 0.37578045642442404, 0.7118644067418025, 0.34839189602706355, 0.7627118644193992, 0.32642812301305346, 0.8135593219480854, 0.2882193908054065, 0.8587570621721318, 0.2991571732500663, 0.8418079095136592, 0.2719156261456725, 0.8644067796256824, 0.25012826908322233, 0.8813559321352443, 0.20661235997768793, 0.9039548022472674, 0.1631258535307468, 0.9209039547568296, 0.14676328370273656, 0.9378531072663917, 0.0870024154035652, 0.9548022599248643, 0.07612343812718159, 0.9604519773784149, 0.010937782593570287, 0.9774011298879769, 2.9402658593325923e-05, 0.9887005649439885, 0.0, 0.994350282397539, 0.0326222304998542, 0.9802259886892074, 0.07610873679788491, 0.9632768361796452, 0.14127969115110997, 0.9491525423224033, 0.19022038749108433, 0.9265536723592906, 0.2174325319368848, 0.909604519700818, 0.2610366488692889, 0.8700564972281432, 0.2938499965010893, 0.8192090395505465, 0.3429377061340303, 0.7683615818729498, 0.35935908127922717, 0.7401129943073761, 0.3702527597359971, 0.731638418052595, 0.38673294019838056, 0.6920903954310099, 0.4362175854245107, 0.5649717514603836, 0.4691779462003674, 0.48587570621721315, 0.5023882295742672, 0.35875706209767655, 0.547271361410929, 0.07344632778961978]]
            """
            if left_label_num == 0:
                leaf_label_set = set([d[0] for d in right_dataset_list])
                left_leaf_label_list = [i for i in leaf_label_set if i != right_dataset_list[0][0]]
                self.child_right_node = Node(right_dataset_list, self.level + 1, leaf_label =right_dataset_list[0][0])
                self.child_left_node = Node(left_dataset_list, self.level + 1, leaf_label = left_leaf_label_list[0])
            else:
                self.child_right_node = Node(right_dataset_list, self.level+1)
                self.child_right_node.create_child_node()

    def get_tree_depth(self, max_level=0):
        self.max_level = max_level
        if isinstance(self.child_left_node.leaf_label, int):
            if self.level > max_level:
                self.max_level = self.level
                return self.max_level
        elif self.child_left_node.leaf_label == None:
            tmp_max_level = self.child_left_node.get_tree_depth(max_level=self.max_level)
            if self.max_level < tmp_max_level:
                self.max_level = tmp_max_level
                return self.max_level

        if isinstance(self.child_right_node.leaf_label, int):
            if self.level > max_level:
                self.max_level = self.level
                return self.max_level
        elif self.child_right_node.leaf_label == None:
            tmp_max_level = self.child_right_node.get_tree_depth(max_level=self.max_level)
            if self.max_level < tmp_max_level:
                self.max_level = tmp_max_level
                return self.max_level

    def root_post_pruning(self, reformed_validation_dataset_list):
        self.get_tree_depth(max_level=0)
        prune_loop_time = pow(2, self.max_level) - 1
        for i in range(prune_loop_time):
            self.post_pruning_1(reformed_validation_dataset_list)

    def post_pruning_1(self, reformed_validation_dataset_list):
        vali_left_dataset_list = [data for data in reformed_validation_dataset_list if data[1][self.col_index] < self.T]
        vali_right_dataset_list = [data for data in reformed_validation_dataset_list if data[1][self.col_index] >= self.T]
        """
        节点的可能性：                 对应处理
            leaf node                不用处理
            left leaf, right leaf    对当前节点剪枝
            left leaf, right node    对右侧子节点剪枝
            left node, right leaf    对左侧子节点剪枝
            left node, right node    对两侧子节点剪枝
        """
        # AttributeError: 'NoneType' object has no attribute 'leaf_label'
        try:
            # left leaf, right leaf    对当前节点剪枝 (条件：1-左侧为叶节点；2-右侧为叶节点；3-未曾修过枝)
            if (not self.prune_flag) and isinstance(self.child_left_node.leaf_label, int) and isinstance(self.child_right_node.leaf_label, int):
                # no pruning: 有叶子节点分支对于验证数据集上的正确率
                old_accuracy = cal_node_accuracy(col_index=self.col_index, T=self.T,\
                                                 left_label=self.child_left_node.leaf_label, \
                                                 right_label=self.child_right_node.leaf_label,\
                                                 reformed_vali_data_list=reformed_validation_dataset_list)
                # pruning: 当前节点原有数据集最多的标签为最终标签，对验证集数据计算正确率
                most_label = max(self.label_count_dict, key=self.label_count_dict.get)
                new_accuracy = sum([1 for data in reformed_validation_dataset_list if data[0] == most_label])
                if new_accuracy >= old_accuracy:
                    self.child_left_node = None
                    self.child_right_node = None
                    self.leaf_label = most_label
                    self.prune_flag = True
                    return
                else:
                    self.prune_flag = True
        except AttributeError as e:
            print('Current Node: prune_flag {0}, leaf_label {1}'.format(self.prune_flag, self.leaf_label))
            print('left node: type {0}, content {1}'.format(type(self.child_left_node), self.child_left_node))
            print('right node: type {0}, content {1}'.format(type(self.child_right_node), self.child_right_node))
            print(e)
        # left leaf, right node    对右侧子节点剪枝
        if (not self.prune_flag) and isinstance(self.child_left_node.leaf_label, int) and isinstance(self.child_right_node, Node):
            self.child_right_node.post_pruning_1(vali_right_dataset_list)
            return
            # left node, right leaf    对左侧子节点剪枝
        if (not self.prune_flag) and isinstance(self.child_left_node, Node) and isinstance(self.child_right_node.leaf_label, int):
            self.child_left_node.post_pruning_1(vali_left_dataset_list)
            return
            # left node, right node    对两侧子节点剪枝
        if (not self.prune_flag) and isinstance(self.child_left_node, Node) and isinstance(self.child_right_node, Node):
            if self.child_left_node.prune_flag == False:
                self.child_left_node.post_pruning_1(vali_left_dataset_list)
            if self.child_right_node.prune_flag == False:
                self.child_right_node.post_pruning_1(vali_right_dataset_list)
            return

    def classify(self, unlabeled_data_list):
        """
        :param
            unlabeled_data_list:
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                不选下方的数据格式：每次递归调用都要转换一下，浪费时间
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        :return:
            label
        """
        # 判断当前节点是否为叶节点
        x = unlabeled_data_list[self.col_index]
        # 子树节点+小于阈值
        if x < self.T:
            # 叶节点
            if isinstance(self.child_left_node.leaf_label, int):
                return self.child_left_node.leaf_label
            elif isinstance(self.child_left_node, Node):
                return self.child_left_node.classify(unlabeled_data_list)
        # 子树节点+大于阈值
        elif x >= self.T:
            # 叶节点
            if isinstance(self.child_right_node.leaf_label, int):
                return self.child_right_node.leaf_label
            elif isinstance(self.child_right_node, Node):
                return self.child_right_node.classify(unlabeled_data_list)

def save_node(node, file_name='node_pickle.file', file_path='./'):
    file_abs_path = os.path.join(file_path, file_name)
    with open(file_abs_path, 'wb') as file:
        pickle.dump(node, file)

def load_node(file_name='node_pickle.file', file_path='./'):
    file_abs_path = os.path.join(file_path, file_name)
    with open(file_abs_path, 'rb') as file:
        node = pickle.load(file)
    return node

# ---------------------------- Test of decision tree ----------------------------
# if __name__ == '__main__':
#     labeled_data_list = [
#         # 5 行 1
#         [1, [(1,1) for i in range(4)]],
#         [1, [(2,2) for i in range(4)]],
#         [1, [(3,3) for i in range(4)]],
#         [1, [(4,4) for i in range(4)]],
#         [1, [(5,5) for i in range(4)]],
#         # 6 行 2
#         [2, [(6,6) for i in range(4)]],
#         [2, [(7,7) for i in range(4)]],
#         [2, [(8,8) for i in range(4)]],
#         [2, [(9,9) for i in range(4)]],
#         [2, [(10,10) for i in range(4)]],
#         [2, [(11,11) for i in range(4)]],
#         # 7 行 3
#         [3, [(12,12) for i in range(4)]],
#         [3, [(13,13) for i in range(4)]],
#         [3, [(14,14) for i in range(4)]],
#         [3, [(15,15) for i in range(4)]],
#         [3, [(16,16) for i in range(4)]],
#         [3, [(17,17) for i in range(4)]],
#         [3, [(18,18) for i in range(4)]],
#     ]
#     reformed_labeled_data_list = reform_labeled_dataset_list(labeled_data_list)
#     node = Node(reformed_labeled_data_list, level=0)
#     node.create_child_node()
#
#     # max_level = node.get_tree_depth(max_level=0)
#     # print(max_level)
#
#     # 此验证数据集为刻意设置为正确率为0（无论是否剪枝），但是根据奥卡姆剃刀原则，结果一样，越简单（剪枝）越好
#     vali_data_list = [
#         [3, [(8, 8) for i in range(4)]],
#         [3, [(9, 9) for i in range(4)]],
#         [3, [(10, 10) for i in range(4)]],
#         [2, [(11, 11) for i in range(4)]],
#         [2, [(12, 12) for i in range(4)]],
#         [2, [(13, 13) for i in range(4)]],
#     ]
#     reformed_vali_data_list = reform_labeled_dataset_list(vali_data_list)
#     node.post_pruning_1(reformed_vali_data_list)
#
#     test_unlabeled_data = [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5]
#     label = node.classify(test_unlabeled_data)
#     print('label :',label)
# ---------------------------- Test of decision tree ----------------------------

def data_duplicate_checker(data_list):
    """
    Function: 当子树被分配的数据集中含有两个及以上的标签种类，此时要再查看每条数据是否相同
        标签不同，数据相同：
            数据标签错误，将这批数据创建在一个叶节点中，标签以多数标签为准
            返回 False
        标签不同，数据不同：
            正常情况，继续分割数据集，创建子树
            返回 True
    :param:
        label_num, int, Number of label type
        data_list = [
                        [label1(int), [num,num,num,num,]]
                        [label1(int), [num,num,num,num,]]
                        [label2(int), [num,num,num,num,]]
                        [label2(int), [num,num,num,num,]]
                        [label2(int), [num,num,num,num,]]
                        [label3(int), [num,num,num,num,]]
                        ...
                        [label2(int), [num,num,num,num,]]
                    ]
    :return:
    """
    label_count_dict = {}

    existed_data_set = set()
    for d in data_list:
        label = d[0]
        if label not in label_count_dict.keys():
            label_count_dict[label] = 1
        else:
            label_count_dict[label] += 1

        num_list = d[1]
        num_tuple = tuple(num_list)
        if num_tuple not in existed_data_set:
            existed_data_set.add(num_tuple)
    # Have duplication: labels are different, but data are the same
    if len(existed_data_set) == 1:
        selected_label = None
        for k, v in label_count_dict.items():
            if v == max(label_count_dict.values()):
                selected_label = k
                break
        return False, selected_label
    # No duplication: labels are different, but data are different too.
    elif len(existed_data_set) > 1:
        return True, None

class Random_Tree:
    def __init__(self, reformed_labeled_dataset_list, leaf_label=None):
        """
        :param
            reformed_labeled_dataset_list
                [
                    [label1, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                    [label3, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                    [label4, [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]]
                    ...
                ]
            level
                int
                record the level in the tree, start from the root (level 0)
            prune_flag
                boolean
                mark the status of this node, if it is pruned, flag = True, else flag = False
            t
                因为属性值为连续属性，需要寻找阈值T，对该属性的范围进行划分
            child_left_node
                属性值 小于 阈值T 的数据 归属到 左侧子分支
            child_right_node
                属性值 大于 阈值T 的数据 归属到 右侧子分支
        """
        self.reformed_labeled_dataset_list = reformed_labeled_dataset_list
        self.leaf_label = leaf_label

        self.child_left_node = None
        self.child_right_node = None

        self.col_index = None
        self.T = None
        self.gain = None

        # 样本属性数量
        try:
            # IndexError: list index out of range
            self.attribute_num = len(self.reformed_labeled_dataset_list[0][1])
        except IndexError as e:
            print(e)
            # sys.exit(1)
        # 在每一节点选用K个属性比较信息增益，选择增益最大的属性进行划分
        self.k = int(math.log(self.attribute_num, 2) + 1)

        self.entropy, self.label_count_dict = cal_entropy(self.reformed_labeled_dataset_list)

        if type(self.leaf_label) != int:
            self.cal_gain()

    def cal_gain(self):
        # 随机选取K个属性,用随机数发生器可能会选重复
        self.random_attribute_index = []
        # -------- This chunk of code selects duplicated attributes, and is wrong -----------
        # while len(set(self.random_attribute_index)) < self.k:
        #     r_a_i = int(random.uniform(0, len(self.reformed_labeled_dataset_list[0][1])))
        #     self.random_attribute_index.append(r_a_i)
        # -------- This chunk of code selects duplicated attributes, and is wrong -----------

        # -------- This chunk of code selects unduplicated attributes, and is right --------
        while len(self.random_attribute_index) < self.k:
            r_a_i = int(random.uniform(0, len(self.reformed_labeled_dataset_list[0][1])))
            if r_a_i not in self.random_attribute_index:
                self.random_attribute_index.append(r_a_i)

        # all_col_max_gain_list = [(column_index, threshold T in this column, maximum Gain with this T), ...]
        all_col_max_gain_list = []
        # 遍历样本的每一个属性（列）
        for col_index in self.random_attribute_index:
            col_list = [data_list[1][col_index] for data_list in self.reformed_labeled_dataset_list]
            """
            一列中有重复数字出现
            1- 随机树的数据来源于重采样，一批样本中会有重复的数据出现， 每一列中也可能会出现重复的数字，在分割阈值的时候就会出现该数字，
                此处应将重复的数字删除
            2- 在EIS数据的末尾，基本上数据最大值的地方，有较大的概率出现【在一个节点上取到最后几列158/159等，数字很可能均为1;或者在在一个节点渠道最初的几列，数字很可能均为0.0】，
                此时col_unique_list = [] 
            """
            col_unique_list = list(set(col_list))
            if len(col_unique_list) < 2:
                continue
            # 对属性进行排序(reverse=False==>Ascending)，计算相邻两点间的平均值（可能的阈值T）
            col_unique_list.sort(reverse=False)
            # 找出该连续属性所有可能的分割点（阈值）T
            T_candidate_list = [(col_unique_list[i] + col_unique_list[i+1]) / 2 for i in range(len(col_unique_list) - 1)]
            gain_list = []
            # 遍历每个可能的阈值T
            for T in T_candidate_list:
                left_dataset_list = [data_list for data_list in self.reformed_labeled_dataset_list if data_list[1][col_index] <= T]
                right_dataset_list = [data_list for data_list in self.reformed_labeled_dataset_list if data_list[1][col_index] > T]
                left_entropy, left_label_count_dict = cal_entropy(left_dataset_list)
                right_entropy, right_label_count_dict = cal_entropy(right_dataset_list)
                # 根据阈值T分割后的结果计算增益Gain
                gain = self.entropy - len(left_dataset_list) * left_entropy / len(self.reformed_labeled_dataset_list) - len(right_dataset_list) * right_entropy / len(self.reformed_labeled_dataset_list)
                gain_list.append(gain)
            try:
                # ValueError: max() arg is an empty sequence
                max_gain_index = gain_list.index(max(gain_list))
            except ValueError as e:
                print(e)
                sys.exit(1)
            all_col_max_gain_list.append((col_index, T_candidate_list[max_gain_index], max(gain_list)))

        # 对增益gain list进行排序找出最大的gain及其对应的属性（列）
        all_col_max_gain_list.sort(key=lambda data: data[2], reverse = True)
        try:
            self.col_index, self.T, self.gain = all_col_max_gain_list[0]
        except IndexError as e:
            label_count_dict = {}
            for d in self.reformed_labeled_dataset_list:
                label = d[0]
                if label not in label_count_dict.keys():
                    label_count_dict[label] = 1
                else:
                    label_count_dict[label] += 1
            for k, v in label_count_dict.items():
                if v == max(label_count_dict.values()):
                    self.leaf_label = k
            print(e)
            # sys.exit(1)

    def create_child_node(self):
        # 按照T分割数据
        # x < T
        # TypeError: list indices must be integers or slices, not NoneType
        try:
            left_dataset_list = [data for data in self.reformed_labeled_dataset_list if data[1][self.col_index] < self.T]
        except TypeError as e:
            print(e)
            sys.exit(1)
        # x >= T
        try:
            right_dataset_list = [data for data in self.reformed_labeled_dataset_list if data[1][self.col_index] >= self.T]
        except TypeError as e:
            print(e)
            sys.exit(1)

        """
        如果分割所得的子集中：1-子集中数据标签种类是否相同；2-子集样本数量；按照这两条标准区分可得一下四种情况，两种结果：
            label同异         样本数量>1(>=2)         结果
            同(label_num=1)      是(data_amount>1)    叶节点               
            同                   否                   叶节点
            -------------------------------------------
            否(label_num>1)      是                   子树的根节点
            否                   否(data_amount=1)    【不存在这种情况】
        """
        left_label_num = len(set([data[0] for data in left_dataset_list]))
        if left_label_num == 1:
            self.child_left_node = Random_Tree(left_dataset_list, leaf_label=left_dataset_list[0][0]) # an int
        elif left_label_num > 1:
            checker, selected_label = data_duplicate_checker(left_dataset_list)
            if checker:
                self.child_left_node = Random_Tree(left_dataset_list)
                self.child_left_node.create_child_node()
            else:
                self.child_left_node = Random_Tree(left_dataset_list, leaf_label=selected_label)

        right_label_num = len(set([data[0] for data in right_dataset_list]))
        if right_label_num == 1:
            self.child_right_node = Random_Tree(right_dataset_list, leaf_label=right_dataset_list[0][0])
        elif right_label_num > 1:
            # if left_label_num == 0:
            #     leaf_label_set = set([d[0] for d in right_dataset_list])
            #     left_leaf_label_list = [i for i in leaf_label_set if i != right_dataset_list[0][0]]
            #     self.child_right_node = Random_Tree(right_dataset_list, leaf_label =right_dataset_list[0][0])
            #     self.child_left_node = Random_Tree(left_dataset_list, leaf_label = left_leaf_label_list[0])
            # else:
            #     self.child_right_node = Random_Tree(right_dataset_list)
            #     if type(self.child_right_node.leaf_label) != int:
            #         self.child_right_node.create_child_node()
            checker, selected_label = data_duplicate_checker(right_dataset_list)
            if checker:
                self.child_right_node = Random_Tree(right_dataset_list)
                self.child_right_node.create_child_node()
            else:
                self.child_right_node = Random_Tree(right_dataset_list, leaf_label=selected_label)

    def classify(self, unlabeled_data_list):
        """
        :param
            unlabeled_data_list:
                [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                不选下方的数据格式：每次递归调用都要转换一下，浪费时间
                [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]
        :return:
            label
        """
        # 判断当前节点是否为叶节点
        x = unlabeled_data_list[self.col_index]
        # 子树节点+小于阈值
        if x < self.T:
            # 叶节点
            try:
                # AttributeError: 'NoneType' object has no attribute 'leaf_label'
                if isinstance(self.child_left_node.leaf_label, int):
                    return self.child_left_node.leaf_label
                elif isinstance(self.child_left_node, Random_Tree):
                    return self.child_left_node.classify(unlabeled_data_list)
            except AttributeError as e:
                print('Leaf label:',self.leaf_label)
                print('child_left_node', self.child_left_node)
                print('child_right_node', self.child_right_node)
                sys.exit(1)
        # 子树节点+大于阈值
        elif x >= self.T:
            # 叶节点
            if isinstance(self.child_right_node.leaf_label, int):
                return self.child_right_node.leaf_label
            elif isinstance(self.child_right_node, Random_Tree):
                return self.child_right_node.classify(unlabeled_data_list)

# ---------------------------- Test of Random decision tree ----------------------------
# if __name__ == '__main__':
#     labeled_data_list = [
#         # 5 行 1
#         [1, [(1,1) for i in range(4)]],
#         [1, [(2,2) for i in range(4)]],
#         [1, [(3,3) for i in range(4)]],
#         [1, [(4,4) for i in range(4)]],
#         [1, [(5,5) for i in range(4)]],
#         # 6 行 2
#         [2, [(6,6) for i in range(4)]],
#         [2, [(7,7) for i in range(4)]],
#         [2, [(8,8) for i in range(4)]],
#         [2, [(9,9) for i in range(4)]],
#         [2, [(10,10) for i in range(4)]],
#         [2, [(11,11) for i in range(4)]],
#         # 7 行 3
#         [3, [(12,12) for i in range(4)]],
#         [3, [(13,13) for i in range(4)]],
#         [3, [(14,14) for i in range(4)]],
#         [3, [(15,15) for i in range(4)]],
#         [3, [(16,16) for i in range(4)]],
#         [3, [(17,17) for i in range(4)]],
#         [3, [(18,18) for i in range(4)]],
#     ]
#     reformed_labeled_data_list = reform_labeled_dataset_list(labeled_data_list)
#     rt = Random_Tree(reformed_labeled_data_list)
#     rt.create_child_node()
#
#     test_unlabeled_data = [6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5, 6.5]
#     label = rt.classify(test_unlabeled_data)
#     print('label :',label)
# ---------------------------- Test of Random decision tree ----------------------------