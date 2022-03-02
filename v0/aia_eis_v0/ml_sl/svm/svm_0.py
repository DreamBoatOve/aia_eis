import sys
import copy
import numpy as np

from ml_sl.ml_data_wrapper import labeled_dataset_2_data_and_label_arr

class SVM:
    """
    SVM
        Support_Vector_Machine
            SMO (Sequential minimal optimization)
        refer:
            paper
                Fast Training of Support Vector Machines using Sequential Minimal Optimization, John C. Platt
            web
                机器学习算法实践-SVM中的SMO算法
                https://zhuanlan.zhihu.com/p/29212107
    """
    def __init__(self, labeled_dataset_list, kernel_paras_dict, C, tol, max_iter):
        """
        :param
            labeled_dataset_list:
                [
                    [label number, points list]
                    [1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    [-1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    [1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    [1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    [-1, [(x0, y0), (x1, y1), (x2, y2), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                    ...
                ]
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
            data_arr
                m * n
                [
                    [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                    [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                    [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                    [x0, y0, x1, y1, x2, y2, ..., xn-2, yn-2, xn-1, yn-1]
                    ...
                ]
            label_arr
                1 * m
                [1, -1, 1, -1, 1, ...]
            kernel_para and kernel_fun:
                                    kernel_para         kernel_fun
                for linear kernel   None                linear
                for poly kernel     [power,  ζ, r]      poly
                for RBF kernel      sigma               RBF
            C
                支持向量的权重与越界数据之间的平衡因子
            tol
                default 10^(-2)
                计算数据点与超平面之间的距离时，允许有微小的误差tol
                tol 在Platt的SMO论文中建议取值10^(-2) ~ 10^(-3)，取值越小，算法收敛越慢
            max_iter
                max iteration time
        """
        self.labeled_dataset_list = labeled_dataset_list
        self.data_arr, self.label_arr = labeled_dataset_2_data_and_label_arr(labeled_dataset_list)

        self.kernel_paras_dict = kernel_paras_dict
        self.kernel_para, self.kernel_fun = self.get_kernel()

        self.row_num, self.col_num = self.data_arr.shape

        self.C = C
        self.tol = tol
        # Initialize all the lagrange multipliers, alpha_arr
        # np.random() 返回随机的浮点数，在半开区间 [0.0, 1.0)
        self.alpha_arr = self.C * np.random.random(self.row_num)
        self.b = np.random.random()

        self.max_iter = max_iter

    def KKT_list_check(self, kkt_list):
        for kkt in kkt_list:
            if kkt:
                continue
            else:
                return False
        return True

    def KKT(self, alpha, label, x_arr):
        f = self.get_prediction(x_arr)
        # yi * f(i) >= 1 and alpha = 0 (outside the boundary)
        if alpha == 0:
            if label * f >= (1 - self.tol):
                return True
            else:
                return False
        # yi * f(i) == 1 and 0 < alpha < C (on the boundary)
        elif (0 < alpha) and (alpha < self.C):
            if abs(label * f - 1) <= self.tol:
                return True
            else:
                return False
        # yi * f(i) <= 1 and alpha = C (between the boundary)
        elif alpha == self.C:
            if label * f <= 1 + self.tol:
                return True
            else:
                return False
        else:
            print('Alpha {0} is beyond the scope 0 ~ {1}'.format(alpha, self.C))
            sys.exit(1)

    def select_J_0(self, i):
        """
        The max difference between data_i.error and data_j.error decides the Index J
        find the samples with the biggest or smallest error, one of the samples is sample_J
        :param i:
        :return:
        """

        Ei = self.E_list[i]
        max_E_diff = -100

        final_J = -1

        index_list = list(range(self.row_num))
        index_list.remove(i)

        for j in index_list:
            Ej = self.E_list[j]
            tmp_E_diff = abs(Ei - Ej)
            if tmp_E_diff > max_E_diff:
                max_E_diff = tmp_E_diff
                final_J = j
        return final_J

    def select_J_1(self, i, non_bound_index_list=None):
        """
        pseudo-code
            The max difference between data_i.error and data_j.error decides the Index J
            find the samples with the biggest or smallest error, one of the samples is sample_J
            sample_J should be selected from non-bound samples
            Routine:
                1-find all the index of non-bound samples, non_bound_index_list
                2-Exclude i from non_bound_index_list
                3-Search the biggest and smallest Error
        :param
            i: index of sample_i
        :return:
        """
        if non_bound_index_list == None:
            non_bound_index_list = [a for a, alpha in enumerate(self.alpha_arr) if (0 < alpha < self.C) and (a != i)]
        if len(non_bound_index_list) == 0:
            return None
        non_bound_E_list = [self.E_list[a] for a in non_bound_index_list]
        # min_E_index = self.E_list.index(min(self.E_list))
        # max_E_index = self.E_list.index(max(self.E_list))
        min_E_index = non_bound_E_list.index(min(non_bound_E_list))
        max_E_index = non_bound_E_list.index(max(non_bound_E_list))
        E_i = self.E_list[i]
        if abs(E_i - min(non_bound_E_list)) >= abs(E_i - max(non_bound_E_list)):
            return non_bound_index_list[min_E_index]
        else:
            return non_bound_index_list[max_E_index]

    def clip_alpha(self, i, j, alpha):
        alpha_diff = self.alpha_arr[j] - self.alpha_arr[i]
        alpha_sum = self.alpha_arr[j] + self.alpha_arr[i]

        # yi != yj
        if self.label_arr[i] != self.label_arr[j]:
            H = min(self.C, self.C + alpha_diff)
            L = max(0, alpha_diff)
        # yi == yj
        else:
            H = min(self.C, alpha_sum)
            L = max(0, alpha_sum - self.C)
        if alpha < L:
            alpha = L
        elif alpha > H:
            alpha = H
        return alpha

    def update_paras(self, i, j):
        """
        function:
            update alpha_i/j
            update b
        :param
            i, j:
                index of sample
        :return:
        """
        # ------------- calculate new alpha_i/j -------------
        i_arr = self.data_arr[i]
        j_arr = self.data_arr[j]
        n = self.kernel_fun(self.kernel_para, i_arr, i_arr) + self.kernel_fun(self.kernel_para, j_arr, j_arr)\
            - 2 * self.kernel_fun(self.kernel_para, i_arr, j_arr)

        if abs(n) < 1e-25:
            print('i={0}, j={1}'.format(i, j))
            print('n={0} is too small'.format(n))
            sys.exit(0)
        alpha_i_0 = self.alpha_arr[i]
        alpha_j_0 = self.alpha_arr[j]

        # copy.deepcopy(0.0) ==> int 0
        alpha_j_1 = copy.deepcopy(alpha_j_0) * 1.0

        # print(self.label_arr[j] * (self.E_list[i] - self.E_list[j]) / n)
        alpha_j_1 = alpha_j_1 + self.label_arr[j] * (self.E_list[i] - self.E_list[j]) / n * 1.0
        # alpha_j_1的上下限被修建后不会太大或太小，不会计算上溢或下溢
        alpha_j_1 = self.clip_alpha(i, j, alpha_j_1)

        # alpha_i_1的计算可能出现上溢或下溢，且下溢的可能性更高
        alpha_i_1 = alpha_i_0 + self.label_arr[i] * self.label_arr[j] * (alpha_j_0 - alpha_j_1) * 1.0
        # alpha_j_new 被清晰的限制在【0 ~ C】之间，但是计算alpha_i_new的时候，alpha_i_new可能会很小的超出【0 ~ C】，
        # 在此处模仿paper Fast Training of Support Vector Machines using Sequential Minimal Optimization 12.3 Pseudo-Code中的处理
        if alpha_i_1 < 1e-10:
            alpha_i_1 = 0.0
        elif alpha_i_1 > self.C - 1e-10:
            alpha_i_1 = self.C
        # ------------- calculate new alpha_i/j -------------

        # ------------- update alpha_i/j -------------
        self.alpha_arr[i] = alpha_i_1
        self.alpha_arr[j] = alpha_j_1
        # ------------- update alpha_i/j -------------

        # This is doable, but can be expressed in a more concise way
        # bi = self.label_arr[i] - (alpha_i_1 - alpha_i_0) * self.label_arr[i] * self.kernel_fun(i_arr, i_arr)\
        #                        - (alpha_j_1 - alpha_j_0) * self.label_arr[j] * self.kernel_fun(i_arr, j_arr)\
        #                        - np.dot(np.multiply(self.label_arr, self.alpha_arr), self.kernel_fun(self.data_arr, i_arr))
        bi = - self.E_list[i] + (alpha_i_0 - alpha_i_1) * self.label_arr[i] * self.kernel_fun(self.kernel_para, i_arr, i_arr) \
                              + (alpha_j_0 - alpha_j_1) * self.label_arr[j] * self.kernel_fun(self.kernel_para, i_arr, j_arr) + self.b
        bj = - self.E_list[j] + (alpha_j_0 - alpha_j_1) * self.label_arr[j] * self.kernel_fun(self.kernel_para, j_arr, j_arr) \
                              + (alpha_i_0 - alpha_i_1) * self.label_arr[i] * self.kernel_fun(self.kernel_para, i_arr, j_arr) + self.b
        """
        if 0 < new_alpha_i/j < C, means that sample_i/j is an support vector, it obeys the equation: y * (W * X + b) = 1
            with y, W and X, b can be got by the equation,
        else:
            take the average of bi and bj as the new b
        """
        # update b
        if (0 < alpha_i_1) and (alpha_i_1 < self.C):
            self.b = bi
        elif (0 < alpha_j_1) and (alpha_j_1 < self.C):
            self.b = bj
        else:
            self.b = (bi + bj) / 2

        self.KKT_list = [self.KKT(self.alpha_arr[i], self.label_arr[i], self.data_arr[i]) for i in range(self.row_num)]
        self.E_list = [self.get_E(i) for i in range(self.row_num)]

    # def smo_0(self):
    #     """
    #     pseudo-code for the overall SMO algorithm
    #         1-Initialize the Error (f(x) - y) for each sample, E_list
    #         2-Iterate over the whole dataset to find non-bound samples, S_non_bound whose alpha is not 0 or C,
    #           check whether each of S_non_bound obey KKT
    #             if one sample in S_non_bound violate KKT, it is the first sample, S1,
    #             then select another sample, S2, by measuring the biggest difference between S1.err and S2.err
    #                 find the sample with the biggest or smallest error, one of them might be S2, see function
    #             calculate alpha_2_new = alpha_2_old + y2 * (S1.err - S2.err) / n
    #                 n = Kernel_fun(S1, S1) + Kernel_fun(S2, S2) - 2 * Kernel_fun(S1, S2)
    #             clip alpha_2_new to get alpha_2_new_clip
    #             calculate alhpa_1_new by : alpha_1_new * y1 + alpha_2_new * y2 = alpha_1_old * y1 + alpha_2_old * y2
    #             calculate b
    #     :param:
    #         loop_all_flag, bool
    #
    #     :return:
    #     """
    #     iter_time = 0
    #     # Iterate over the whole dataset
    #     loop_all_flag = True
    #     alpha_changed_count = 0
    #     self.KKT_list = [self.KKT(self.alpha_arr[i], self.label_arr[i], self.data_arr[i]) for i in range(self.row_num)]
    #     self.E_list = [self.get_E(i) for i in range(self.row_num)]
    #     while (iter_time < self.max_iter) and (loop_all_flag or (alpha_changed_count > 0)):
    #         alpha_changed_count = 0
    #
    #         if loop_all_flag:
    #             # 1-遍历 non-bound samples == 0 < alpha < C
    #             # 2-是否违反KKT条件
    #             for i in range(self.row_num):
    #             # for i, kkt in enumerate(self.KKT_list):
    #                 if not self.KKT_list[i]:
    #                     # choose for alpha_i, the first KKT violation is alpha_i
    #                     j = self.select_J_1(i)
    #                     self.update_paras(i, j)
    #                     alpha_changed_count += 1
    #         # Loop over in-bound instances
    #         else:
    #             # find the instances with inbound alpha
    #             inbound_data_index_list = [i for i in range(self.row_num) if (0 < self.alpha_arr[i]) and (self.alpha_arr[i] < self.C)]
    #             for i in inbound_data_index_list:
    #                 j = self.select_J_1(i)
    #                 self.update_paras(i, j)
    #                 alpha_changed_count += 1
    #
    #         if loop_all_flag:
    #             loop_all_flag = False
    #         if alpha_changed_count == 0:
    #             loop_all_flag = True
    #         if self.KKT_list_check(self.KKT_list):
    #             return self.alpha_arr, self.b
    #         iter_time += 1
    #     return self.alpha_arr, self.b

    def smo_1(self):
        """
        pseudo-code for the overall SMO algorithm
            1-Initialize the Error (f(x) - y) for each sample, E_list
            2-Iterate over the whole dataset to find non-bound samples, S_non_bound whose alpha is not 0 or C,
              check whether each of S_non_bound obey KKT
                if one sample in S_non_bound violate KKT, it is the first sample, S1,
                then select another sample, S2, by measuring the biggest difference between S1.err and S2.err
                    find the sample with the biggest or smallest error, one of them might be S2, see function
                calculate alpha_2_new = alpha_2_old + y2 * (S1.err - S2.err) / n
                    n = Kernel_fun(S1, S1) + Kernel_fun(S2, S2) - 2 * Kernel_fun(S1, S2)
                clip alpha_2_new to get alpha_2_new_clip
                calculate alhpa_1_new by : alpha_1_new * y1 + alpha_2_new * y2 = alpha_1_old * y1 + alpha_2_old * y2
                calculate b
        :param:
            loop_all_flag, bool

        :return:
        """
        iter_time = 0
        # Iterate over the whole dataset
        loop_all_flag = True
        alpha_changed_count = 0
        self.KKT_list = [self.KKT(self.alpha_arr[i], self.label_arr[i], self.data_arr[i]) for i in range(self.row_num)]
        self.E_list = [self.get_E(i) for i in range(self.row_num)]
        while (iter_time < self.max_iter) and (loop_all_flag or (alpha_changed_count > 0)):
            alpha_changed_count = 0

            if loop_all_flag:
                for i in range(self.row_num):
                    # 1-遍历 non-bound samples == 0 < alpha < C
                    # 2-是否违反KKT条件
                    # if (0 < self.alpha_arr[i] < self.C) and (not self.KKT_list[i]):
                    if not self.KKT_list[i]:
                        # choose for alpha_i, the first KKT violation is alpha_i
                        j = self.select_J_1(i)
                        if j == None:
                            continue
                        else:
                            self.update_paras(i, j)
                            alpha_changed_count += 1
            # Loop over non-bound instances
            else:
                # After one pass through the training set, the outer loop iterates over only those examples
                # whose Lagrange multipliers are neither 0 nor C (the non- bound examples).
                non_bound_data_index_list = [i for i in range(self.row_num) if (0 < self.alpha_arr[i]) and (self.alpha_arr[i] < self.C)]
                for non_bound_data_index in non_bound_data_index_list:
                    # Again, each example is checked against the KKT conditions
                    if not self.KKT_list[non_bound_data_index]:
                        # and violating examples are eligible for immediate optimization and update.
                        j = self.select_J_1(non_bound_data_index)
                        if j == None:
                            continue
                        else:
                            self.update_paras(non_bound_data_index, j)
                            alpha_changed_count += 1
            if loop_all_flag:
                loop_all_flag = False
            if alpha_changed_count == 0:
                loop_all_flag = True
            if self.KKT_list_check(self.KKT_list):
                # return self.alpha_arr, self.b
                return
            iter_time += 1
        # return self.alpha_arr, self.b
        return

    def get_kernel(self):
        kernel_type = self.kernel_paras_dict['type']
        if kernel_type == 'linear':
            null_para = None
            return null_para, self.linear_kernel
        elif kernel_type == 'poly':
            paras_list = self.kernel_paras_dict['paras']
            return paras_list, self.poly_kernel
        elif kernel_type == 'rbf':
            sigma = self.kernel_paras_dict['paras']
            return sigma, self.rbf_kernel

    def linear_kernel(self, null_para, data_arr, x2_arr):
        """
        :param
            null_para:
            x1_arr:
                one dimension np.array
            x2_arr:
                one dimension np.array
        :return:
        """
        # 为了让三种核函数的输入参数形式统一，在此设立null_para占用一个位置
        return np.dot(data_arr, x2_arr)

    def poly_kernel(self, paras_list, data_arr, x2_arr):
        """
        function
            多项式核函数
            参照 台湾 林轩田 203_handout, page 9/22
            K2(X, X') = (C + r X^T X')^2
            C: constant
            r: Quadratic coefficient
        :param
            paras_list:
                [power, constant, qua_coe]
                power
                    多项式的指数，几次方的多项式
                    K2 is commonly used
                constant,
                    常数项
                qua_coe
                    coe, short for coefficient(系数)
                    X.T * X前的系数
            data_arr:
                the vector of all samples
                or the vector of sample1
            x2_arr:
                the vector of sample2
        :return:
        """
        power, constant, qua_coe = paras_list
        return (constant + np.dot(data_arr, x2_arr)) ** power

    def rbf_kernel(self, sigma, data_arr, x2_arr):
        """
        function
            高斯核函数
            参照 台湾 林轩田 203_handout, page 12-13/22
        :param
            sigma:
                exp(- sigma * ||x1 - x2||^2)
            data_arr:
                the vector of all samples
                or the vector of sample1
            x2_arr:
                the vector of sample2
        :return:
        """
        """
        data_arr - x2_arr
            if data_arr(100 * 2 dimension) and x2_arr(1 * 2 dimension)
                data_arr - x2_arr, means each row in data_arr minus x2_arr separately(element-wise)
        """
        differ_arr = data_arr - x2_arr
        # 当data_arr是一条数据的情况
        if data_arr.ndim == 1:
            return np.exp(- sigma * np.dot(differ_arr, differ_arr) )
        # 当data_arr是整个训练数据集，多条数据的情况
        elif data_arr.ndim == 2:
            return np.exp(- sigma * np.sum(a=np.multiply(differ_arr, differ_arr), axis=1) )

    def get_prediction(self, x_arr):
        # Get precise calculation of f(x) (float)
        # multiply 单个元素相乘； dot 二维矩阵相乘，一维数组单个元素相乘
        # print(np.multiply(self.alpha_arr, self.label_arr))
        # print(self.kernel_fun(self.kernel_para, self.data_arr, x_arr))
        f = np.dot(np.multiply(self.alpha_arr, self.label_arr), self.kernel_fun(self.kernel_para, self.data_arr, x_arr)) + self.b
        return f

    def get_E(self, data_index):
        # Error = f(Xi) - yi
        E = self.get_prediction(self.data_arr[data_index]) - self.label_arr[data_index]
        return E

    def classify(self, x_arr):
        f = self.get_prediction(x_arr)
        label = np.sign(f).astype(int)
        return label

def save_svm():
    pass

def load_svm():
    pass
# ----------------------- Test linear kernel -----------------------
# if __name__ == '__main__':
#     labeled_dataset_list = [
#                             [-1, [(0,-1)]],
#                             [-1, [(1,0)]],
#                             [-1, [(2,1)]],
#                             [-1, [(3,2)]],
#                             [-1, [(4,3)]],
#                             [1, [(0,1)]],
#                              [1, [(1,2)]],
#                              [1, [(2,3)]],
#                              [1, [(3,4)]],
#                              [1, [(4,5)]]
#                            ]
#     # Ideal seperating plane: y = x
#     kernel_paras_dict = {'type': 'linear'}
#     svm = SVM(labeled_dataset_list, kernel_paras_dict, C=0.6, err=0.001, max_iter=100)
#     svm.smo()
#     d1 = np.array([5, 4]) # label = -1
#     d2 = np.array([5, 6]) # label = 1
#     p1 = svm.classify(d1)
#     p2 = svm.classify(d2)
#     print('prediction for d1 :', p1)
#     print('prediction for d2 :', p2)
# ----------------------- Test linear kernel -----------------------

# ----------------------- Test Poly kernel -----------------------
# ----------------------- Test Poly kernel -----------------------

# ----------------------- Test RBF kernel -----------------------
# ----------------------- Test RBF kernel -----------------------