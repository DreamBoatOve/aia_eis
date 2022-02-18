import math
import random
import copy

"""
通过递归(Recursion)创建在所搜空间中的网格
可以直接使用的工具有：
    itertools.product，一日一技：如何用Python遍历多个列表元素的所有组合：https://cloud.tencent.com/developer/article/1450862
    [Python]多个列表的排列组合,史上最简单的方法：https://blog.csdn.net/wuuud1/article/details/83991951
    Python进阶：递归算法：https://www.jianshu.com/p/45ef6cc19f54
"""
# ------------------ Wrong code, abndoned ------------------
# def create_search_grid_0(x_list_pack):
#     x1_list = x_list_pack.pop()
#     # x1_list = [[x] for x in x1_list]
#
#     def f_recursion(x_list, x_pack):
#         x2_list = x_pack.pop()
#         print('x2=', x2_list)
#         if len(x_pack) == 0:
#             x2_list = [[x2] for x2 in x2_list]
#             return [x2.append(x1) for x1 in x_list for x2 in x2_list]
#         else:
#             return f_recursion(x_list, f_recursion(x2_list, x_pack))
#     return f_recursion(x1_list, x_list_pack)
# ------------------ Wrong code, abndoned ------------------
def create_search_grid_1(x_list_pack):
    x1_list = x_list_pack.pop()

    def f_recursion(x_list, x_pack, flag=1):
        if flag == 0:
            tmp_x_list = []
            for x1 in x_list:
                for x2 in x_pack:
                    x2_c = copy.deepcopy(x2)
                    x2_c.append(x1)
                    tmp_x_list.append(x2_c)
            return tmp_x_list, flag
        elif len(x_pack) == 1:
            x2_list = x_pack.pop()
            # print('x2=', x2_list)
            flag = 0
            x2_list = [[x2] for x2 in x2_list]
            tmp_x_list = []
            for x1 in x_list:
                for x2 in x2_list:
                    x2_c = copy.deepcopy(x2)
                    x2_c.append(x1)
                    tmp_x_list.append(x2_c)
            return tmp_x_list, flag
        else:
            x2_list = x_pack.pop()
            # print('x2=', x2_list)
            return f_recursion(x_list, *f_recursion(x2_list, x_pack))
    res =  f_recursion(x1_list, x_list_pack, flag=1)
    return res[0]

# Implementation by reduce 写不动，不写了
def create_search_grid_2(x_list_pack):
    pass

# Implementation by itertools.product 实现最简单
def create_search_grid_3(x_list_pack):
    from itertools import product
    return product(*x_list_pack)

# -------------------- Recursion Test--------------------
# if __name__ == '__main__':
#     list1 = [1, 2]
#     list2 = [3, 4]
#     list3 = [5, 6]
#     list4 = [7,8,9,10,11]
#     x_list_pack = [list1, list2, list3, list4]

    # ----- 1: Mine original implementation -----
    # r1 = create_search_grid_1(x_list_pack)
    # print(r1, len(r1[0]))
    # ----- 1: Mine original implementation -----

    # ----- 2: Implementation by reduce -----

    # ----- 2: Implementation by reduce -----

    # ----- 3: Implementation by itertools.product -----
    # For example, product(A, B) returns the same as:  ((x,y) for x in A for y in B)
    # r3 = create_search_grid_3(x_list_pack)
    # print(r3)
    # count = 0
    # for r in r3:
    #     print(r, type(r))
    #     count += 1
    # print(count)
    # ----- 3: Implementation by itertools.product -----
# -------------------- Recursion Test--------------------
# Calculate the distance between the probe_i and probe_j
def cal_dis(x1_list, x2_list):
    return math.sqrt(sum([pow((x1 - x2), 2) for x1, x2 in zip(x1_list, x2_list)]))

class CFO:
    """
    Refered paper
        CENTRAL FORCE OPTIMIZATION: A NEW METAHEURISTIC WITH APPLICATIONS IN APPLIED ELECTROMAGNETICS
        这个算法的初始设计是求函数的最大值（适应度），我们要求最小值，要注意转换
    """
    class Probe:
        def __init__(self, limits_list, fitness_function, fit_amplifier):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            # CFO算法对于个体初始位置的分布安排是网格分布，个体在搜索空间中均匀分布
            # 如果内有被分派初始的位置，那就随机初始化的位置
            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.a_list = [0 for i in range(len(self.limits_list))]

            self.fit_amplifier = fit_amplifier
            self.fitness = fit_amplifier / (1.0 + fitness_function(self.x_list))

        def update(self):
            # Use acceleration to calculate new position of probe
            # print('original x list {} and fitness {}'.format(self.x_list, self.fitness))
            self.x_list = [x + 0.5 * a for x, a in zip(self.x_list, self.a_list)]
            # print('updated x list:', self.x_list)
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fit_amplifier / (1.0 + self.fitness_function(self.x_list))
            # print('updated fitness:', self.fitness)

    def __init__(self, iter_num, probe_num_1_dim, limits_list, fitness_function, fit_amplifier, beta=1):
        self.iter_num = iter_num
        self.probe_num_1_dim = probe_num_1_dim
        self.probe_num = probe_num_1_dim ** len(limits_list)

        self.limits_list = limits_list

        self.fit_amplifier = fit_amplifier
        self.fitness_function = fitness_function

        self.beta = beta

        self.probe_list = []
        # 网格化的初始化所有探针的位置
        # 生成每个维度上的均匀分布的点位
        x_all_dim_list_pack = [[limit[0] + i * (limit[1] - limit[0]) / (self.probe_num_1_dim - 1) for i in range(self.probe_num_1_dim)] for limit in self.limits_list]

        # 将位置列表赋值给Probe的位置列表，更新计算适应度
        for p_x_tuple in create_search_grid_3(x_all_dim_list_pack):
            probe = self.Probe(limits_list, fitness_function, self.fit_amplifier)
            probe.x_list = list(p_x_tuple)
            probe.update()
            self.probe_list.append(probe)

        self.global_best_probe = self.Probe(limits_list, fitness_function, fit_amplifier)
        # 在CFO里是fitness越大越好，所以全局最佳的probe的fitness初始值要最小
        self.global_best_probe.fitness = 0.0

    def evolve(self):
        current_best_probe_list = []
        global_best_probe_list = []

        for iter_index in range(self.iter_num):
            current_best_probe = sorted(self.probe_list, key= lambda probe : probe.fitness, reverse=True)[0]
            # print('Iter {}, best probe {} and fitness {}'.format(iter_index, current_best_probe.x_list, current_best_probe.fitness))
            if current_best_probe.fitness > self.global_best_probe.fitness:
                # print('Iter {}, best probe {} and fitness {}'.format(iter_index, current_best_probe.x_list, current_best_probe.fitness))
                self.global_best_probe = current_best_probe
            current_best_probe_list.append(current_best_probe)
            global_best_probe_list.append(self.global_best_probe)

            all_probe_a_sum_list = []
            G = 15
            alpha = 0
            # beta代表距离绝对值的次幂，分成2+1看待，2代表距离绝对值的二次方，1代表 (Rs - Rp)方向向量 / ||Rs - Rp||距离的绝对值 = 单位方向向量
            # Update acceleration and position of each probe
            for p_i in range(self.probe_num):
                probe_i = self.probe_list[p_i]
                probe_i_a_list = []
                for p_j in range(self.probe_num):
                    probe_i_tmp_a_list = []
                    # Judge whether the difference (fitness) is positive or not == function U
                    if (p_i != p_j):
                        if self.probe_list[p_j].fitness > self.probe_list[p_i].fitness:
                            probe_j = self.probe_list[p_j]
                            # Calculate the distance between the probe_i and probe_j
                            d = cal_dis(probe_i.x_list, probe_j.x_list)
                            # Calculate the acceleration in each dimension
                            for x_i in range(len(self.limits_list)):
                                a = G * (probe_j.fitness - probe_i.fitness) * (probe_j.x_list[x_i] - probe_i.x_list[x_i]) / pow(d, self.beta)
                                probe_i_tmp_a_list.append(a)
                        else:
                            probe_i_tmp_a_list.extend([0 for i in range(len(self.limits_list))])
                    else:
                        probe_i_tmp_a_list.extend([0 for i in range(len(self.limits_list))])
                    probe_i_a_list.append(probe_i_tmp_a_list)

                probe_i_a_sum_list = []
                for i in range(len(self.limits_list)):
                    a_sum = 0.0
                    for probe_i_a in probe_i_a_list:
                        # print(probe_i_a[i])
                        a_sum += probe_i_a[i]
                    probe_i_a_sum_list.append(a_sum)
                all_probe_a_sum_list.append(probe_i_a_sum_list)
            # Update each probe according to their acceleration
            for index, probe_a_sum_list in enumerate(all_probe_a_sum_list):
                self.probe_list[index].a_list = probe_a_sum_list
                self.probe_list[index].update()
        return current_best_probe_list, global_best_probe_list

if __name__ == '__main__':
    iter_num = 1000
    # 此处是每一维度上都均匀的初始化10个样本，整个样本数量 = 每个维度上的点数 ** 空间维度
    # 每个维度上设置的探针数量
    Np_Nd = 3
    dim = 10
    probe_num = Np_Nd ** dim

    f1_limits_list = [[-120, 80] for i in range(dim)]
    from GA_pack.fittness_functions.f1 import f1
    f1_fitness_function = f1

    fit_amplifier = 1000

    cfo = CFO(iter_num, Np_Nd, f1_limits_list, f1_fitness_function, fit_amplifier, beta=2)
    current_best_probe_list, global_best_probe_list = cfo.evolve()
    print('Best entity position:', cfo.global_best_probe.x_list)
    print('The biggest target is {}'.format(fit_amplifier))
    print('Fitness:', cfo.global_best_probe.fitness)

    # Draw the best entity in each iteration.
    iter_list = [i for i in range(iter_num)]
    cur_fitness_list = [probe.fitness for probe in current_best_probe_list]
    cur_global_fitness_list = [probe.fitness for probe in global_best_probe_list]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nprobe number {1}\nDimension {2}'.format(iter_num, probe_num, dim))
    line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
    line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nprobe number {1}\nDimension {2}'.format(iter_num, probe_num, dim))
    line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    ax.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Error rate')
    plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
    plt.show()

class CFO_1:
    """
    Refered paper
        CENTRAL FORCE OPTIMIZATION: A NEW METAHEURISTIC WITH APPLICATIONS IN APPLIED ELECTROMAGNETICS
        这个算法的初始设计是求函数的最大值（适应度），我们要求最小值，要注意转换
    Version:
        Second
    """
    class Probe:
        def __init__(self, limits_list, fit_amplifier, fitness_function):
            self.limits_list = limits_list
            self.fit_amplifier = fit_amplifier
            self.fitness_function = fitness_function

            # CFO算法对于个体初始位置的分布安排是网格分布，个体在搜索空间中均匀分布
            # 如果没有被分派初始的位置，那就随机初始化的位置
            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.a_list = [0 for i in range(len(self.limits_list))]

            self.fitness = fitness_function(self.x_list)
            self.inversed_fitness = self.fit_amplifier / self.fitness

        def update(self, tmp_x_list=None):
            """
            Function
                The update of probe's position has two forms:
                    1-Given a new position
                    2-Given acceleration
                    Both requires boundary check after update
            :param
                tmp_x_list:
                    None or list(float)
            :return:
            """
            # 1- Update by given a new position
            if type(tmp_x_list) == list:
                for i, t_x in enumerate(tmp_x_list):
                    x_min = self.limits_list[i][0]
                    x_max = self.limits_list[i][1]
                    if t_x < self.limits_list[i][0]:
                        t_x = x_min + 0.5 * (self.x_list[i] - x_min)
                    elif t_x > self.limits_list[i][1]:
                        t_x = x_max - 0.5 * (x_max - self.x_list[i])
                    self.x_list[i] = t_x

            # 2- Update by given acceleration, Use acceleration to calculate new position of probe
            elif tmp_x_list == None:
                tmp_x_list = [x + 0.5 * a for x, a in zip(self.x_list, self.a_list)]
                for i, t_x in enumerate(tmp_x_list):
                    x_min = self.limits_list[i][0]
                    x_max = self.limits_list[i][1]
                    if t_x < self.limits_list[i][0]:
                        t_x = x_min + 0.5 * (self.x_list[i] - x_min)
                    elif t_x > self.limits_list[i][1]:
                        t_x = x_max - 0.5 * (x_max - self.x_list[i])
                    self.x_list[i] = t_x

            self.fitness = self.fitness_function(self.x_list)
            self.inversed_fitness = self.fit_amplifier / self.fitness
            # print('updated fitness:', self.fitness)

    def __init__(self, iter_num, Np_Nd, limits_list, fit_amplifier, fitness_function):
        """
        :param
            iter_num:
            Np_Nd: 每个维度上设置的探针数量, see 5.1. CFO Pseudocode (2) Initialize Run (A)
            limits_list:
            fitness_function:
            fit_amplifier:
                Very important parameter, should spend more time to adjust it
                inversed_fitness = fit_amplifier / fitness
                fitness might be very huge number, thus it leads to very small inversed_fitness(Mass)
                the difference between two small mass will leads to small attraction to each other
            beta:
        """
        self.iter_num = iter_num
        self.Np_Nd = Np_Nd

        # Np_Nd^dim
        # self.probe_num = Np_Nd ** len(limits_list)
        # Np_Nd x dim
        self.probe_num = Np_Nd * len(limits_list)

        self.limits_list = limits_list
        self.fit_amplifier = fit_amplifier
        self.fitness_function = fitness_function

        # --------------- The initialization of probe according to 5.1 CFO Pseudocode ---------------
        self.probe_list = []
        # # 网格化的初始化所有探针的位置
        # # 生成每个维度上的均匀分布的点位
        x_all_dim_list_pack = [[limit[0] + i * (limit[1] - limit[0]) / (self.Np_Nd - 1)\
                                for i in range(self.Np_Nd)] \
                                for limit in self.limits_list]

        # 将位置列表赋值给Probe的位置列表，更新计算适应度
        for p_x_tuple in create_search_grid_3(x_all_dim_list_pack):
            probe = self.Probe(limits_list, self.fit_amplifier, fitness_function)
            probe.update(tmp_x_list = list(p_x_tuple))
            self.probe_list.append(probe)
        # --------------- The initialization of probe according to 5.1 CFO Pseudocode ---------------

        # --------------- The mine random initialization of probe ---------------
        # self.probe_list = [self.Probe(limits_list, self.fit_amplifier, fitness_function) for i in range(self.probe_num)]
        # --------------- The mine random initialization of probe ---------------

        self.global_best_probe = self.Probe(limits_list, self.fit_amplifier, fitness_function)
        # 在CFO里是Mass越大越好，所以全局最佳的probe的fitness初始值要最小
        self.global_best_probe.fitness = float('inf')

    def evolve(self, G=2, alpha=3, beta=1):
        """
        :param
            G
                CFO's gravitational constant
                G > 0
            α and β
                there are no physical parameters corresponding to the exponents α > 0 and β > 0,
            Many cases in paper use the following parameters: G = 2, alpha = 2, beta = 2
        :return:
        """
        print('G={0}, alpha={1}, beta={2}'.format(G, alpha, beta))
        current_best_probe_list = []
        global_best_probe_list = []

        for iter_index in range(self.iter_num):
            G = 2 + 198 * (iter_index) / self.iter_num
            current_best_probe = sorted(self.probe_list, key= lambda probe : probe.fitness, reverse=False)[0]
            # print('Iter {}, best probe {} and fitness {}'.format(iter_index, current_best_probe.x_list, current_best_probe.fitness))
            if current_best_probe.fitness < self.global_best_probe.fitness:
                self.global_best_probe = copy.deepcopy(current_best_probe)
                print('Iter {}, best probe {} and fitness {}'.format(iter_index, self.global_best_probe.x_list, self.global_best_probe.fitness))
            current_best_probe_list.append(copy.deepcopy(current_best_probe))
            global_best_probe_list.append(copy.deepcopy(self.global_best_probe))

            # beta代表距离绝对值的次幂，分成2+1看待，2代表距离绝对值的二次方，1代表 (Rs - Rp)方向向量 / ||Rs - Rp||距离的绝对值 = 单位方向向量
            # Update acceleration and position of each probe
            all_probe_a_sum_list = []
            for p_i in range(self.probe_num):
                probe_i = self.probe_list[p_i]
                probe_i_a_list = []
                for p_j in range(self.probe_num):
                    probe_i_tmp_a_list = []
                    if (p_i != p_j):
                        # Judge whether the difference (fitness) is positive or not == function U
                        # Mass_J > Mass_i, U = 1
                        if self.probe_list[p_j].inversed_fitness > self.probe_list[p_i].inversed_fitness:
                            probe_j = self.probe_list[p_j]
                            # Calculate the distance between the probe_i and probe_j
                            d = cal_dis(probe_i.x_list, probe_j.x_list)
                            # Calculate the acceleration in each dimension
                            for x_i in range(len(self.limits_list)):
                                a = G * pow(probe_j.inversed_fitness - probe_i.inversed_fitness, alpha) * (probe_j.x_list[x_i] - probe_i.x_list[x_i]) / pow(d, beta)
                                probe_i_tmp_a_list.append(a)
                        # Mass_J < Mass_i, U = 0
                        else:
                            probe_i_tmp_a_list.extend([0 for i in range(len(self.limits_list))])
                    else:
                        probe_i_tmp_a_list.extend([0 for i in range(len(self.limits_list))])
                    probe_i_a_list.append(probe_i_tmp_a_list)

                probe_i_a_sum_list = []
                for i in range(len(self.limits_list)):
                    a_sum = 0.0
                    for probe_i_a in probe_i_a_list:
                        # print(probe_i_a[i])
                        a_sum += probe_i_a[i]
                    probe_i_a_sum_list.append(a_sum)
                all_probe_a_sum_list.append(probe_i_a_sum_list)

            # Update each probe according to their acceleration
            for index, probe_a_sum_list in enumerate(all_probe_a_sum_list):
                # print('A',probe_i_a_sum_list)
                self.probe_list[index].a_list = probe_a_sum_list
                self.probe_list[index].update()
        return current_best_probe_list, global_best_probe_list

# if __name__ == '__main__':
#     import sys
#     sys.path.append('../../../')
#
#     iter_num = 5000
#     # 此处是每一维度上都均匀的初始化10个样本，整个样本数量 = 每个维度上的点数 ** 空间维度
#     # 每个维度上设置的探针数量
#     Np_Nd = 5
#     dim = 3
#     fit_amplifier = 100000
#
#     # Np_Nd^dim
#     # probe_num = Np_Nd ** dim
#     # Np_Nd x dim
#     probe_num = Np_Nd * dim
#
#     f1_limits_list = [[-120, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     cfo = CFO_1(iter_num, Np_Nd, f1_limits_list, fit_amplifier, f1_fitness_function)
#     current_best_probe_list, global_best_probe_list = cfo.evolve()
#     print('Best entity position:', cfo.global_best_probe.x_list)
#     print('Fitness:', cfo.global_best_probe.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [probe.fitness for probe in current_best_probe_list]
#     cur_global_fitness_list = [probe.fitness for probe in global_best_probe_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nprobe number {1}\nDimension {2}'.format(iter_num, probe_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nprobe number {1}\nDimension {2}'.format(iter_num, probe_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()