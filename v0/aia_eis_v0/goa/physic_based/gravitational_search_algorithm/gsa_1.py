import copy
import math
import random

from time import perf_counter
import os
import sys
sys.path.append('../../../')
from utils.file_utils.filename_utils import get_ecm_num_str, get_Num_len
from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict

from goa.GOA_criterions import goa_criterion_pack
from GA_pack.fittness_functions.eis_fitness import cal_EIS_WSE_fitness_1

"""
这是第1版代码，考虑Kbest
    Kbest是随着迭代次数逐渐减少的
    Kbest0 = mass_num
    Kbest_final = 1
    Kbest_i = mass_num * (iter_num - iter_index - 1) / iter_num + 1 
    (iter_num - iter_index - 1) / iter_num 确保在iter_index=0时，计算结果是小于1
"""
class GSA_0:
    class Mass:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

            self.v_list = [0.0 for i in range(len(self.limits_list))]

        def update(self):
            self.x_list = [x + v for x, v in zip(self.x_list, self.v_list)]
            for x_i, x in enumerate(self.x_list):
                if (x < self.limits_list[x_i][0]) or (x > self.limits_list[x_i][1]):
                    self.x_list[x_i] = random.uniform(self.limits_list[x_i][0], self.limits_list[x_i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, mass_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.mass_num = mass_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.mass_list = [self.Mass(self.limits_list, self.fitness_function) for i in range(self.mass_num)]

        self.global_best_mass = self.Mass(self.limits_list, self.fitness_function)
        self.global_best_mass.fitness = float('inf')

    def evolve(self):
        cur_best_mass_list = []
        global_best_mass_list = []

        for iter_index in range(self.iter_num):
            # Select the best mass
            sorted_mass_list = sorted(self.mass_list, key= lambda mass : mass.fitness, reverse=False)
            cur_best_mass = sorted_mass_list[0]
            cur_worst_mass = sorted_mass_list[-1]
            if cur_best_mass.fitness < self.global_best_mass.fitness:
                print('{} time: current best fitness {}'.format(iter_index, cur_best_mass.fitness))
                self.global_best_mass = cur_best_mass
            cur_best_mass_list.append(cur_best_mass)
            print('{} time: current global best fitness {}'.format(iter_index, self.global_best_mass.fitness))
            global_best_mass_list.append(self.global_best_mass)

            # relative_fitness对应公式15中的mi
            try:
                relative_fitness_list = [(mass.fitness - cur_worst_mass.fitness)/(cur_best_mass.fitness - cur_worst_mass.fitness) for mass in self.mass_list]
            except ZeroDivisionError as e:
                # 如果运行到这一步，说明算法已经不能再优化，已经收敛，结果不能在改进了
                print(e)
                print('Best x = {} fitness = {}'.format(cur_best_mass.x_list, cur_best_mass.fitness))
                print('Worst x = {} fitness = {}'.format(cur_worst_mass.x_list, cur_worst_mass.fitness))
                continue
            # relative_Fitness对应公式16中的Mi
            relative_Fitness_list = [r_f / sum(relative_fitness_list) for r_f in relative_fitness_list]

            """
            健康值最小的个体，相对健康值为0，如果真的处于很边缘的边界，可能无论迭代多少次都不会更新
            在此处将其健康值 = 相对健康值倒数第二小/2
            """
            # Get the index of Mass with second small relative fitness
            sec_small_r_F = sorted(relative_Fitness_list, reverse=False)[1]
            relative_Fitness_list = [r_F if r_F != 0.0 else sec_small_r_F / 2 for r_F in relative_Fitness_list]

            # 按照Kbest策略计算，每次迭代时要选择前多少个优秀的Mass
            K = math.floor(self.mass_num * (self.iter_num - iter_index - 1) / self.iter_num + 1)
            # Select the first K best Mass
            k_best_mass_list = sorted(self.mass_list, key= lambda mass: mass.fitness, reverse=False)[:K]
            k_best_r_F = [relative_Fitness_list[relative_fitness_list.index((k_b_mass.fitness - cur_worst_mass.fitness)/(cur_best_mass.fitness - cur_worst_mass.fitness))]\
                          for k_b_mass in k_best_mass_list]

            # Calculate the Force and Acceleration
            # 重力常数的初始值，以及迭代公式和相关计算系数参考公式28
            G0 = 100
            alpha = 20
            G = G0 * math.exp(- alpha * iter_index / self.iter_num)
            c = 0.0001
            tmp_mass_list = []
            for m_i, mass_i in enumerate(self.mass_list):
                # Calculate the F in each dimension
                force_list = []
                # for m_j, mass_j in enumerate(self.mass_list):
                #     if m_i != m_j:
                #         # Calculate the distance between mass_i and mass_j
                #         d = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(mass_i.x_list, mass_j.x_list)]))
                #         random_force_list = []
                #         for dim_i in range(len(self.limits_list)):
                #             # Eq 7 in GSA paper
                #             force = G * relative_Fitness_list[m_j] * relative_Fitness_list[m_i] * (mass_j.x_list[dim_i] - mass_i.x_list[dim_i]) / (d + c)
                #             # force乘一个随机数，引入随机扰动
                #             random_force_list.append(force * random.random())
                #         force_list.append(random_force_list)
                for k_mass_j, k_r_F in zip(k_best_mass_list, k_best_r_F):
                    # Calculate the distance between mass_i and mass_j
                    d = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(mass_i.x_list, k_mass_j.x_list)]))
                    random_force_list = []
                    for dim_i in range(len(self.limits_list)):
                        # Eq 7 in GSA paper
                        force = G * k_r_F * relative_Fitness_list[m_i] * (k_mass_j.x_list[dim_i] - mass_i.x_list[dim_i]) / (d + c)
                        # force乘一个随机数，引入随机扰动
                        random_force_list.append(force * random.random())
                    force_list.append(random_force_list)
                # Calculate the a in each dimension
                a_list = []
                for i in range(len(self.limits_list)):
                    f_sum = 0.0
                    for f_l in force_list:
                        f_sum += f_l[i]
                    try:
                        # 每一轮最差的结果，其【相对】健康值必定为0，就会出现【除0错误】，此时用一个很小的值代替
                        a = f_sum / relative_Fitness_list[m_i]
                        a_list.append(a)
                    except ZeroDivisionError as e:
                        a = 0.0
                        a_list.append(a)
                # Update Velocity and Position
                tmp_v_list = [random.random() * v + a for v, a in zip(mass_i.v_list, a_list)]
                tmp_mass = copy.deepcopy(mass_i)
                tmp_mass.v_list = tmp_v_list
                tmp_mass.update()
                tmp_mass_list.append(tmp_mass)
            self.mass_list = tmp_mass_list
        return cur_best_mass_list, global_best_mass_list

# if __name__ == '__main__':
#     iter_num = 500
#     mass_num = 10
#     dim = 5
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     gsa = GSA(iter_num, mass_num, f1_limits_list, f1_fitness_function)
#     cur_best_mass_list, global_best_mass_list = gsa.evolve()
#     print('Best entity position:', gsa.global_best_mass.x_list)
#     print('Fitness:', gsa.global_best_mass.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_mass_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_mass_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, mass_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, mass_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

def calc_G(iter, iter_num):
    # Calculate gravity by Eq 28
    G0 = 100
    alpha = 20
    G = G0 * math.exp(- alpha * iter / iter_num)
    return G

class GSA_1:
    """
    Refer:
        paper:
            GSA: A Gravitational Search Algorithm
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        在用前Kbest个个体来吸引其他粒子时，要把这KBest个个体单独复制一份出来，这样算法才有正常的效果
        如果Kbest个个体在吸引其他粒子时，自己也在变化，算法效果很差
    """
    class Mass:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

            self.v_list = [0.0 for i in range(len(self.limits_list))]
            self.m = None
            self.M = None

        def update(self):
            self.x_list = [x + v for x, v in zip(self.x_list, self.v_list)]
            for x_i, x in enumerate(self.x_list):
                if (x < self.limits_list[x_i][0]) or (x > self.limits_list[x_i][1]):
                    self.x_list[x_i] = random.uniform(self.limits_list[x_i][0], self.limits_list[x_i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, mass_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.mass_num = mass_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.mass_list = [self.Mass(self.limits_list, self.fitness_function) for i in range(self.mass_num)]

        self.global_best_mass = self.Mass(self.limits_list, self.fitness_function)
        self.global_best_mass.fitness = float('inf')

    def search(self):
        cur_best_mass_list = []
        global_best_mass_list = []

        for iter_index in range(self.iter_num):
            # Gravity at current iteration, 重力常数的初始值，以及迭代公式和相关计算系数参考公式28
            G = calc_G(iter=iter_index, iter_num=self.iter_num)
            # 按照Kbest策略计算，每次迭代时要选择前多少个优秀的Mass
            Kbest_num = math.floor(self.mass_num * (self.iter_num - iter_index - 1) / self.iter_num + 1)

            self.mass_list.sort(key=lambda mass:mass.fitness, reverse=False)
            # Select the current best and worst mass
            cur_best_mass = self.mass_list[0]
            cur_worst_mass = self.mass_list[-1]
            if cur_best_mass.fitness < self.global_best_mass.fitness:
                self.global_best_mass = copy.deepcopy(cur_best_mass)
            cur_best_mass_list.append(copy.deepcopy(cur_best_mass))
            global_best_mass_list.append(copy.deepcopy(self.global_best_mass))

            # Calculate m for each agent by Eq 15
            m_list = []
            for i in range(self.mass_num):
                m = (self.mass_list[i].fitness - cur_worst_mass.fitness) / (cur_best_mass.fitness - cur_worst_mass.fitness)
                self.mass_list[i].m = m
                m_list.append(m)

            # Calculate M for each agent by Eq 16
            m_sum = sum(m_list)
            for i in range(self.mass_num):
                self.mass_list[i].M = self.mass_list[i].m / m_sum

            Kbest_mass_list = copy.deepcopy(self.mass_list[: Kbest_num])
            for i in range(self.mass_num):
                mass_i = self.mass_list[i]
                a_pack_list = []
                for j in range(Kbest_num):
                    if i != j:
                        a_list = []
                        # mass_j = self.mass_list[j]
                        mass_j = Kbest_mass_list[j]
                        # Calculate the euclidean distance between mass_i and mass_j
                        R = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(mass_i.x_list, mass_j.x_list)]))
                        for dim_i in range(len(self.limits_list)):
                            x_i = mass_i.x_list[dim_i]
                            x_j = mass_j.x_list[dim_i]
                            a = G * mass_j.M * (x_j - x_i) / (R + 1e-10)
                            a_list.append(a)
                        a_pack_list.append(a_list)

                # Update Velocity and Position of agent
                a_sum_list = [random.random() * sum([t_a[i] for t_a in a_pack_list]) for i in range(len(self.limits_list))]
                mass_i.v_list = [random.random() * v + a for v, a in zip(mass_i.v_list, a_sum_list)]
                mass_i.update()
        return cur_best_mass_list, global_best_mass_list

# if __name__ == '__main__':
#     iter_num = 1000
#     mass_num = 10
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     gsa = GSA_1(iter_num, mass_num, f1_limits_list, f1_fitness_function)
#     cur_best_mass_list, global_best_mass_list = gsa.search()
#     print('Best entity position:', gsa.global_best_mass.x_list)
#     print('Fitness:', gsa.global_best_mass.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_mass_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_mass_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, mass_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, mass_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class GSA_EIS:
    """
    Refer:
        paper:
            GSA: A Gravitational Search Algorithm
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        1-Kbest个个体的使用
            在用前Kbest个个体来吸引其他粒子时，要把这KBest个个体单独复制一份出来，这样算法才有正常的效果
            如果Kbest个个体在吸引其他粒子时，自己也在变化，算法效果很差
        2-F=G * M1 * M2 / R
            In paper: R = ||X1 - X2||, when a dimension with small range, like [0.001~0.1], F is bouncy
            SO, I replace R_i = ||X1_dim_i - X2_dim_i||, the distance of two particle in one dimension
    """
    class Mass:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

            self.v_list = [0.0 for i in range(len(self.limits_list))]
            self.m = None
            self.M = None

        def update(self):
            self.x_list = [x + v for x, v in zip(self.x_list, self.v_list)]
            for x_i, x in enumerate(self.x_list):
                if (x < self.limits_list[x_i][0]) or (x > self.limits_list[x_i][1]):
                    self.x_list[x_i] = random.uniform(self.limits_list[x_i][0], self.limits_list[x_i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, mass_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.mass_num = mass_num
        self.fitness_function = fitness_function

        self.mass_list = [self.Mass(self.exp_data_dict, self.fitness_function) for i in range(self.mass_num)]
        self.global_best_mass = self.Mass(self.exp_data_dict, self.fitness_function)
        self.global_best_mass.fitness = float('inf')

    def search(self):
        cur_best_mass_list = []
        global_best_mass_list = []

        continue_criterion = True
        iter_index = 0
        while continue_criterion:
            # Gravity at current iteration, 重力常数的初始值，以及迭代公式和相关计算系数参考公式28
            G = calc_G(iter=iter_index, iter_num=self.iter_num)
            # 按照Kbest策略计算，每次迭代时要选择前多少个优秀的Mass
            Kbest_num = math.floor(self.mass_num * (self.iter_num - iter_index - 1) / self.iter_num + 1)

            self.mass_list.sort(key=lambda mass:mass.fitness, reverse=False)
            # Select the current best and worst mass
            cur_best_mass = self.mass_list[0]
            cur_worst_mass = self.mass_list[-1]
            if cur_best_mass.fitness < self.global_best_mass.fitness:
                self.global_best_mass = copy.deepcopy(cur_best_mass)
            cur_best_mass_list.append(copy.deepcopy(cur_best_mass))
            global_best_mass_list.append(copy.deepcopy(self.global_best_mass))

            # Calculate m for each agent by Eq 15
            m_list = []
            for i in range(self.mass_num):
                m = (self.mass_list[i].fitness - cur_worst_mass.fitness) / (cur_best_mass.fitness - cur_worst_mass.fitness)
                self.mass_list[i].m = m
                m_list.append(m)

            # Calculate M for each agent by Eq 16
            m_sum = sum(m_list)
            for i in range(self.mass_num):
                self.mass_list[i].M = self.mass_list[i].m / m_sum

            Kbest_mass_list = copy.deepcopy(self.mass_list[: Kbest_num])
            for i in range(self.mass_num):
                mass_i = self.mass_list[i]
                a_pack_list = []
                for j in range(Kbest_num):
                    if i != j:
                        a_list = []
                        # mass_j = self.mass_list[j]
                        mass_j = Kbest_mass_list[j]
                        # Calculate the euclidean distance between mass_i and mass_j
                        # R = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(mass_i.x_list, mass_j.x_list)]))
                        for dim_i in range(len(self.limits_list)):
                            x_i = mass_i.x_list[dim_i]
                            x_j = mass_j.x_list[dim_i]
                            # a = G * mass_j.M * (x_j - x_i) / (R + 1e-10)
                            a = G * mass_j.M * (x_j - x_i) / (abs(x_j - x_i) + 1e-10)
                            a_list.append(a)
                        a_pack_list.append(a_list)

                # Update Velocity and Position of agent
                a_sum_list = [random.random() * sum([t_a[i] for t_a in a_pack_list]) for i in range(len(self.limits_list))]
                mass_i.v_list = [random.random() * v + a for v, a in zip(mass_i.v_list, a_sum_list)]
                mass_i.update()

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter_index >= 1:
                x_lists_list = [global_best_mass_list[-2].x_list, global_best_mass_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter_index,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter_index += 1
        return cur_best_mass_list, global_best_mass_list, iter_index, chi_squared

class GSA_EIS_access:
    """
    Refer:
        paper:
            GSA: A Gravitational Search Algorithm
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        1-Kbest个个体的使用
            在用前Kbest个个体来吸引其他粒子时，要把这KBest个个体单独复制一份出来，这样算法才有正常的效果
            如果Kbest个个体在吸引其他粒子时，自己也在变化，算法效果很差
        2-F=G * M1 * M2 / R
            In paper: R = ||X1 - X2||, when a dimension with small range, like [0.001~0.1], F is bouncy
            SO, I replace R_i = ||X1_dim_i - X2_dim_i||, the distance of two particle in one dimension
    """
    class Mass:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

            self.v_list = [0.0 for i in range(len(self.limits_list))]
            self.m = None
            self.M = None

        def update(self):
            self.x_list = [x + v for x, v in zip(self.x_list, self.v_list)]
            for x_i, x in enumerate(self.x_list):
                if (x < self.limits_list[x_i][0]) or (x > self.limits_list[x_i][1]):
                    self.x_list[x_i] = random.uniform(self.limits_list[x_i][0], self.limits_list[x_i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, mass_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.mass_num = mass_num
        self.fitness_function = fitness_function

        self.mass_list = [self.Mass(self.exp_data_dict, self.fitness_function) for i in range(self.mass_num)]
        self.global_best_mass = self.Mass(self.exp_data_dict, self.fitness_function)
        self.global_best_mass.fitness = float('inf')

    def search(self, res_fn, start_time):
        cur_best_mass_list = []
        global_best_mass_list = []

        continue_criterion = True
        iter_index = 0
        while continue_criterion:
            # Gravity at current iteration, 重力常数的初始值，以及迭代公式和相关计算系数参考公式28
            G = calc_G(iter=iter_index, iter_num=self.iter_num)
            # 按照Kbest策略计算，每次迭代时要选择前多少个优秀的Mass
            Kbest_num = math.floor(self.mass_num * (self.iter_num - iter_index - 1) / self.iter_num + 1)

            self.mass_list.sort(key=lambda mass:mass.fitness, reverse=False)
            # Select the current best and worst mass
            cur_best_mass = self.mass_list[0]
            cur_worst_mass = self.mass_list[-1]
            if cur_best_mass.fitness < self.global_best_mass.fitness:
                self.global_best_mass = copy.deepcopy(cur_best_mass)
            cur_best_mass_list.append(copy.deepcopy(cur_best_mass))
            global_best_mass_list.append(copy.deepcopy(self.global_best_mass))

            # Calculate m for each agent by Eq 15
            m_list = []
            for i in range(self.mass_num):
                m = (self.mass_list[i].fitness - cur_worst_mass.fitness) / (cur_best_mass.fitness - cur_worst_mass.fitness)
                self.mass_list[i].m = m
                m_list.append(m)

            # Calculate M for each agent by Eq 16
            m_sum = sum(m_list)
            for i in range(self.mass_num):
                self.mass_list[i].M = self.mass_list[i].m / m_sum

            Kbest_mass_list = copy.deepcopy(self.mass_list[: Kbest_num])
            for i in range(self.mass_num):
                mass_i = self.mass_list[i]
                a_pack_list = []
                for j in range(Kbest_num):
                    if i != j:
                        a_list = []
                        # mass_j = self.mass_list[j]
                        mass_j = Kbest_mass_list[j]
                        # Calculate the euclidean distance between mass_i and mass_j
                        # R = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(mass_i.x_list, mass_j.x_list)]))
                        for dim_i in range(len(self.limits_list)):
                            x_i = mass_i.x_list[dim_i]
                            x_j = mass_j.x_list[dim_i]
                            # a = G * mass_j.M * (x_j - x_i) / (R + 1e-10)
                            a = G * mass_j.M * (x_j - x_i) / (abs(x_j - x_i) + 1e-10)
                            a_list.append(a)
                        a_pack_list.append(a_list)

                # Update Velocity and Position of agent
                a_sum_list = [random.random() * sum([t_a[i] for t_a in a_pack_list]) for i in range(len(self.limits_list))]
                mass_i.v_list = [random.random() * v + a for v, a in zip(mass_i.v_list, a_sum_list)]
                mass_i.update()

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter_index >= 1:
                x_lists_list = [global_best_mass_list[-2].x_list, global_best_mass_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter_index, \
                                                                max_iter_time=self.iter_num, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter_index) + ',[' \
                           + ','.join([str(para) for para in global_best_mass_list[-1].x_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter_index += 1

def access_GSA_EIS():
    counter = 0
    # Iterate on 9 ECMs
    for i in range(1, 10):
        ecm_sim_folder = '../../../datasets/goa_datasets/simulated'
        ecm_num = i
        ecm_num_str = get_ecm_num_str(ecm_num)
        file_path = os.path.join(ecm_sim_folder, 'ecm_' + ecm_num_str)
        sim_ecm = load_sim_ecm_para_config_dict(ecm_num, file_path)
        para_num = len(sim_ecm['para'])

        # Iterate for 100 times
        for j in range(100):
            t_start = perf_counter()
            # ------------------------------  Change GOA name ------------------------------
            goa = GSA_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, mass_num=10*para_num)
            res_fn = 'gsa_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('GSA left: {0}'.format(900 - counter))
access_GSA_EIS()