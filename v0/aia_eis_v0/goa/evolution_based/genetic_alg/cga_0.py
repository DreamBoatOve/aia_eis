import sys
sys.path.append('../../../')

import copy
import math
import random

from time import perf_counter

import os
from utils.file_utils.filename_utils import get_ecm_num_str, get_Num_len
from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict

from goa.GOA_criterions import goa_criterion_pack
from GA_pack.fittness_functions.eis_fitness import cal_EIS_WSE_fitness_1

class CGA_0:
    """
    Continuous Genetic Algorithm
    Refer:
        Book:
            Book0: 遗传算法原理及应用
        Paper:
            paper0: A Continuous Genetic Algorithm Designed for the Global Optimization of Multimodal Functions
                Its citation is more than 300. So, it is trusted
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        1
            Previous version.0 is at 'from GA_pack.GAs.basic.sga'
            This version use float variable, and is more convenient and compact with other algorithms
            每轮迭代中，参数的上下界为该轮所有个体的坐标的极值，使得参数的上下界迅速减少，迭代十几次就出现上下界相同的情况
    """
    class Entity:
        def __init__(self, tmp_limits_list, fitness_function):
            self.tmp_limits_list = tmp_limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.tmp_limits_list]
            self.fitness = self.fitness_function(self.x_list)

        def update(self):
            for i, limit in enumerate(self.tmp_limits_list):
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        # MaxIter
        self.iter_num = iter_num
        """
        Setting of part parameters is described in paper0-section 3
            Initialization. The parameters fixed at the beginning are the following ones:
                – search domain of each function variable;
                – starting point (randomly generated);
                – initial population (see on Section 2.2.1)
            The parameters which are automatically built by using the parameters fixed at the beginning are the following ones:
                – the neighborhood parameter r_neighb;
                – the length δ of the smallest edge of the initial hyperrectangular search domain;
                – the radius ε of the individual neighborhood;
                – the maximal number of successive generations without any improvement of the best
                known objective function value MaxGen;
                – the maximal number of iterations MaxIter;
                – the accepted distance between the best point and the generated individuals P_abs.
            Fixed parameters are:
                the initial population size IPopSize (equal to 30)
                the final population size FPopSize (equal to 10)
                the initial mutation IPMut and crossover probabilities PXcross (respectively equal to 0.9 and 0.85)
                the reduction parameters P_red and d_PopSize (respectively equal to 2 and 5)
        """
        # Equals to the size of initial population, IPopSize
        self.entity_num = entity_num
        # Size of final population
        self.FPopSize = int(0.33 * self.entity_num)

        # self.p_crossover = 0.85
        # initial value of Mutation probability
        self.I_P_mutate = 0.9
        self.p_mutate = self.I_P_mutate
        self.I_K = 1.0
        self.k = self.I_K

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        """
        Information about radius (ε) in paper0:
            1- A ball B(s, ε), centered on s with the radius ε, contains all points s' such as: || s' - s || <= ε
            2- In the intensification module, we reduce the search domain, the population size, the neighborhood radius,
            the probability of mutation, we generate the new population around the best point found, and we restart the 
            genetic procedure inside this new search domain.
            3- the highest distance between the best point up to now BestPoint and the generated individuals Ind_i 
            is smaller than a given neighborhood radius r_abs
            4- These parameters are fixed empirically, but the radius ε of individual neighborhood is determined by 
            taking into account the objective function characteristics.
            5- The search domain and the radius ε of the individual neighborhood are divided per r_red
            6- The radius ε of individual neighborhood is equal to δ / r_neighb
                the length δ of the smallest edge of the initial hyperrectangular search domain;
                the neighborhood parameter r_neighb = the product of the initial population size *
                                                      the dimension of the problem
        """
        self.r_neighb = self.entity_num * len(self.limits_list)
        self.radius_list = [(limit[1] - limit[0]) / self.r_neighb for limit in self.limits_list]

        # ------------------- Generation of the initial population -------------------
        # A generated individual is accepted as an initial individual, if it does not belong to the
        # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
        all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
        while len(all_entity_x_list) < self.entity_num:
            x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Evaluate whether the point is in any entities' neighborhood
            all_flag_list = []
            for e_x_list in all_entity_x_list:
                e_flag_list = []
                for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                    if abs(x - e_x) >= radius:
                        e_flag_list.append(True)
                if False in e_flag_list:
                    continue
                else:
                    all_flag_list.append(True)
            all_entity_x_list.append(x_list)

        self.entity_list = []
        for x_list in all_entity_x_list:
            entity = self.Entity(self.limits_list, self.fitness_function)
            entity.x_list = x_list
            entity.update()
            self.entity_list.append(copy.deepcopy(entity))
        # ------------------- Generation of the initial population -------------------
        self.global_best_entity = self.Entity(self.limits_list, fitness_function)

    def evolve(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        for iter in range(self.iter_num):
            self.entity_list.sort(key=lambda entity:entity.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            print('Iter:', iter,\
                  '\nCurrent limit:',self.limits_list,\
                  '\nCurrent best entity, x:{0}, fitness:{1}'.format(cur_best_entity.x_list, cur_best_entity.fitness))
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # -------------------------------------- Generate new population --------------------------------------
            new_entity_list = []
            # 1-Generate how many offsprings?
            # I can not understand paper0, so, I take my way: generate half of entity_num offsprings
            for i in range(int(math.ceil(self.entity_num * 0.5))):
                # -------------- Selection, select two parents by roulette-wheel selection --------------
                fitness_list = [entity.fitness for entity in self.entity_list]
                inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
                parent_indexs = []
                while len(parent_indexs) < 2:
                    random_pointer = random.uniform(0, sum(inv_fitness_list))
                    inv_fitness_sum = 0.0
                    for index, inv_fitness in enumerate(inv_fitness_list):
                        inv_fitness_sum += inv_fitness
                        if random_pointer < inv_fitness_sum:
                            parent_indexs.append(index)
                            break
                parent_i = self.entity_list[parent_indexs[0]]
                parent_j = self.entity_list[parent_indexs[1]]
                # -------------- Selection, select two parents by roulette-wheel selection --------------

                # -------------- Recombination --------------
                random_cross_index = random.randint(0, len(self.limits_list) - 1)
                tmp_i_x_list = copy.deepcopy(parent_i.x_list)
                tmp_j_x_list = copy.deepcopy(parent_j.x_list)
                random_M = random.randint(1, 1000)
                d_x = tmp_i_x_list[random_cross_index] / random_M
                d_y = tmp_j_x_list[random_cross_index] / random_M
                tmp_i_x_list[random_cross_index] = tmp_i_x_list[random_cross_index] + d_y - d_x
                tmp_j_x_list[random_cross_index] = tmp_j_x_list[random_cross_index] - d_y + d_x

                # All the components situated on the left of this crossing point are not affected,
                # those situated after the crossing point are exchanged
                tmp_i_x_list = tmp_i_x_list[: random_cross_index + 1] + tmp_j_x_list[random_cross_index + 1 :]
                tmp_j_x_list = tmp_j_x_list[: random_cross_index + 1] + copy.deepcopy(parent_i.x_list[random_cross_index + 1 :])
                # -------------- Recombination --------------

                # -------------- Mutation --------------
                if random.random() < self.p_mutate:
                    for t_x_list in [tmp_i_x_list, tmp_j_x_list]:
                        random_mutate_index = random.randint(0, len(self.limits_list) - 1)
                        limit = self.limits_list[random_mutate_index]
                        d_mutate_x = (limit[1] - limit[0]) / random.randint(1, 10)
                        t_x_list[random_mutate_index] = t_x_list[random_mutate_index]\
                                                        + pow(-1, random.randint(1, 2)) * d_mutate_x
                # -------------- Mutation --------------
                tmp_entity_i = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity_i.x_list = tmp_i_x_list
                tmp_entity_i.update()
                tmp_entity_j = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity_j.x_list = tmp_j_x_list
                tmp_entity_j.update()
                new_entity_list.extend([tmp_entity_i, tmp_entity_j])
                # When to stop diversification and when to starts intensification?

            # 2-How to merge offsprings and parents?
            # I can not understand paper0, so, I take my way: merge old population with offsprings,
            # and select the first entity_num ones
            self.entity_list = sorted(self.entity_list + new_entity_list, key=lambda en : en.fitness, reverse=False)[: self.entity_num]
            # -------------------------------------- Generate new population --------------------------------------

            # -------------------------------------- Update global settings --------------------------------------
            # Narrow limitation
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                x_col_list = []
                for en in self.entity_list:
                    x_col_list.append(en.x_list[i])
                tmp_limit_list.append([min(x_col_list), max(x_col_list)])
            self.limits_list = copy.deepcopy(tmp_limit_list)
            # k
            self.k = self.I_K * (self.iter_num - iter) / self.iter_num
            # radius
            self.radius_list = [r * (self.iter_num - iter) / self.iter_num for r in self.radius_list]
            # mutation ratio
            self.p_mutate = self.I_P_mutate * pow((self.iter_num - iter) / self.iter_num, 3)
            # -------------------------------------- Update global settings --------------------------------------
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 20
#     entity_num = 15
#     dim = 5
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     cga = CGA_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = cga.evolve()
#     print('Best entity position:', cga.global_best_entity.x_list)
#     print('Fitness:', cga.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#     for i, c_en, g_en in zip(range(iter_num), current_best_entity_list, global_best_entity_list):
#         print(i, c_en.fitness, g_en.fitness)
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class CGA_1:
    """
    Continuous Genetic Algorithm
    Refer:
        Book:
            Book0: 遗传算法原理及应用
        Paper:
            paper0: A Continuous Genetic Algorithm Designed for the Global Optimization of Multimodal Functions
                Its citation is more than 300. So, it is trusted
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        1
            Previous version.0 is at 'from GA_pack.GAs.basic.sga'
            This version use float variable, and is more convenient and compact with other algorithms
            每轮迭代中，参数的上下界为该轮所有个体的坐标的极值，使得参数的上下界迅速减少，迭代十几次就出现上下界相同的情况
        2
            每轮迭代中，参数的上下界逐步线性搜小，同时考虑该轮所有个体坐标的上下界，取二者中的较大值
            随着迭代，其他个体逐渐向最优个体靠拢并逐渐变成一样（x_list, fitness），导致轮盘赌选择父母的时候陷入死循环
    """
    class Entity:
        def __init__(self, tmp_limits_list, fitness_function):
            self.tmp_limits_list = tmp_limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.tmp_limits_list]
            self.fitness = self.fitness_function(self.x_list)

        def update(self):
            for i, limit in enumerate(self.tmp_limits_list):
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        # MaxIter
        self.iter_num = iter_num
        """
        Setting of part parameters is described in paper0-section 3
            Initialization. The parameters fixed at the beginning are the following ones:
                – search domain of each function variable;
                – starting point (randomly generated);
                – initial population (see on Section 2.2.1)
            The parameters which are automatically built by using the parameters fixed at the beginning are the following ones:
                – the neighborhood parameter r_neighb;
                – the length δ of the smallest edge of the initial hyperrectangular search domain;
                – the radius ε of the individual neighborhood;
                – the maximal number of successive generations without any improvement of the best
                known objective function value MaxGen;
                – the maximal number of iterations MaxIter;
                – the accepted distance between the best point and the generated individuals P_abs.
            Fixed parameters are:
                the initial population size IPopSize (equal to 30)
                the final population size FPopSize (equal to 10)
                the initial mutation IPMut and crossover probabilities PXcross (respectively equal to 0.9 and 0.85)
                the reduction parameters P_red and d_PopSize (respectively equal to 2 and 5)
        """
        # Equals to the size of initial population, IPopSize
        self.entity_num = entity_num
        # Size of final population
        self.FPopSize = int(0.33 * self.entity_num)

        self.p_crossover = 0.85
        # initial value of Mutation probability
        self.I_P_mutate = 0.9
        self.p_mutate = self.I_P_mutate
        self.I_K = 1.0
        self.k = self.I_K

        self.limits_list = limits_list
        self.limit_decrease_step_list = [(limit[1] - limit[0]) * 0.9 / (2 * self.iter_num) for limit in self.limits_list]
        self.fitness_function = fitness_function

        """
        Information about radius (ε) in paper0:
            1- A ball B(s, ε), centered on s with the radius ε, contains all points s' such as: || s' - s || <= ε
            2- In the intensification module, we reduce the search domain, the population size, the neighborhood radius,
            the probability of mutation, we generate the new population around the best point found, and we restart the 
            genetic procedure inside this new search domain.
            3- the highest distance between the best point up to now BestPoint and the generated individuals Ind_i 
            is smaller than a given neighborhood radius r_abs
            4- These parameters are fixed empirically, but the radius ε of individual neighborhood is determined by 
            taking into account the objective function characteristics.
            5- The search domain and the radius ε of the individual neighborhood are divided per r_red
            6- The radius ε of individual neighborhood is equal to δ / r_neighb
                the length δ of the smallest edge of the initial hyperrectangular search domain;
                the neighborhood parameter r_neighb = the product of the initial population size *
                                                      the dimension of the problem
        """
        self.r_neighb = self.entity_num * len(self.limits_list)
        self.radius_list = [(limit[1] - limit[0]) / self.r_neighb for limit in self.limits_list]

        # ------------------- Generation of the initial population -------------------
        # A generated individual is accepted as an initial individual, if it does not belong to the
        # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
        all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
        while len(all_entity_x_list) < self.entity_num:
            x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Evaluate whether the point is in any entities' neighborhood
            all_flag_list = []
            for e_x_list in all_entity_x_list:
                e_flag_list = []
                for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                    if abs(x - e_x) >= radius:
                        e_flag_list.append(True)
                if False in e_flag_list:
                    continue
                else:
                    all_flag_list.append(True)
            all_entity_x_list.append(x_list)

        self.entity_list = []
        for x_list in all_entity_x_list:
            entity = self.Entity(self.limits_list, self.fitness_function)
            entity.x_list = x_list
            entity.update()
            self.entity_list.append(copy.deepcopy(entity))
        # ------------------- Generation of the initial population -------------------
        self.global_best_entity = self.Entity(self.limits_list, fitness_function)

    def evolve(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        for iter in range(self.iter_num):
            self.entity_list.sort(key=lambda entity:entity.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            print('Iter:', iter,\
                  '\nCurrent limit:',self.limits_list,\
                  '\nCurrent best entity, x:{0}, fitness:{1}'.format(cur_best_entity.x_list, cur_best_entity.fitness))
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # -------------------------------------- Generate new population --------------------------------------
            new_entity_list = []
            # 1-Generate how many offsprings?
            # I can not understand paper0, so, I take my way: generate half of entity_num offsprings
            for i in range(int(math.ceil(self.entity_num * 0.5))):
                # -------------- Selection, select two parents by roulette-wheel selection --------------
                fitness_list = [entity.fitness for entity in self.entity_list]
                inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
                parent_indexs = []
                while len(parent_indexs) < 2:
                    random_pointer = random.uniform(0, sum(inv_fitness_list))
                    inv_fitness_sum = 0.0
                    for index, inv_fitness in enumerate(inv_fitness_list):
                        inv_fitness_sum += inv_fitness
                        if random_pointer < inv_fitness_sum:
                            parent_indexs.append(index)
                            break
                parent_i = self.entity_list[parent_indexs[0]]
                parent_j = self.entity_list[parent_indexs[1]]
                tmp_i_x_list = copy.deepcopy(parent_i.x_list)
                tmp_j_x_list = copy.deepcopy(parent_j.x_list)
                # -------------- Selection, select two parents by roulette-wheel selection --------------

                # -------------- Recombination --------------
                if random.random() < self.p_crossover:
                    random_cross_index = random.randint(0, len(self.limits_list) - 1)
                    random_M = random.randint(1, 1000)
                    d_x = tmp_i_x_list[random_cross_index] / random_M
                    d_y = tmp_j_x_list[random_cross_index] / random_M
                    tmp_i_x_list[random_cross_index] = tmp_i_x_list[random_cross_index] + d_y - d_x
                    tmp_j_x_list[random_cross_index] = tmp_j_x_list[random_cross_index] - d_y + d_x

                    # All the components situated on the left of this crossing point are not affected,
                    # those situated after the crossing point are exchanged
                    tmp_i_x_list = tmp_i_x_list[: random_cross_index + 1] + tmp_j_x_list[random_cross_index + 1 :]
                    tmp_j_x_list = tmp_j_x_list[: random_cross_index + 1] + copy.deepcopy(parent_i.x_list[random_cross_index + 1 :])
                # -------------- Recombination --------------

                # -------------- Mutation --------------
                if random.random() < self.p_mutate:
                    for t_x_list in [tmp_i_x_list, tmp_j_x_list]:
                        random_mutate_index = random.randint(0, len(self.limits_list) - 1)
                        limit = self.limits_list[random_mutate_index]
                        d_mutate_x = (limit[1] - limit[0]) / random.randint(1, 10)
                        t_x_list[random_mutate_index] = t_x_list[random_mutate_index]\
                                                        + pow(-1, random.randint(1, 2)) * d_mutate_x
                # -------------- Mutation --------------
                tmp_entity_i = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity_i.x_list = tmp_i_x_list
                tmp_entity_i.update()
                tmp_entity_j = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity_j.x_list = tmp_j_x_list
                tmp_entity_j.update()
                new_entity_list.extend([tmp_entity_i, tmp_entity_j])
                # When to stop diversification and when to starts intensification?

            # 2-How to merge offsprings and parents?
            # I can not understand paper0, so, I take my way: merge old population with offsprings,
            # and select the first entity_num ones
            self.entity_list = sorted(self.entity_list + new_entity_list, key=lambda en : en.fitness, reverse=False)[: self.entity_num]
            # -------------------------------------- Generate new population --------------------------------------

            # -------------------------------------- Update global settings --------------------------------------
            limit_scale_factor = max((self.iter_num - iter) / self.iter_num, 0.05)
            tmp_linear_limit_list = [[limit[0] + d_s, limit[1] - d_s] \
                                     for limit, d_s in zip(self.limits_list, self.limit_decrease_step_list)]

            # Narrow limitation
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_linear_limit = tmp_linear_limit_list[i]
                x_col_list = []
                for en in self.entity_list:
                    x_col_list.append(en.x_list[i])
                tmp_limit_list.append([min(min(x_col_list), tmp_linear_limit[0]), max(max(x_col_list), tmp_linear_limit[1])])
            self.limits_list = copy.deepcopy(tmp_limit_list)
            # k
            self.k = self.I_K * (self.iter_num - iter) / self.iter_num
            # radius
            self.radius_list = [r * (self.iter_num - iter) / self.iter_num for r in self.radius_list]
            # mutation ratio
            self.p_mutate = self.I_P_mutate * pow((self.iter_num - iter) / self.iter_num, 3)
            # -------------------------------------- Update global settings --------------------------------------
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 50
#     entity_num = 10
#     dim = 2
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     cga = CGA_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = cga.evolve()
#     print('Best entity position:', cga.global_best_entity.x_list)
#     print('Fitness:', cga.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#     for i, c_en, g_en in zip(range(iter_num), current_best_entity_list, global_best_entity_list):
#         print(i, c_en.fitness, g_en.fitness)
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class CGA_2:
    """
    Continuous Genetic Algorithm
    Refer:
        Book:
            Book0: 遗传算法原理及应用
        Paper:
            paper0: A Continuous Genetic Algorithm Designed for the Global Optimization of Multimodal Functions
                Its citation is more than 300. So, it is trusted
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        1
            Previous version.0 is at 'from GA_pack.GAs.basic.sga'
            This version use float variable, and is more convenient and compact with other algorithms
            每轮迭代中，参数的上下界为该轮所有个体的坐标的极值，使得参数的上下界迅速减少，迭代十几次就出现上下界相同的情况
        2
            每轮迭代中，参数的上下界逐步线性搜小，同时考虑该轮所有个体坐标的上下界，取二者中的较大值
            随着迭代，其他个体逐渐向最优个体靠拢并逐渐变成一样（x_list, fitness），导致轮盘赌选择父母的时候陷入死循环
        3
            当所有个体均移动至相同的位置，在当前的参数范围下重新生成一批个体，继续迭代
    """
    class Entity:
        def __init__(self, tmp_limits_list, fitness_function):
            self.tmp_limits_list = tmp_limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.tmp_limits_list]
            self.fitness = self.fitness_function(self.x_list)

        def update(self):
            for i, limit in enumerate(self.tmp_limits_list):
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        # MaxIter
        self.iter_num = iter_num
        """
        Setting of part parameters is described in paper0-section 3
            Initialization. The parameters fixed at the beginning are the following ones:
                – search domain of each function variable;
                – starting point (randomly generated);
                – initial population (see on Section 2.2.1)
            The parameters which are automatically built by using the parameters fixed at the beginning are the following ones:
                – the neighborhood parameter r_neighb;
                – the length δ of the smallest edge of the initial hyperrectangular search domain;
                – the radius ε of the individual neighborhood;
                – the maximal number of successive generations without any improvement of the best
                known objective function value MaxGen;
                – the maximal number of iterations MaxIter;
                – the accepted distance between the best point and the generated individuals P_abs.
            Fixed parameters are:
                the initial population size IPopSize (equal to 30)
                the final population size FPopSize (equal to 10)
                the initial mutation IPMut and crossover probabilities PXcross (respectively equal to 0.9 and 0.85)
                the reduction parameters P_red and d_PopSize (respectively equal to 2 and 5)
        """
        # Equals to the size of initial population, IPopSize
        self.entity_num = entity_num
        # Size of final population
        self.FPopSize = int(0.33 * self.entity_num)

        self.p_crossover = 0.85
        # initial value of Mutation probability
        self.I_P_mutate = 0.9
        self.p_mutate = self.I_P_mutate
        self.I_K = 1.0
        self.k = self.I_K

        # Initial Limitation
        self.I_limits_list = limits_list
        self.limits_list = limits_list
        self.limit_decrease_step_list = [(limit[1] - limit[0]) * 0.9 / (2 * self.iter_num) for limit in self.limits_list]
        self.fitness_function = fitness_function

        """
        Information about radius (ε) in paper0:
            1- A ball B(s, ε), centered on s with the radius ε, contains all points s' such as: || s' - s || <= ε
            2- In the intensification module, we reduce the search domain, the population size, the neighborhood radius,
            the probability of mutation, we generate the new population around the best point found, and we restart the 
            genetic procedure inside this new search domain.
            3- the highest distance between the best point up to now BestPoint and the generated individuals Ind_i 
            is smaller than a given neighborhood radius r_abs
            4- These parameters are fixed empirically, but the radius ε of individual neighborhood is determined by 
            taking into account the objective function characteristics.
            5- The search domain and the radius ε of the individual neighborhood are divided per r_red
            6- The radius ε of individual neighborhood is equal to δ / r_neighb
                the length δ of the smallest edge of the initial hyperrectangular search domain;
                the neighborhood parameter r_neighb = the product of the initial population size *
                                                      the dimension of the problem
        """
        self.r_neighb = self.entity_num * len(self.limits_list)
        self.radius_list = [(limit[1] - limit[0]) / self.r_neighb for limit in self.limits_list]

        # Reduction time
        self.reduction_num = 0
        # ------------------- Generation of the initial population -------------------
        # A generated individual is accepted as an initial individual, if it does not belong to the
        # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
        all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
        while len(all_entity_x_list) < self.entity_num:
            x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Evaluate whether the point is in any entities' neighborhood
            all_flag_list = []
            for e_x_list in all_entity_x_list:
                e_flag_list = []
                for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                    if abs(x - e_x) >= radius:
                        e_flag_list.append(True)
                if False in e_flag_list:
                    continue
                else:
                    all_flag_list.append(True)
            all_entity_x_list.append(x_list)

        self.entity_list = []
        for x_list in all_entity_x_list:
            entity = self.Entity(self.limits_list, self.fitness_function)
            entity.x_list = x_list
            entity.update()
            self.entity_list.append(copy.deepcopy(entity))
        # ------------------- Generation of the initial population -------------------
        self.global_best_entity = self.Entity(self.limits_list, fitness_function)

    def evolve(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        for iter in range(self.iter_num):

            fitness_list = [entity.fitness for entity in self.entity_list]
            inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
            # When all entity have the same position and fitness, regenerate the whole population
            # if sum(inv_fitness_list) == 0.0:
            if len(set(fitness_list)) == 1:
                self.reduction_num += 1
                old_best_entity = copy.deepcopy(self.entity_list[0])
                # ------------------- Generation of the initial population -------------------
                # A generated individual is accepted as an initial individual, if it does not belong to the
                # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
                all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
                while len(all_entity_x_list) < self.entity_num:
                    x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
                    # Evaluate whether the point is in any entities' neighborhood
                    all_flag_list = []
                    for e_x_list in all_entity_x_list:
                        e_flag_list = []
                        for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                            if abs(x - e_x) >= radius:
                                e_flag_list.append(True)
                        if False in e_flag_list:
                            continue
                        else:
                            all_flag_list.append(True)
                    all_entity_x_list.append(x_list)

                self.entity_list = []
                for x_list in all_entity_x_list:
                    entity = self.Entity(self.limits_list, self.fitness_function)
                    entity.x_list = x_list
                    entity.update()
                    self.entity_list.append(copy.deepcopy(entity))

                self.entity_list.sort(key=lambda en: en.fitness, reverse=False)
                if old_best_entity.fitness < self.entity_list[-1].fitness:
                    self.entity_list[-1] = copy.deepcopy(old_best_entity)
                # ------------------- Generation of the initial population -------------------
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            # print('Iter:', iter,\
            #       '\nCurrent limit:',self.limits_list,\
            #       '\nCurrent best entity, x:{0}, fitness:{1}'.format(cur_best_entity.x_list, cur_best_entity.fitness))
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # -------------------------------------- Generate new population --------------------------------------
            new_entity_list = []
            # 1-Generate how many offsprings?
            # I can not understand paper0, so, I take my way: generate half of entity_num offsprings
            for i in range(int(math.ceil(self.entity_num * 0.5))):
                # -------------- Selection, select two parents by roulette-wheel selection --------------
                fitness_list = [entity.fitness for entity in self.entity_list]
                inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
                parent_indexs = []
                while len(parent_indexs) < 2:
                    random_pointer = random.uniform(0, sum(inv_fitness_list))
                    inv_fitness_sum = 0.0
                    for index, inv_fitness in enumerate(inv_fitness_list):
                        inv_fitness_sum += inv_fitness
                        if random_pointer < inv_fitness_sum:
                            parent_indexs.append(index)
                            break
                parent_i = self.entity_list[parent_indexs[0]]
                parent_j = self.entity_list[parent_indexs[1]]
                tmp_i_x_list = copy.deepcopy(parent_i.x_list)
                tmp_j_x_list = copy.deepcopy(parent_j.x_list)
                # -------------- Selection, select two parents by roulette-wheel selection --------------

                # -------------- Recombination --------------
                if random.random() < self.p_crossover:
                    random_cross_index = random.randint(0, len(self.limits_list) - 1)
                    random_M = random.randint(1, 1000)
                    d_x = tmp_i_x_list[random_cross_index] / random_M
                    d_y = tmp_j_x_list[random_cross_index] / random_M
                    tmp_i_x_list[random_cross_index] = tmp_i_x_list[random_cross_index] + d_y - d_x
                    tmp_j_x_list[random_cross_index] = tmp_j_x_list[random_cross_index] - d_y + d_x

                    # All the components situated on the left of this crossing point are not affected,
                    # those situated after the crossing point are exchanged
                    tmp_i_x_list = tmp_i_x_list[: random_cross_index + 1] + tmp_j_x_list[random_cross_index + 1 :]
                    tmp_j_x_list = tmp_j_x_list[: random_cross_index + 1] + copy.deepcopy(parent_i.x_list[random_cross_index + 1 :])
                # -------------- Recombination --------------

                # -------------- Mutation --------------
                if random.random() < self.p_mutate:
                    for t_x_list in [tmp_i_x_list, tmp_j_x_list]:
                        random_mutate_index = random.randint(0, len(self.limits_list) - 1)
                        limit = self.limits_list[random_mutate_index]
                        d_mutate_x = (limit[1] - limit[0]) / random.randint(1, 10)
                        t_x_list[random_mutate_index] = t_x_list[random_mutate_index]\
                                                        + pow(-1, random.randint(1, 2)) * d_mutate_x
                # -------------- Mutation --------------
                tmp_entity_i = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity_i.x_list = tmp_i_x_list
                tmp_entity_i.update()
                tmp_entity_j = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity_j.x_list = tmp_j_x_list
                tmp_entity_j.update()
                new_entity_list.extend([tmp_entity_i, tmp_entity_j])
                # When to stop diversification and when to starts intensification?

            # 2-How to merge offsprings and parents?
            # I can not understand paper0, so, I take my way: merge old population with offsprings,
            # and select the first entity_num ones
            self.entity_list = sorted(self.entity_list + new_entity_list, key=lambda en : en.fitness, reverse=False)[: self.entity_num]
            # -------------------------------------- Generate new population --------------------------------------

            # -------------------------------------- Update global settings --------------------------------------
            limit_scale_factor = max((self.iter_num - iter) / self.iter_num, 0.05)
            tmp_linear_limit_list = [[limit[0] + d_s, limit[1] - d_s] \
                                     for limit, d_s in zip(self.limits_list, self.limit_decrease_step_list)]

            # Narrow limitation
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_linear_limit = tmp_linear_limit_list[i]
                x_col_list = []
                for en in self.entity_list:
                    x_col_list.append(en.x_list[i])
                tmp_limit_list.append([min(min(x_col_list), tmp_linear_limit[0]), max(max(x_col_list), tmp_linear_limit[1])])
            self.limits_list = copy.deepcopy(tmp_limit_list)
            # k
            self.k = self.I_K * (self.iter_num - iter) / self.iter_num
            # radius
            self.radius_list = [r * (self.iter_num - iter) / self.iter_num for r in self.radius_list]
            # mutation ratio
            self.p_mutate = self.I_P_mutate * math.exp(- self.reduction_num)
            # -------------------------------------- Update global settings --------------------------------------
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 30
#     dim = 5
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     cga = CGA_2(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = cga.evolve()
#     print('Best entity position:', cga.global_best_entity.x_list)
#     print('Fitness:', cga.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#     # for i, c_en, g_en in zip(range(iter_num), current_best_entity_list, global_best_entity_list):
#     #     print(i, c_en.fitness, g_en.fitness)
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class CGA_EIS:
    """
    Continuous Genetic Algorithm
    Refer:
        Book:
            Book0: 遗传算法原理及应用
        Paper:
            paper0: A Continuous Genetic Algorithm Designed for the Global Optimization of Multimodal Functions
                Its citation is more than 300. So, it is trusted
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        1
            Previous version.0 is at 'from GA_pack.GAs.basic.sga'
            This version use float variable, and is more convenient and compact with other algorithms
            每轮迭代中，参数的上下界为该轮所有个体的坐标的极值，使得参数的上下界迅速减少，迭代十几次就出现上下界相同的情况
        2
            每轮迭代中，参数的上下界逐步线性搜小，同时考虑该轮所有个体坐标的上下界，取二者中的较大值
            随着迭代，其他个体逐渐向最优个体靠拢并逐渐变成一样（x_list, fitness），导致轮盘赌选择父母的时候陷入死循环
        3
            当所有个体均移动至相同的位置，在当前的参数范围下重新生成一批个体，继续迭代
    """
    class Entity:
        def __init__(self, exp_data_dict, tmp_limits_list, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.tmp_limits_list = tmp_limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.tmp_limits_list]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, limit in enumerate(self.tmp_limits_list):
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        # Initial Limitation
        self.I_limits_list = exp_data_dict['limit']
        self.limits_list = exp_data_dict['limit']

        # MaxIter
        self.iter_num = iter_num
        """
        Setting of part parameters is described in paper0-section 3
            Initialization. The parameters fixed at the beginning are the following ones:
                – search domain of each function variable;
                – starting point (randomly generated);
                – initial population (see on Section 2.2.1)
            The parameters which are automatically built by using the parameters fixed at the beginning are the following ones:
                – the neighborhood parameter r_neighb;
                – the length δ of the smallest edge of the initial hyperrectangular search domain;
                – the radius ε of the individual neighborhood;
                – the maximal number of successive generations without any improvement of the best
                known objective function value MaxGen;
                – the maximal number of iterations MaxIter;
                – the accepted distance between the best point and the generated individuals P_abs.
            Fixed parameters are:
                the initial population size IPopSize (equal to 30)
                the final population size FPopSize (equal to 10)
                the initial mutation IPMut and crossover probabilities PXcross (respectively equal to 0.9 and 0.85)
                the reduction parameters P_red and d_PopSize (respectively equal to 2 and 5)
        """
        # Equals to the size of initial population, IPopSize
        self.entity_num = entity_num
        # Size of final population
        self.FPopSize = int(0.33 * self.entity_num)

        self.p_crossover = 0.85
        # initial value of Mutation probability
        self.I_P_mutate = 0.9
        self.p_mutate = self.I_P_mutate
        self.I_K = 1.0
        self.k = self.I_K

        self.limit_decrease_step_list = [(limit[1] - limit[0]) * 0.9 / (2 * self.iter_num) for limit in self.limits_list]
        self.fitness_function = fitness_function

        """
        Information about radius (ε) in paper0:
            1- A ball B(s, ε), centered on s with the radius ε, contains all points s' such as: || s' - s || <= ε
            2- In the intensification module, we reduce the search domain, the population size, the neighborhood radius,
            the probability of mutation, we generate the new population around the best point found, and we restart the 
            genetic procedure inside this new search domain.
            3- the highest distance between the best point up to now BestPoint and the generated individuals Ind_i 
            is smaller than a given neighborhood radius r_abs
            4- These parameters are fixed empirically, but the radius ε of individual neighborhood is determined by 
            taking into account the objective function characteristics.
            5- The search domain and the radius ε of the individual neighborhood are divided per r_red
            6- The radius ε of individual neighborhood is equal to δ / r_neighb
                the length δ of the smallest edge of the initial hyperrectangular search domain;
                the neighborhood parameter r_neighb = the product of the initial population size *
                                                      the dimension of the problem
        """
        self.r_neighb = self.entity_num * len(self.limits_list)
        self.radius_list = [(limit[1] - limit[0]) / self.r_neighb for limit in self.limits_list]

        # Reduction time
        self.reduction_num = 0
        # ------------------- Generation of the initial population -------------------
        # A generated individual is accepted as an initial individual, if it does not belong to the
        # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
        all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
        while len(all_entity_x_list) < self.entity_num:
            x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Evaluate whether the point is in any entities' neighborhood
            all_flag_list = []
            for e_x_list in all_entity_x_list:
                e_flag_list = []
                for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                    if abs(x - e_x) >= radius:
                        e_flag_list.append(True)
                if False in e_flag_list:
                    continue
                else:
                    all_flag_list.append(True)
            all_entity_x_list.append(x_list)

        self.entity_list = []
        for x_list in all_entity_x_list:
            entity = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
            entity.x_list = x_list
            entity.update()
            self.entity_list.append(copy.deepcopy(entity))
        # ------------------- Generation of the initial population -------------------
        self.global_best_entity = self.Entity(self.exp_data_dict, self.limits_list, fitness_function)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            fitness_list = [entity.fitness for entity in self.entity_list]
            inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
            # When all entity have the same position and fitness, regenerate the whole population
            # if sum(inv_fitness_list) == 0.0:
            if len(set(fitness_list)) == 1:
                self.reduction_num += 1
                old_best_entity = copy.deepcopy(self.entity_list[0])
                # ------------------- Generation of the initial population -------------------
                # A generated individual is accepted as an initial individual, if it does not belong to the
                # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
                all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
                while len(all_entity_x_list) < self.entity_num:
                    x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
                    # Evaluate whether the point is in any entities' neighborhood
                    all_flag_list = []
                    for e_x_list in all_entity_x_list:
                        e_flag_list = []
                        for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                            if abs(x - e_x) >= radius:
                                e_flag_list.append(True)
                        if False in e_flag_list:
                            continue
                        else:
                            all_flag_list.append(True)
                    all_entity_x_list.append(x_list)

                self.entity_list = []
                for x_list in all_entity_x_list:
                    entity = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
                    entity.x_list = x_list
                    entity.update()
                    self.entity_list.append(copy.deepcopy(entity))

                self.entity_list.sort(key=lambda en: en.fitness, reverse=False)
                if old_best_entity.fitness < self.entity_list[-1].fitness:
                    self.entity_list[-1] = copy.deepcopy(old_best_entity)
                # ------------------- Generation of the initial population -------------------
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            # print('Iter:', iter,\
            #       '\nCurrent limit:',self.limits_list,\
            #       '\nCurrent best entity, x:{0}, fitness:{1}'.format(cur_best_entity.x_list, cur_best_entity.fitness))
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # -------------------------------------- Generate new population --------------------------------------
            new_entity_list = []
            # 1-Generate how many offsprings?
            # I can not understand paper0, so, I take my way: generate half of entity_num offsprings
            for i in range(int(math.ceil(self.entity_num * 0.5))):
                # -------------- Selection, select two parents by roulette-wheel selection --------------
                fitness_list = [entity.fitness for entity in self.entity_list]
                inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
                parent_indexs = []
                while len(parent_indexs) < 2:
                    random_pointer = random.uniform(0, sum(inv_fitness_list))
                    inv_fitness_sum = 0.0
                    for index, inv_fitness in enumerate(inv_fitness_list):
                        inv_fitness_sum += inv_fitness
                        if random_pointer < inv_fitness_sum:
                            parent_indexs.append(index)
                            break
                parent_i = self.entity_list[parent_indexs[0]]
                parent_j = self.entity_list[parent_indexs[1]]
                tmp_i_x_list = copy.deepcopy(parent_i.x_list)
                tmp_j_x_list = copy.deepcopy(parent_j.x_list)
                # -------------- Selection, select two parents by roulette-wheel selection --------------

                # -------------- Recombination --------------
                if random.random() < self.p_crossover:
                    random_cross_index = random.randint(0, len(self.limits_list) - 1)
                    random_M = random.randint(1, 1000)
                    d_x = tmp_i_x_list[random_cross_index] / random_M
                    d_y = tmp_j_x_list[random_cross_index] / random_M
                    tmp_i_x_list[random_cross_index] = tmp_i_x_list[random_cross_index] + d_y - d_x
                    tmp_j_x_list[random_cross_index] = tmp_j_x_list[random_cross_index] - d_y + d_x

                    # All the components situated on the left of this crossing point are not affected,
                    # those situated after the crossing point are exchanged
                    tmp_i_x_list = tmp_i_x_list[: random_cross_index + 1] + tmp_j_x_list[random_cross_index + 1 :]
                    tmp_j_x_list = tmp_j_x_list[: random_cross_index + 1] + copy.deepcopy(parent_i.x_list[random_cross_index + 1 :])
                # -------------- Recombination --------------

                # -------------- Mutation --------------
                if random.random() < self.p_mutate:
                    for t_x_list in [tmp_i_x_list, tmp_j_x_list]:
                        random_mutate_index = random.randint(0, len(self.limits_list) - 1)
                        limit = self.limits_list[random_mutate_index]
                        d_mutate_x = (limit[1] - limit[0]) / random.randint(1, 10)
                        t_x_list[random_mutate_index] = t_x_list[random_mutate_index]\
                                                        + pow(-1, random.randint(1, 2)) * d_mutate_x
                # -------------- Mutation --------------
                tmp_entity_i = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
                tmp_entity_i.x_list = tmp_i_x_list
                tmp_entity_i.update()
                tmp_entity_j = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
                tmp_entity_j.x_list = tmp_j_x_list
                tmp_entity_j.update()
                new_entity_list.extend([tmp_entity_i, tmp_entity_j])
                # When to stop diversification and when to starts intensification?

            # 2-How to merge offsprings and parents?
            # I can not understand paper0, so, I take my way: merge old population with offsprings,
            # and select the first entity_num ones
            self.entity_list = sorted(self.entity_list + new_entity_list, key=lambda en : en.fitness, reverse=False)[: self.entity_num]
            # -------------------------------------- Generate new population --------------------------------------

            # -------------------------------------- Update global settings --------------------------------------
            tmp_linear_limit_list = [[limit[0] + d_s, limit[1] - d_s] \
                                     for limit, d_s in zip(self.limits_list, self.limit_decrease_step_list)]

            # Narrow limitation
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_linear_limit = tmp_linear_limit_list[i]
                x_col_list = []
                for en in self.entity_list:
                    x_col_list.append(en.x_list[i])
                tmp_limit_list.append([min(min(x_col_list), tmp_linear_limit[0]), max(max(x_col_list), tmp_linear_limit[1])])
            self.limits_list = copy.deepcopy(tmp_limit_list)
            # k
            self.k = self.I_K * (self.iter_num - iter) / self.iter_num
            # radius
            self.radius_list = [r * (self.iter_num - iter) / self.iter_num for r in self.radius_list]
            # mutation ratio
            self.p_mutate = self.I_P_mutate * math.exp(- self.reduction_num)
            # -------------------------------------- Update global settings --------------------------------------

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_entity_list[-2].x_list, global_best_entity_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return cur_best_entity_list, global_best_entity_list, iter, chi_squared

class CGA_EIS_access:
    """
    Continuous Genetic Algorithm
    Refer:
        Book:
            Book0: 遗传算法原理及应用
        Paper:
            paper0: A Continuous Genetic Algorithm Designed for the Global Optimization of Multimodal Functions
                Its citation is more than 300. So, it is trusted
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        1
            Previous version.0 is at 'from GA_pack.GAs.basic.sga'
            This version use float variable, and is more convenient and compact with other algorithms
            每轮迭代中，参数的上下界为该轮所有个体的坐标的极值，使得参数的上下界迅速减少，迭代十几次就出现上下界相同的情况
        2
            每轮迭代中，参数的上下界逐步线性搜小，同时考虑该轮所有个体坐标的上下界，取二者中的较大值
            随着迭代，其他个体逐渐向最优个体靠拢并逐渐变成一样（x_list, fitness），导致轮盘赌选择父母的时候陷入死循环
        3
            当所有个体均移动至相同的位置，在当前的参数范围下重新生成一批个体，继续迭代
    """
    class Entity:
        def __init__(self, exp_data_dict, tmp_limits_list, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.tmp_limits_list = tmp_limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.tmp_limits_list]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, limit in enumerate(self.tmp_limits_list):
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        # Initial Limitation
        self.I_limits_list = exp_data_dict['limit']
        self.limits_list = exp_data_dict['limit']

        # MaxIter
        self.iter_num = iter_num
        """
        Setting of part parameters is described in paper0-section 3
            Initialization. The parameters fixed at the beginning are the following ones:
                – search domain of each function variable;
                – starting point (randomly generated);
                – initial population (see on Section 2.2.1)
            The parameters which are automatically built by using the parameters fixed at the beginning are the following ones:
                – the neighborhood parameter r_neighb;
                – the length δ of the smallest edge of the initial hyperrectangular search domain;
                – the radius ε of the individual neighborhood;
                – the maximal number of successive generations without any improvement of the best
                known objective function value MaxGen;
                – the maximal number of iterations MaxIter;
                – the accepted distance between the best point and the generated individuals P_abs.
            Fixed parameters are:
                the initial population size IPopSize (equal to 30)
                the final population size FPopSize (equal to 10)
                the initial mutation IPMut and crossover probabilities PXcross (respectively equal to 0.9 and 0.85)
                the reduction parameters P_red and d_PopSize (respectively equal to 2 and 5)
        """
        # Equals to the size of initial population, IPopSize
        self.entity_num = entity_num
        # Size of final population
        self.FPopSize = int(0.33 * self.entity_num)

        self.p_crossover = 0.85
        # initial value of Mutation probability
        self.I_P_mutate = 0.9
        self.p_mutate = self.I_P_mutate
        self.I_K = 1.0
        self.k = self.I_K

        self.limit_decrease_step_list = [(limit[1] - limit[0]) * 0.9 / (2 * self.iter_num) for limit in self.limits_list]
        self.fitness_function = fitness_function

        """
        Information about radius (ε) in paper0:
            1- A ball B(s, ε), centered on s with the radius ε, contains all points s' such as: || s' - s || <= ε
            2- In the intensification module, we reduce the search domain, the population size, the neighborhood radius,
            the probability of mutation, we generate the new population around the best point found, and we restart the 
            genetic procedure inside this new search domain.
            3- the highest distance between the best point up to now BestPoint and the generated individuals Ind_i 
            is smaller than a given neighborhood radius r_abs
            4- These parameters are fixed empirically, but the radius ε of individual neighborhood is determined by 
            taking into account the objective function characteristics.
            5- The search domain and the radius ε of the individual neighborhood are divided per r_red
            6- The radius ε of individual neighborhood is equal to δ / r_neighb
                the length δ of the smallest edge of the initial hyperrectangular search domain;
                the neighborhood parameter r_neighb = the product of the initial population size *
                                                      the dimension of the problem
        """
        self.r_neighb = self.entity_num * len(self.limits_list)
        self.radius_list = [(limit[1] - limit[0]) / self.r_neighb for limit in self.limits_list]

        # Reduction time
        self.reduction_num = 0
        # ------------------- Generation of the initial population -------------------
        # A generated individual is accepted as an initial individual, if it does not belong to the
        # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
        all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
        while len(all_entity_x_list) < self.entity_num:
            x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Evaluate whether the point is in any entities' neighborhood
            all_flag_list = []
            for e_x_list in all_entity_x_list:
                e_flag_list = []
                for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                    if abs(x - e_x) >= radius:
                        e_flag_list.append(True)
                if False in e_flag_list:
                    continue
                else:
                    all_flag_list.append(True)
            all_entity_x_list.append(x_list)

        self.entity_list = []
        for x_list in all_entity_x_list:
            entity = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
            entity.x_list = x_list
            entity.update()
            self.entity_list.append(copy.deepcopy(entity))
        # ------------------- Generation of the initial population -------------------
        self.global_best_entity = self.Entity(self.exp_data_dict, self.limits_list, fitness_function)

    def search(self, res_fn, start_time):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            fitness_list = [entity.fitness for entity in self.entity_list]
            inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
            # When all entity have the same position and fitness, regenerate the whole population
            # if sum(inv_fitness_list) == 0.0:
            if len(set(fitness_list)) == 1:
                self.reduction_num += 1
                old_best_entity = copy.deepcopy(self.entity_list[0])
                # ------------------- Generation of the initial population -------------------
                # A generated individual is accepted as an initial individual, if it does not belong to the
                # neighborhood of any already selected individual (see figure 2, in the two-dimension case).
                all_entity_x_list = [[random.uniform(limit[0], limit[1]) for limit in self.limits_list]]
                while len(all_entity_x_list) < self.entity_num:
                    x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
                    # Evaluate whether the point is in any entities' neighborhood
                    all_flag_list = []
                    for e_x_list in all_entity_x_list:
                        e_flag_list = []
                        for x, e_x, radius in zip(x_list, e_x_list, self.radius_list):
                            if abs(x - e_x) >= radius:
                                e_flag_list.append(True)
                        if False in e_flag_list:
                            continue
                        else:
                            all_flag_list.append(True)
                    all_entity_x_list.append(x_list)

                self.entity_list = []
                for x_list in all_entity_x_list:
                    entity = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
                    entity.x_list = x_list
                    entity.update()
                    self.entity_list.append(copy.deepcopy(entity))

                self.entity_list.sort(key=lambda en: en.fitness, reverse=False)
                if old_best_entity.fitness < self.entity_list[-1].fitness:
                    self.entity_list[-1] = copy.deepcopy(old_best_entity)
                # ------------------- Generation of the initial population -------------------
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            # print('Iter:', iter,\
            #       '\nCurrent limit:',self.limits_list,\
            #       '\nCurrent best entity, x:{0}, fitness:{1}'.format(cur_best_entity.x_list, cur_best_entity.fitness))
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # -------------------------------------- Generate new population --------------------------------------
            new_entity_list = []
            # 1-Generate how many offsprings?
            # I can not understand paper0, so, I take my way: generate half of entity_num offsprings
            for i in range(int(math.ceil(self.entity_num * 0.5))):
                # -------------- Selection, select two parents by roulette-wheel selection --------------
                fitness_list = [entity.fitness for entity in self.entity_list]
                inv_fitness_list = [max(fitness_list) - fitness for fitness in fitness_list]
                parent_indexs = []
                while len(parent_indexs) < 2:
                    random_pointer = random.uniform(0, sum(inv_fitness_list))
                    inv_fitness_sum = 0.0
                    for index, inv_fitness in enumerate(inv_fitness_list):
                        inv_fitness_sum += inv_fitness
                        if random_pointer < inv_fitness_sum:
                            parent_indexs.append(index)
                            break
                parent_i = self.entity_list[parent_indexs[0]]
                parent_j = self.entity_list[parent_indexs[1]]
                tmp_i_x_list = copy.deepcopy(parent_i.x_list)
                tmp_j_x_list = copy.deepcopy(parent_j.x_list)
                # -------------- Selection, select two parents by roulette-wheel selection --------------

                # -------------- Recombination --------------
                if random.random() < self.p_crossover:
                    random_cross_index = random.randint(0, len(self.limits_list) - 1)
                    random_M = random.randint(1, 1000)
                    d_x = tmp_i_x_list[random_cross_index] / random_M
                    d_y = tmp_j_x_list[random_cross_index] / random_M
                    tmp_i_x_list[random_cross_index] = tmp_i_x_list[random_cross_index] + d_y - d_x
                    tmp_j_x_list[random_cross_index] = tmp_j_x_list[random_cross_index] - d_y + d_x

                    # All the components situated on the left of this crossing point are not affected,
                    # those situated after the crossing point are exchanged
                    tmp_i_x_list = tmp_i_x_list[: random_cross_index + 1] + tmp_j_x_list[random_cross_index + 1 :]
                    tmp_j_x_list = tmp_j_x_list[: random_cross_index + 1] + copy.deepcopy(parent_i.x_list[random_cross_index + 1 :])
                # -------------- Recombination --------------

                # -------------- Mutation --------------
                if random.random() < self.p_mutate:
                    for t_x_list in [tmp_i_x_list, tmp_j_x_list]:
                        random_mutate_index = random.randint(0, len(self.limits_list) - 1)
                        limit = self.limits_list[random_mutate_index]
                        d_mutate_x = (limit[1] - limit[0]) / random.randint(1, 10)
                        t_x_list[random_mutate_index] = t_x_list[random_mutate_index]\
                                                        + pow(-1, random.randint(1, 2)) * d_mutate_x
                # -------------- Mutation --------------
                tmp_entity_i = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
                tmp_entity_i.x_list = tmp_i_x_list
                tmp_entity_i.update()
                tmp_entity_j = self.Entity(self.exp_data_dict, self.limits_list, self.fitness_function)
                tmp_entity_j.x_list = tmp_j_x_list
                tmp_entity_j.update()
                new_entity_list.extend([tmp_entity_i, tmp_entity_j])
                # When to stop diversification and when to starts intensification?

            # 2-How to merge offsprings and parents?
            # I can not understand paper0, so, I take my way: merge old population with offsprings,
            # and select the first entity_num ones
            self.entity_list = sorted(self.entity_list + new_entity_list, key=lambda en : en.fitness, reverse=False)[: self.entity_num]
            # -------------------------------------- Generate new population --------------------------------------

            # -------------------------------------- Update global settings --------------------------------------
            tmp_linear_limit_list = [[limit[0] + d_s, limit[1] - d_s] \
                                     for limit, d_s in zip(self.limits_list, self.limit_decrease_step_list)]

            # Narrow limitation
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_linear_limit = tmp_linear_limit_list[i]
                x_col_list = []
                for en in self.entity_list:
                    x_col_list.append(en.x_list[i])
                tmp_limit_list.append([min(min(x_col_list), tmp_linear_limit[0]), max(max(x_col_list), tmp_linear_limit[1])])
            self.limits_list = copy.deepcopy(tmp_limit_list)
            # k
            self.k = self.I_K * (self.iter_num - iter) / self.iter_num
            # radius
            self.radius_list = [r * (self.iter_num - iter) / self.iter_num for r in self.radius_list]
            # mutation ratio
            self.p_mutate = self.I_P_mutate * math.exp(- self.reduction_num)
            # -------------------------------------- Update global settings --------------------------------------

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_entity_list[-2].x_list, global_best_entity_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list,\
                                                                iter=iter,\
                                                                max_iter_time=self.iter_num, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_entity_list[-1].x_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_CGA_EIS():
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
            goa = CGA_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'cga_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('CGA left: {0}'.format(900 - counter))
access_CGA_EIS()