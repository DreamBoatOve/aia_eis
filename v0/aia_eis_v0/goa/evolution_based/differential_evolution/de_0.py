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

class DE_0:
    """
    Refer:
        Paper:
            paper0: Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces
        webs:
            差分进化算法（Differential Evolution)
                https://blog.csdn.net/qq_37423198/article/details/77856744
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        0
    """
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            for i in range(len(self.limits_list)):
                if self.x_list[i] > self.limits_list[i][1]:
                    self.x_list[i] = self.limits_list[i][1]
                if self.x_list[i] < self.limits_list[i][0]:
                    self.x_list[i] = self.limits_list[i][0]
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function, F=None, CR=0.9):
        self.iter_num = iter_num
        # paper0: a reasonable choice for entity_num is 5 * D ~ 10 * D (D: dimension), and has to > 4
        self.entity_num = entity_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function
        # paper0: F = 0.5
        self.F = F
        # paper0: CR = 0.1
        self.CR = CR

        self.entities_list = [self.Entity(limits_list, fitness_function) for i in range(entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

    def search(self):
        current_best_entity_list = []
        global_best_entity_list = []

        for iter_index in range(self.iter_num):
            current_best_entity = sorted(self.entities_list, key=lambda entity:entity.fitness, reverse=False)[0]
            current_best_entity_list.append(current_best_entity)
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
                global_best_entity_list.append(current_best_entity)
            else:
                global_best_entity_list.append(self.global_best_entity)
            if self.F == None:
                self.F = random.random() / 2 + 0.5
            for x_index in range(self.entity_num):
                x1_index = random.randint(0, self.entity_num - 1)
                while x1_index == x_index:
                    x1_index = random.randint(0, self.entity_num - 1)
                x2_index = random.randint(0, self.entity_num - 1)
                while (x2_index == x1_index) or (x2_index == x_index):
                    x2_index = random.randint(0, self.entity_num - 1)
                x3_index = random.randint(0, self.entity_num - 1)
                while (x3_index == x2_index) or (x3_index == x1_index) or (x3_index == x_index):
                    x3_index = random.randint(0, self.entity_num - 1)

                tmp_x_list = []
                counter = 0
                for a, b, c in zip(self.entities_list[x1_index].x_list, self.entities_list[x2_index].x_list, self.entities_list[x3_index].x_list):
                    # Mutation
                    t_x = a + self.F * (b - c)

                    # Crossover
                    if random.random() < self.CR:
                        tmp_x_list.append(t_x)
                    else:
                        tmp_x_list.append(self.entities_list[x_index].x_list[counter])
                    counter += 1

                tmp_entity = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity.x_list = tmp_x_list
                tmp_entity.update()

                # Selection (Greedy)
                if tmp_entity.fitness < self.entities_list[x_index].fitness:
                    self.entities_list[x_index] = copy.deepcopy(tmp_entity)

        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 5
#
#     f1_limits_list = [[-180, 70] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     de = DE_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = de.search()
#     print('Best entity position:', de.global_best_entity.x_list)
#     print('Fitness:', de.global_best_entity.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     # line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     # line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class DE_1:
    """
    Refer:
        Paper:
            paper0: Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces
        webs:
            差分进化算法（Differential Evolution)
                https://blog.csdn.net/qq_37423198/article/details/77856744
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        0
    """

    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            for i in range(len(self.limits_list)):
                if self.x_list[i] > self.limits_list[i][1]:
                    self.x_list[i] = self.limits_list[i][1]
                if self.x_list[i] < self.limits_list[i][0]:
                    self.x_list[i] = self.limits_list[i][0]
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function, F=None, CR=0.9):
        self.iter_num = iter_num
        # paper0: a reasonable choice for entity_num is 5 * D ~ 10 * D (D: dimension), and has to > 4
        self.entity_num = entity_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function
        # paper0: F = 0.5
        self.F = F
        # paper0: CR = 0.1
        self.CR = CR

        self.entities_list = [self.Entity(limits_list, fitness_function) for i in range(entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

    def search(self):
        current_best_entity_list = []
        global_best_entity_list = []

        for iter_index in range(self.iter_num):
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            current_best_entity = self.entities_list[0]
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            if self.F == None:
                self.F = random.random() / 2 + 0.5
            for x_index in range(self.entity_num):
                x1_index, x2_index, x3_index = random.sample(range(self.entity_num), 3)
                tmp_x_list = []
                for i, a, b, c in zip(range(len(self.limits_list)),
                                      self.entities_list[x1_index].x_list,
                                      self.entities_list[x2_index].x_list,
                                      self.entities_list[x3_index].x_list):
                    # Mutation
                    t_x = a + self.F * (b - c)

                    # Crossover
                    if random.random() < self.CR:
                        tmp_x_list.append(t_x)
                    else:
                        tmp_x_list.append(self.entities_list[x_index].x_list[i])

                tmp_entity = self.Entity(self.limits_list, self.fitness_function)
                tmp_entity.x_list = tmp_x_list
                tmp_entity.update()

                # Selection (Greedy)
                if tmp_entity.fitness < self.entities_list[x_index].fitness:
                    self.entities_list[x_index] = copy.deepcopy(tmp_entity)

        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 5
#
#     f1_limits_list = [[-180, 70] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#
#     de = DE_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = de.search()
#     print('Best entity position:', de.global_best_entity.x_list)
#     print('Fitness:', de.global_best_entity.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list,
#                      label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     # line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     # line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class DE_EIS:
    """
    Refer:
        Paper:
            paper0: Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces
        webs:
            差分进化算法（Differential Evolution)
                https://blog.csdn.net/qq_37423198/article/details/77856744
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        0
    """

    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i in range(len(self.limits_list)):
                if self.x_list[i] > self.limits_list[i][1]:
                    self.x_list[i] = self.limits_list[i][1]
                if self.x_list[i] < self.limits_list[i][0]:
                    self.x_list[i] = self.limits_list[i][0]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1, F=None, CR=0.9):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        # paper0: a reasonable choice for entity_num is 5 * D ~ 10 * D (D: dimension), and has to > 4
        self.entity_num = entity_num
        self.fitness_function = fitness_function
        # paper0: F = 0.5
        self.F = F
        # paper0: CR = 0.1
        self.CR = CR

        self.entities_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

    def search(self):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            current_best_entity = self.entities_list[0]
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            if self.F == None:
                self.F = random.random() / 2 + 0.5
            for x_index in range(self.entity_num):
                x1_index, x2_index, x3_index = random.sample(range(self.entity_num), 3)
                tmp_x_list = []
                for i, a, b, c in zip(range(len(self.limits_list)),
                                      self.entities_list[x1_index].x_list,
                                      self.entities_list[x2_index].x_list,
                                      self.entities_list[x3_index].x_list):
                    # Mutation
                    t_x = a + self.F * (b - c)

                    # Crossover
                    if random.random() < self.CR:
                        tmp_x_list.append(t_x)
                    else:
                        tmp_x_list.append(self.entities_list[x_index].x_list[i])

                tmp_entity = self.Entity(self.exp_data_dict, self.fitness_function)
                tmp_entity.x_list = tmp_x_list
                tmp_entity.update()

                # Selection (Greedy)
                if tmp_entity.fitness < self.entities_list[x_index].fitness:
                    self.entities_list[x_index] = copy.deepcopy(tmp_entity)
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
        return current_best_entity_list, global_best_entity_list, iter, chi_squared

class DE_EIS_access:
    """
    Refer:
        Paper:
            paper0: Differential Evolution – A Simple and Efficient Heuristic for Global Optimization over Continuous Spaces
        webs:
            差分进化算法（Differential Evolution)
                https://blog.csdn.net/qq_37423198/article/details/77856744
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        0
    """

    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i in range(len(self.limits_list)):
                if self.x_list[i] > self.limits_list[i][1]:
                    self.x_list[i] = self.limits_list[i][1]
                if self.x_list[i] < self.limits_list[i][0]:
                    self.x_list[i] = self.limits_list[i][0]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1, F=None, CR=0.9):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        # paper0: a reasonable choice for entity_num is 5 * D ~ 10 * D (D: dimension), and has to > 4
        self.entity_num = entity_num
        self.fitness_function = fitness_function
        # paper0: F = 0.5
        self.F = F
        # paper0: CR = 0.1
        self.CR = CR

        self.entities_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

    def search(self, res_fn, start_time):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            current_best_entity = self.entities_list[0]
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            if self.F == None:
                self.F = random.random() / 2 + 0.5
            for x_index in range(self.entity_num):
                x1_index, x2_index, x3_index = random.sample(range(self.entity_num), 3)
                tmp_x_list = []
                for i, a, b, c in zip(range(len(self.limits_list)),
                                      self.entities_list[x1_index].x_list,
                                      self.entities_list[x2_index].x_list,
                                      self.entities_list[x3_index].x_list):
                    # Mutation
                    t_x = a + self.F * (b - c)

                    # Crossover
                    if random.random() < self.CR:
                        tmp_x_list.append(t_x)
                    else:
                        tmp_x_list.append(self.entities_list[x_index].x_list[i])

                tmp_entity = self.Entity(self.exp_data_dict, self.fitness_function)
                tmp_entity.x_list = tmp_x_list
                tmp_entity.update()

                # Selection (Greedy)
                if tmp_entity.fitness < self.entities_list[x_index].fitness:
                    self.entities_list[x_index] = copy.deepcopy(tmp_entity)
            # -------------------------------------- Update global settings --------------------------------------

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_entity_list[-2].x_list, global_best_entity_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter, \
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

def access_DE_EIS():
    counter = 0
    # Iterate on 9 ECMs
    # for i in range(1, 10):
    for i in range(2, 10):
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
            goa = DE_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'de_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('DE left: {0}'.format(900 - counter))
# access_DE_EIS()