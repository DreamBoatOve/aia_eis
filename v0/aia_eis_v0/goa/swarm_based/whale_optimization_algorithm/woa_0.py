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

class WOA:
    class Whale:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.position_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness_function = fitness_function
            self.fitness = fitness_function(self.position_list)
        def update_fitness(self):
            self.fitness = self.fitness_function(self.position_list)

    def __init__(self, iter_time, whale_num, limits_list, fitness_function):
        self.iter_time = iter_time
        self.whale_num = whale_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.whales_list = [self.Whale(self.limits_list, self.fitness_function) for i in range(self.whale_num)]

        self.global_best_whale = self.Whale(self.limits_list, self.fitness_function)
        self.global_best_whale.fitness = float('inf')

    def search_best_whale(self):
        # The positions of each whale is updated, but the fitness is not calculated yet,
        # Here to update the fitness of each whale
        for i in range(len(self.whales_list)):
            self.whales_list[i].update_fitness()
        # The target is to get the minimum of the function
        current_best_whale = sorted(self.whales_list, key = lambda w : w.fitness, reverse = False)[0]
        if current_best_whale.fitness < self.global_best_whale.fitness:
            self.global_best_whale = current_best_whale
        return current_best_whale

    def forage(self):
        current_best_whale_record_list = []
        for iter in range(self.iter_time):
            # a1 decreases linearly from 2 to 0 in Eq.(2.3)
            a1 = 2 - 2 * iter / self.iter_time
            A = a1 * random.uniform(0 , 1)
            C = random.uniform(0, 1)

            # a2 linearly dicreases from -1 to -2 to calculate t in Eq.(3.12), do not find the eq(3.12) in the paper...
            a2 = -1 - 1 * iter/ self.iter_time
            L = (a2 - 1) * random.uniform(0, 1) + 1

            current_best_whale = self.search_best_whale()
            current_best_whale_record_list.append((iter, current_best_whale.fitness))
            for whale_index in range(self.whale_num):
                p = random.uniform(0, 1)
                for pos_index in range(len(self.limits_list)):
                    if p < 0.5:
                        if abs(A) >= 1:
                            # Expand the search area
                            random_whale = self.whales_list[random.randint(0, len(self.whales_list) - 1)]
                            x_distance = abs(self.whales_list[whale_index].position_list[pos_index] -\
                                             random_whale.position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = random_whale.position_list[pos_index] -\
                                                                                     A * x_distance
                        else: # means abs(A) < 1
                            x_distance = abs(C * current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = current_best_whale.position_list[pos_index] - A * x_distance
                    elif p >= 0.5:
                        x_distance = abs(current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                        self.whales_list[whale_index].position_list[pos_index] = x_distance * pow(math.e, 1 * L) * math.cos(2 * math.pi * L)\
                                                                                 + current_best_whale.position_list[pos_index]

                    # After serval iteration, the value of position is over the variable range, it is necessary to keep positions in limitations
                    pos_i = self.whales_list[whale_index].position_list[pos_index]
                    limit_range = self.limits_list[pos_index][1] - self.limits_list[pos_index][0]
                    if pos_i < self.limits_list[pos_index][0]:
                        pos_err = self.limits_list[pos_index][0] - pos_i
                        pos_i = pos_err / limit_range + self.limits_list[pos_index][0]
                    elif pos_i > self.limits_list[pos_index][1]:
                        pos_err = pos_i - self.limits_list[pos_index][1]
                        pos_i = self.limits_list[pos_index][1] - (pos_err / limit_range)
                    self.whales_list[whale_index].position_list[pos_index] = pos_i
        return self.global_best_whale, current_best_whale_record_list

# if __name__ == '__main__':
#     iter_time = 500
#     whale_num = 60
#     dim = 30
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#     woa = WOA(iter_time, whale_num, limits_list= f1_limits_list, fitness_function= f1_fitness_function)
#     best_whale, current_best_whale_record_list = woa.forage()
#     print('Best position:', best_whale.position_list)
#     print('Found minimum of the f1 function', best_whale.fitness)
#
#     # 画出每次迭代鲸鱼的最好成绩
#     iter_list = [i for i in range(iter_time)]
#     fitness_list = [record[1] for record in current_best_whale_record_list]
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, fitness_list, label='Iteration {0}\nWhale number {1}\nDimension {2}'.format(iter_time, whale_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Fitness')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class WOA_1:
    """
    Refer:
        paper: The Whale Optimization Algorithm
            2.2.1. Encircling prey
            2.2.2. Bubble-net attacking method (exploitation phase)
                1 Shrinking encircling mechanism, 缩小包围圈
                2 Spiral updating position，螺旋围绕
                两种行进策略均不断靠近当前迭代的最优解，只是前进的方式不一样，详见图4 a-b
            2.2.3. Search for prey (exploration phase)
    """
    class Whale:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.position_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness_function = fitness_function
            self.fitness = fitness_function(self.position_list)

        def update_fitness(self):
            self.fitness = self.fitness_function(self.position_list)

    def __init__(self, iter_time, whale_num, limits_list, fitness_function):
        self.iter_time = iter_time
        self.whale_num = whale_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.whales_list = [self.Whale(self.limits_list, self.fitness_function) for i in range(self.whale_num)]

        self.global_best_whale = self.Whale(self.limits_list, self.fitness_function)
        self.global_best_whale.fitness = float('inf')

    def forage(self):
        current_best_whale_list = []
        global_best_whale_list = []
        for iter in range(self.iter_time):

            # ------------------------ search for the current and global best whale ------------------------
            # The positions of each whale is updated, but the fitness is not calculated yet,
            # Here to update the fitness of each whale
            for i in range(len(self.whales_list)):
                self.whales_list[i].update_fitness()
            # The target is to get the minimum of the function
            current_best_whale = sorted(self.whales_list, key=lambda w: w.fitness, reverse=False)[0]
            current_best_whale_list.append(copy.deepcopy(current_best_whale))

            # if current_best_whale.fitness < self.global_best_whale.fitness:
            #     self.global_best_whale = current_best_whale
            if current_best_whale.fitness < self.global_best_whale.fitness:
                self.global_best_whale = copy.deepcopy(current_best_whale)
            # else:
            #     print('Current best whale is not better the global one')
            # global_best_whale_list.append(copy.deepcopy(self.global_best_whale))
            global_best_whale_list.append(self.global_best_whale)
            # ------------------------ search for the current and global best whale ------------------------

            # Update a(or a1, eq 2.3), A(eq 2.3), C(eq 2.4), l, and p
            # a1 decreases linearly from 2 to 0 in Eq.(2.3)
            a1 = 2 - 2 * iter / self.iter_time

            # A = a1 * random.uniform(0 , 1)
            # C = random.uniform(0, 1)

            # a2 linearly dicreases from -1 to -2 to calculate t in Eq.(3.12), do not find the eq(3.12) in the paper...
            a2 = -1 - 1 * iter / self.iter_time
            L = (a2 - 1) * random.uniform(0, 1) + 1

            for whale_index in range(self.whale_num):
                A_vector_list = [(2 * random.uniform(0, 1) - 1) * a1 for i in range(len(self.limits_list))]
                C_vector_list = [2 * random.uniform(0, 1) for i in range(len(self.limits_list))]
                """
                random p
                    decides the mechanism a whale will take.
                    According to Eq 2.6, 
                        if p < 0.5, the whale will take Shrinking encircling mechanism (Eq 2.3);
                        elif p >= 0.5, the whale will take Spiral updating position (Eq 2.5)
                """
                p = random.uniform(0, 1)
                for pos_index in range(len(self.limits_list)):
                    A = A_vector_list[pos_index]
                    C = C_vector_list[pos_index]
                    # Shrinking encircling mechanism,
                    if p < 0.5:
                        # get closer to a random whale
                        if abs(A) >= 1:
                            # Expand the search area
                            random_whale = self.whales_list[random.randint(0, len(self.whales_list) - 1)]
                            x_distance = abs(C * random_whale.position_list[pos_index] -\
                                             self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = random_whale.position_list[pos_index] -\
                                                                                     A * x_distance
                        # get closer to the prey
                        else: # means abs(A) < 1
                            x_distance = abs(C * current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = current_best_whale.position_list[pos_index] - A * x_distance
                    # Spiral updating position,
                    elif p >= 0.5:
                        x_distance = abs(current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                        self.whales_list[whale_index].position_list[pos_index] = x_distance * pow(math.e, 1 * L) * math.cos(2 * math.pi * L)\
                                                                                 + current_best_whale.position_list[pos_index]

                    # After serval iteration, the value of position is over the variable range, it is necessary to keep positions in limitations
                    pos_i = self.whales_list[whale_index].position_list[pos_index]

                    """
                    The logic of updating x in a position_list is WRONG
                    if limit_range is very small and equals to 0.01 and pos_err = 1.5, the updated x is beyond range again
                    
                    limit_range = self.limits_list[pos_index][1] - self.limits_list[pos_index][0]
                    if pos_i < self.limits_list[pos_index][0]:
                        pos_err = self.limits_list[pos_index][0] - pos_i
                        pos_i = pos_err / limit_range + self.limits_list[pos_index][0]
                    elif pos_i > self.limits_list[pos_index][1]:
                        pos_err = pos_i - self.limits_list[pos_index][1]
                        pos_i = self.limits_list[pos_index][1] - (pos_err / limit_range)
                    """
                    limit = self.limits_list[pos_index]
                    if (pos_i < limit[0]) or (pos_i > limit[1]):
                        self.whales_list[whale_index].position_list[pos_index] = random.uniform(limit[0], limit[1])
        return current_best_whale_list, global_best_whale_list

# if __name__ == '__main__':
#     iter_time = 200
#     whale_num = 20
#     dim = 30
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#     woa = WOA_1(iter_time, whale_num, limits_list= f1_limits_list, fitness_function= f1_fitness_function)
#     current_best_whale_list, global_best_whale_list = woa.forage()
#     best_whale = global_best_whale_list[-1]
#     print('Best position:', best_whale.position_list)
#     print('Found minimum of the f1 function', best_whale.fitness)
#
#     # 画出每次迭代鲸鱼的最好成绩
#     iter_list = [i for i in range(iter_time)]
#     current_fitness_list = [whale.fitness for whale in current_best_whale_list]
#     global_fitness_list = [whale.fitness for whale in global_best_whale_list]
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, current_fitness_list, linestyle='-',label='Current: Iteration {0}\nWhale number {1}\nDimension {2}'.format(iter_time, whale_num, dim))
#     # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, global_fitness_list, linestyle='--', label='Global: Iteration {0}\nWhale number {1}\nDimension {2}'.format(iter_time, whale_num, dim))
#     # line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Fitness')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class WOA_EIS:
    class Whale:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness_function = fitness_function
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update_fitness(self):
            # the constrain of the position of a Whale is put at the end of WOA_EIS.forage function
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, whale_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.whale_num = whale_num
        self.fitness_function = fitness_function

        self.whales_list = [self.Whale(self.exp_data_dict, self.fitness_function) for i in range(self.whale_num)]

        self.global_best_whale = self.Whale(self.exp_data_dict, self.fitness_function)
        self.global_best_whale.fitness = float('inf')

    def search(self):
        current_best_whale_list = []
        global_best_whale_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # ------------------------ search for the current and global best whale ------------------------
            # The positions of each whale is updated, but the fitness is not calculated yet,
            # Here to update the fitness of each whale
            for i in range(len(self.whales_list)):
                self.whales_list[i].update_fitness()
            # The target is to get the minimum of the function
            current_best_whale = sorted(self.whales_list, key=lambda w: w.fitness, reverse=False)[0]
            current_best_whale_list.append(copy.deepcopy(current_best_whale))

            if current_best_whale.fitness < self.global_best_whale.fitness:
                self.global_best_whale = copy.deepcopy(current_best_whale)
            global_best_whale_list.append(self.global_best_whale)
            # ------------------------ search for the current and global best whale ------------------------

            # Update a(or a1, eq 2.3), A(eq 2.3), C(eq 2.4), l, and p
            # a1 decreases linearly from 2 to 0 in Eq.(2.3)
            a1 = 2 - 2 * iter / self.iter_time

            # A = a1 * random.uniform(0 , 1)
            # C = random.uniform(0, 1)

            # a2 linearly dicreases from -1 to -2 to calculate t in Eq.(3.12), do not find the eq(3.12) in the paper...
            a2 = -1 - 1 * iter / self.iter_time
            L = (a2 - 1) * random.uniform(0, 1) + 1

            for whale_index in range(self.whale_num):
                A_vector_list = [(2 * random.uniform(0, 1) - 1) * a1 for i in range(len(self.limits_list))]
                C_vector_list = [2 * random.uniform(0, 1) for i in range(len(self.limits_list))]
                """
                random p
                    decides the mechanism a whale will take.
                    According to Eq 2.6, 
                        if p < 0.5, the whale will take Shrinking encircling mechanism (Eq 2.3);
                        elif p >= 0.5, the whale will take Spiral updating position (Eq 2.5)
                """
                p = random.uniform(0, 1)
                for pos_index in range(len(self.limits_list)):
                    A = A_vector_list[pos_index]
                    C = C_vector_list[pos_index]
                    # Shrinking encircling mechanism,
                    if p < 0.5:
                        # get closer to a random whale
                        if abs(A) >= 1:
                            # Expand the search area
                            random_whale = self.whales_list[random.randint(0, len(self.whales_list) - 1)]
                            x_distance = abs(C * random_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = random_whale.position_list[pos_index] -  A * x_distance
                        # get closer to the prey
                        else:  # means abs(A) < 1
                            x_distance = abs(C * current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = current_best_whale.position_list[pos_index] - A * x_distance
                    # Spiral updating position,
                    elif p >= 0.5:
                        x_distance = abs(current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                        self.whales_list[whale_index].position_list[pos_index] = x_distance * pow(math.e, 1 * L) * math.cos(2 * math.pi * L) \
                                                                                 + current_best_whale.position_list[pos_index]

                    # After serval iteration, the value of position is over the variable range, it is necessary to keep positions in limitations
                    pos_i = self.whales_list[whale_index].position_list[pos_index]
                    limit = self.limits_list[pos_index]
                    if (pos_i < limit[0]) or (pos_i > limit[1]):
                        self.whales_list[whale_index].position_list[pos_index] = random.uniform(limit[0], limit[1])

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_whale_list[-2].position_list, global_best_whale_list[-1].position_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_time,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        # return self.global_best_whale, current_best_whale_record_list
        return current_best_whale_list, global_best_whale_list, iter, chi_squared

class WOA_EIS_access:
    class Whale:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness_function = fitness_function
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update_fitness(self):
            # the constrain of the position of a Whale is put at the end of WOA_EIS.forage function
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, whale_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.whale_num = whale_num
        self.fitness_function = fitness_function

        self.whales_list = [self.Whale(self.exp_data_dict, self.fitness_function) for i in range(self.whale_num)]

        self.global_best_whale = self.Whale(self.exp_data_dict, self.fitness_function)
        self.global_best_whale.fitness = float('inf')

    def search(self, res_fn, start_time):
        current_best_whale_list = []
        global_best_whale_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # ------------------------ search for the current and global best whale ------------------------
            # The positions of each whale is updated, but the fitness is not calculated yet,
            # Here to update the fitness of each whale
            for i in range(len(self.whales_list)):
                self.whales_list[i].update_fitness()
            # The target is to get the minimum of the function
            current_best_whale = sorted(self.whales_list, key=lambda w: w.fitness, reverse=False)[0]
            current_best_whale_list.append(copy.deepcopy(current_best_whale))

            if current_best_whale.fitness < self.global_best_whale.fitness:
                self.global_best_whale = copy.deepcopy(current_best_whale)
            global_best_whale_list.append(self.global_best_whale)
            # ------------------------ search for the current and global best whale ------------------------

            # Update a(or a1, eq 2.3), A(eq 2.3), C(eq 2.4), l, and p
            # a1 decreases linearly from 2 to 0 in Eq.(2.3)
            a1 = 2 - 2 * iter / self.iter_time

            # A = a1 * random.uniform(0 , 1)
            # C = random.uniform(0, 1)

            # a2 linearly dicreases from -1 to -2 to calculate t in Eq.(3.12), do not find the eq(3.12) in the paper...
            a2 = -1 - 1 * iter / self.iter_time
            L = (a2 - 1) * random.uniform(0, 1) + 1

            for whale_index in range(self.whale_num):
                A_vector_list = [(2 * random.uniform(0, 1) - 1) * a1 for i in range(len(self.limits_list))]
                C_vector_list = [2 * random.uniform(0, 1) for i in range(len(self.limits_list))]
                """
                random p
                    decides the mechanism a whale will take.
                    According to Eq 2.6, 
                        if p < 0.5, the whale will take Shrinking encircling mechanism (Eq 2.3);
                        elif p >= 0.5, the whale will take Spiral updating position (Eq 2.5)
                """
                p = random.uniform(0, 1)
                for pos_index in range(len(self.limits_list)):
                    A = A_vector_list[pos_index]
                    C = C_vector_list[pos_index]
                    # Shrinking encircling mechanism,
                    if p < 0.5:
                        # get closer to a random whale
                        if abs(A) >= 1:
                            # Expand the search area
                            random_whale = self.whales_list[random.randint(0, len(self.whales_list) - 1)]
                            x_distance = abs(C * random_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = random_whale.position_list[pos_index] -  A * x_distance
                        # get closer to the prey
                        else:  # means abs(A) < 1
                            x_distance = abs(C * current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                            self.whales_list[whale_index].position_list[pos_index] = current_best_whale.position_list[pos_index] - A * x_distance
                    # Spiral updating position,
                    elif p >= 0.5:
                        x_distance = abs(current_best_whale.position_list[pos_index] - self.whales_list[whale_index].position_list[pos_index])
                        self.whales_list[whale_index].position_list[pos_index] = x_distance * pow(math.e, 1 * L) * math.cos(2 * math.pi * L) \
                                                                                 + current_best_whale.position_list[pos_index]

                    # After serval iteration, the value of position is over the variable range, it is necessary to keep positions in limitations
                    pos_i = self.whales_list[whale_index].position_list[pos_index]
                    limit = self.limits_list[pos_index]
                    if (pos_i < limit[0]) or (pos_i > limit[1]):
                        self.whales_list[whale_index].position_list[pos_index] = random.uniform(limit[0], limit[1])

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_whale_list[-2].position_list, global_best_whale_list[-1].position_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_time,
                                                                data_dict=self.exp_data_dict)
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list,\
                                                                iter=iter, \
                                                                max_iter_time=self.iter_time, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_whale_list[-1].position_list]) \
                           + '],' + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)
                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_WOA_EIS():
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
            goa = WOA_EIS_access(exp_data_dict=sim_ecm, iter_time=10000, whale_num=10*para_num)
            res_fn = 'woa_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('WOA left: {0}'.format(900 - counter))
# access_WOA_EIS()