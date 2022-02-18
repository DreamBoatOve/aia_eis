import math
import random
import copy

from time import perf_counter
import os
import sys
sys.path.append('../../../')
from utils.file_utils.filename_utils import get_ecm_num_str, get_Num_len
from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict

from goa.GOA_criterions import goa_criterion_pack
from GA_pack.fittness_functions.eis_fitness import cal_EIS_WSE_fitness_1

def normalize(nums_list, type = 2):
    num_sum = sum([pow(num, type) for num in nums_list])
    return [num / math.sqrt(num_sum) for num in nums_list]

class MVO_1:
    """
    Refer:
        Paper0: Multi-Verse Optimizer: a nature-inspired algorithm for global optimization
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        Inflation rate describes the quality of an search agent, the bigger, the better
    """
    class Universe:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.uni_objs_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.infl_rate = fitness_function(self.uni_objs_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for obj_index, uni_obj in enumerate(self.uni_objs_list):
                if uni_obj > self.limits_list[obj_index][1]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][1]
                elif uni_obj < self.limits_list[obj_index][0]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][0]
            self.infl_rate = self.fitness_function(self.uni_objs_list)

    def __init__(self, iter_time, universe_num, limits_list, fitness_function):
        self.iter_time = iter_time
        self.universe_num = universe_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize the universes
        self.universes_list = [self.Universe(self.limits_list, self.fitness_function) for i in range(self.universe_num)]

        # Initialize the global best universe
        self.global_best_universe = self.Universe(limits_list = self.limits_list, fitness_function = fitness_function)
        self.global_best_universe.infl_rate = float('inf')

    def search_best_infl_rate(self):
        self.sorted_universes_list = sorted(self.universes_list, key=lambda uni: uni.infl_rate, reverse=False)
        current_best_universe = self.sorted_universes_list[0]
        if current_best_universe.infl_rate < self.global_best_universe.infl_rate:
            self.global_best_universe = copy.deepcopy(current_best_universe)
        self.normalized_uni_infl_rate = normalize([universe.infl_rate for universe in self.sorted_universes_list])

    def roulette_wheel_select_white_hole(self):
        # Inflation rate is reverse of fitness, One bigger, the ohter smaller
        sum_normalized_uni_infl_rate = sum(self.normalized_uni_infl_rate)
        # Because we want the minimium of the target function, so we use the sum to divide the normalized_uni_infl_rate to make it big and take more space in the roulette wheel
        wheel_sum = sum([sum_normalized_uni_infl_rate / infl_rate for infl_rate in self.normalized_uni_infl_rate])
        random_pointer = wheel_sum * random.random()
        current_pointer = 0.0
        for index, infl_rate in enumerate(self.normalized_uni_infl_rate):
            current_pointer += sum_normalized_uni_infl_rate / infl_rate
            if current_pointer > random_pointer:
                return index

    def inflate(self):
        current_best_universes_list = []
        for iter in range(self.iter_time):
            # WEP: wormhole exist probability, eq 3.3
            WEP = 0.2 + (iter + 1) * (1 - 0.2) / self.iter_time
            # TDR: Travelling distance rate, eq 3.4
            TDR = 1 - pow(iter + 1, 1/6) / pow(self.iter_time, 1/6)
            self.search_best_infl_rate()
            current_best_universes_list.append(self.global_best_universe)
            # print('At the {} iteration, the current best universe {}, its inflation rate {}'.format(iter, self.global_best_universe.uni_objs_list, self.global_best_universe.infl_rate))
            for uni_index in range(self.universe_num):
                for obj_index in range(len(self.limits_list)):
                    # Exploration：For each dimension, we explore first. Select a bad choice (white hole with high inflation rate) to sabotage the good ones for more wider search space
                    r1 = random.random()
                    # print('r1 = ',r1)
                    if r1 < self.normalized_uni_infl_rate[uni_index]:
                        white_hole_index = self.roulette_wheel_select_white_hole()
                        self.sorted_universes_list[uni_index].uni_objs_list[obj_index] = self.sorted_universes_list[white_hole_index].uni_objs_list[obj_index]
                    # Exploitation: make each universe similar to the best one. Along with the increase of WEP, its chance of moving towards the best universe is greater
                    r2 = random.random()
                    # print('r2 = ', r2)
                    if r2 < WEP:
                        r3 = random.random()
                        r4 = random.random()
                        # print('r3 =',r3, 'r4 =', r4)
                        boundary_min = self.limits_list[obj_index][0]
                        boundary_max = self.limits_list[obj_index][1]
                        if r3 < 0.5:
                            self.sorted_universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index] + TDR * (boundary_max - boundary_min) * r4
                        else:
                            self.sorted_universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index] - TDR * (boundary_max - boundary_min) * r4
            # After the current iteration, update the inflation rate
            for uni_index in range(self.universe_num):
                self.sorted_universes_list[uni_index].update()
        return current_best_universes_list

# if __name__ == '__main__':
#     iter_time = 1000
#     universe_num = 50
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     mvo = MVO_1(iter_time, universe_num, limits_list=f1_limits_list, fitness_function=f1_fitness_function)
#     current_best_universes_list = mvo.inflate()
#     global_best_univese = mvo.global_best_universe
#     print('Best position:', global_best_univese.uni_objs_list)
#     print('Found minimum of the f1 function', global_best_univese.infl_rate)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_time)]
#     infl_rate_list = [universe.infl_rate for universe in current_best_universes_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, infl_rate_list, label='Iteration {0}\nUniverse number {1}\nDimension {2}'.format(iter_time, universe_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Inflation rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

def normalize_1(nums_list):
    num_sum = sum(nums_list)
    inv_fitness_list = [num_sum / num for num in nums_list]
    inv_fitness_sum = sum(inv_fitness_list)
    normed_inv_fit_list = [inv_fit / inv_fitness_sum for inv_fit in inv_fitness_list]
    return normed_inv_fit_list

class MVO_2:
    """
    Refer:
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            WEP: wormhole exist probability, eq 3.3
            TDR: Travelling distance rate, eq 3.4
    Attention:
        Inflation rate describes the quality of an search agent, the bigger, the better
    """
    class Universe:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.uni_objs_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.infl_rate = fitness_function(self.uni_objs_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for obj_index, uni_obj in enumerate(self.uni_objs_list):
                if uni_obj > self.limits_list[obj_index][1]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][1]
                elif uni_obj < self.limits_list[obj_index][0]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][0]
            self.infl_rate = self.fitness_function(self.uni_objs_list)

    def __init__(self, iter_time, universe_num, limits_list, fitness_function):
        self.iter_time = iter_time
        self.universe_num = universe_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize the universes
        self.universes_list = [self.Universe(self.limits_list, self.fitness_function) for i in range(self.universe_num)]

        # Initialize the global best universe
        self.global_best_universe = self.Universe(limits_list = self.limits_list, fitness_function = fitness_function)
        self.global_best_universe.infl_rate = float('inf')

    def roulette_wheel_select_white_hole(self):
        random_pointer = random.random()
        current_pointer = 0.0
        for index, normalized_uni_infl in enumerate(self.normalized_uni_infl_rate):
            current_pointer += normalized_uni_infl
            if current_pointer > random_pointer:
                return index

    def search(self):
        current_best_universes_list = []
        global_best_universes_list = []

        for iter in range(self.iter_time):
            # WEP: wormhole exist probability, eq 3.3
            WEP = 0.2 + (iter + 1) * (1 - 0.2) / self.iter_time
            # TDR: Travelling distance rate, eq 3.4
            TDR = 1 - pow(iter + 1, 1/6) / pow(self.iter_time, 1/6)

            self.universes_list.sort(key=lambda uni: uni.infl_rate, reverse=False)
            current_best_universe = self.universes_list[0]
            if current_best_universe.infl_rate < self.global_best_universe.infl_rate:
                self.global_best_universe = copy.deepcopy(current_best_universe)
            current_best_universes_list.append(copy.deepcopy(current_best_universe))
            global_best_universes_list.append(copy.deepcopy(self.global_best_universe))

            self.normalized_uni_infl_rate = normalize_1([universe.infl_rate for universe in self.universes_list])
            for uni_index in range(self.universe_num):
                NI = self.normalized_uni_infl_rate[uni_index]
                for obj_index in range(len(self.limits_list)):
                    # Exploration：For each dimension, we explore first.
                    # Select a bad choice (white hole with high inflation rate) to sabotage the good ones for more wider search space
                    r1 = random.random()
                    if r1 < NI:
                        white_hole_index = self.roulette_wheel_select_white_hole()
                        self.universes_list[uni_index].uni_objs_list[obj_index] = self.universes_list[white_hole_index].uni_objs_list[obj_index]

                    # Exploitation: make each universe similar to the so far best one. Along with the increase of WEP,
                    # its chance of moving towards the best universe is greater
                    r2 = random.random()
                    if r2 < WEP:
                        r3 = random.random()
                        r4 = random.random()
                        boundary_min = self.limits_list[obj_index][0]
                        boundary_max = self.limits_list[obj_index][1]
                        if r3 < 0.5:
                            self.universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index]\
                                                                                      + TDR * ((boundary_max - boundary_min) * r4 + boundary_min)
                        else:
                            self.universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index]\
                                                                                      - TDR * ((boundary_max - boundary_min) * r4 + boundary_min)
            # After the current iteration, update the inflation rate
            for uni_index in range(self.universe_num):
                self.universes_list[uni_index].update()
        return current_best_universes_list, global_best_universes_list

# if __name__ == '__main__':
#     iter_time = 1000
#     universe_num = 50
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     mvo = MVO_2(iter_time, universe_num, limits_list=f1_limits_list, fitness_function=f1_fitness_function)
#     current_best_universes_list, global_best_universes_list = mvo.search()
#     global_best_univese = mvo.global_best_universe
#     print('Best position:', global_best_univese.uni_objs_list)
#     print('Found minimum of the f1 function', global_best_univese.infl_rate)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_time)]
#     cur_infl_rate_list = [universe.infl_rate for universe in current_best_universes_list]
#     global_infl_rate_list = [universe.infl_rate for universe in global_best_universes_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_infl_rate_list, label='Iteration {0}\nUniverse number {1}\nDimension {2}'.format(iter_time, universe_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, global_infl_rate_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_time, universe_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Inflation rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class MVO_EIS:
    """
    Refer:
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            WEP: wormhole exist probability, eq 3.3
            TDR: Travelling distance rate, eq 3.4
    Attention:
        Inflation rate describes the quality of an search agent, the bigger, the better
    """
    class Universe:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.fitness_function = fitness_function

            self.uni_objs_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.infl_rate = fitness_function(self.exp_data_dict, self.uni_objs_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for obj_index, uni_obj in enumerate(self.uni_objs_list):
                if uni_obj > self.limits_list[obj_index][1]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][1]
                elif uni_obj < self.limits_list[obj_index][0]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][0]
            self.infl_rate = self.fitness_function(self.exp_data_dict, self.uni_objs_list)

    def __init__(self, exp_data_dict, iter_time, universe_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.universe_num = universe_num
        self.fitness_function = fitness_function

        # Initialize the universes
        self.universes_list = [self.Universe(self.exp_data_dict, self.fitness_function) for i in range(self.universe_num)]

        # Initialize the global best universe
        self.global_best_universe = self.Universe(self.exp_data_dict, fitness_function)
        self.global_best_universe.infl_rate = float('inf')

    def roulette_wheel_select_white_hole(self):
        random_pointer = random.random()
        current_pointer = 0.0
        for index, normalized_uni_infl in enumerate(self.normalized_uni_infl_rate):
            current_pointer += normalized_uni_infl
            if current_pointer > random_pointer:
                return index

    def search(self):
        current_best_universes_list = []
        global_best_universes_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # WEP: wormhole exist probability, eq 3.3
            WEP = 0.2 + (iter + 1) * (1 - 0.2) / self.iter_time
            # TDR: Travelling distance rate, eq 3.4
            TDR = 1 - pow(iter + 1, 1/6) / pow(self.iter_time, 1/6)

            self.universes_list.sort(key=lambda uni: uni.infl_rate, reverse=False)
            current_best_universe = self.universes_list[0]
            if current_best_universe.infl_rate < self.global_best_universe.infl_rate:
                self.global_best_universe = copy.deepcopy(current_best_universe)
            current_best_universes_list.append(copy.deepcopy(current_best_universe))
            global_best_universes_list.append(copy.deepcopy(self.global_best_universe))

            self.normalized_uni_infl_rate = normalize_1([universe.infl_rate for universe in self.universes_list])
            for uni_index in range(self.universe_num):
                NI = self.normalized_uni_infl_rate[uni_index]
                for obj_index in range(len(self.limits_list)):
                    # Exploration：For each dimension, we explore first.
                    # Select a bad choice (white hole with high inflation rate) to sabotage the good ones for more wider search space
                    r1 = random.random()
                    if r1 < NI:
                        white_hole_index = self.roulette_wheel_select_white_hole()
                        self.universes_list[uni_index].uni_objs_list[obj_index] = self.universes_list[white_hole_index].uni_objs_list[obj_index]

                    # Exploitation: make each universe similar to the so far best one. Along with the increase of WEP,
                    # its chance of moving towards the best universe is greater
                    r2 = random.random()
                    if r2 < WEP:
                        r3 = random.random()
                        r4 = random.random()
                        boundary_min = self.limits_list[obj_index][0]
                        boundary_max = self.limits_list[obj_index][1]
                        if r3 < 0.5:
                            self.universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index]\
                                                                                      + TDR * ((boundary_max - boundary_min) * r4 + boundary_min)
                        else:
                            self.universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index]\
                                                                                      - TDR * ((boundary_max - boundary_min) * r4 + boundary_min)
            # After the current iteration, update the inflation rate
            for uni_index in range(self.universe_num):
                self.universes_list[uni_index].update()

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_universes_list[-2].uni_objs_list, global_best_universes_list[-1].uni_objs_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_time,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return current_best_universes_list, global_best_universes_list, iter, chi_squared

class MVO_EIS_access:
    """
    Refer:
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            WEP: wormhole exist probability, eq 3.3
            TDR: Travelling distance rate, eq 3.4
    Attention:
        Inflation rate describes the quality of an search agent, the bigger, the better
    """
    class Universe:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.fitness_function = fitness_function

            self.uni_objs_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.infl_rate = fitness_function(self.exp_data_dict, self.uni_objs_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for obj_index, uni_obj in enumerate(self.uni_objs_list):
                if uni_obj > self.limits_list[obj_index][1]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][1]
                elif uni_obj < self.limits_list[obj_index][0]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][0]
            self.infl_rate = self.fitness_function(self.exp_data_dict, self.uni_objs_list)

    def __init__(self, exp_data_dict, iter_time, universe_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.universe_num = universe_num
        self.fitness_function = fitness_function

        # Initialize the universes
        self.universes_list = [self.Universe(self.exp_data_dict, self.fitness_function) for i in range(self.universe_num)]

        # Initialize the global best universe
        self.global_best_universe = self.Universe(self.exp_data_dict, fitness_function)
        self.global_best_universe.infl_rate = float('inf')

    def roulette_wheel_select_white_hole(self):
        random_pointer = random.random()
        current_pointer = 0.0
        for index, normalized_uni_infl in enumerate(self.normalized_uni_infl_rate):
            current_pointer += normalized_uni_infl
            if current_pointer > random_pointer:
                return index

    def search(self, res_fn, start_time):
        current_best_universes_list = []
        global_best_universes_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # WEP: wormhole exist probability, eq 3.3
            WEP = 0.2 + (iter + 1) * (1 - 0.2) / self.iter_time
            # TDR: Travelling distance rate, eq 3.4
            TDR = 1 - pow(iter + 1, 1/6) / pow(self.iter_time, 1/6)

            self.universes_list.sort(key=lambda uni: uni.infl_rate, reverse=False)
            current_best_universe = self.universes_list[0]
            if current_best_universe.infl_rate < self.global_best_universe.infl_rate:
                self.global_best_universe = copy.deepcopy(current_best_universe)
            current_best_universes_list.append(copy.deepcopy(current_best_universe))
            global_best_universes_list.append(copy.deepcopy(self.global_best_universe))

            self.normalized_uni_infl_rate = normalize_1([universe.infl_rate for universe in self.universes_list])
            for uni_index in range(self.universe_num):
                NI = self.normalized_uni_infl_rate[uni_index]
                for obj_index in range(len(self.limits_list)):
                    # Exploration：For each dimension, we explore first.
                    # Select a bad choice (white hole with high inflation rate) to sabotage the good ones for more wider search space
                    r1 = random.random()
                    if r1 < NI:
                        white_hole_index = self.roulette_wheel_select_white_hole()
                        self.universes_list[uni_index].uni_objs_list[obj_index] = self.universes_list[white_hole_index].uni_objs_list[obj_index]

                    # Exploitation: make each universe similar to the so far best one. Along with the increase of WEP,
                    # its chance of moving towards the best universe is greater
                    r2 = random.random()
                    if r2 < WEP:
                        r3 = random.random()
                        r4 = random.random()
                        boundary_min = self.limits_list[obj_index][0]
                        boundary_max = self.limits_list[obj_index][1]
                        if r3 < 0.5:
                            self.universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index]\
                                                                                      + TDR * ((boundary_max - boundary_min) * r4 + boundary_min)
                        else:
                            self.universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index]\
                                                                                      - TDR * ((boundary_max - boundary_min) * r4 + boundary_min)
            # After the current iteration, update the inflation rate
            for uni_index in range(self.universe_num):
                self.universes_list[uni_index].update()

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_universes_list[-2].uni_objs_list, global_best_universes_list[-1].uni_objs_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter, \
                                                                max_iter_time=self.iter_time, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_universes_list[-1].uni_objs_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_MVO_EIS():
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
            goa = MVO_EIS_access(exp_data_dict=sim_ecm, iter_time=10000, universe_num=10*para_num)
            res_fn = 'mvo_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('MVO left: {0}'.format(900 - counter))
# access_MVO_EIS()