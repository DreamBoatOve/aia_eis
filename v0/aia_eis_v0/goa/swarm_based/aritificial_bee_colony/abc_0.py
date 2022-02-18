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

class ABC:
    """
    refer:
        00- AN IDEA BASED ON HONEY BEE SWARM FOR NUMERICAL OPTIMIZATION (TECHNICAL REPORT-TR06, OCTOBER, 2005)
        01- On the performance of artificial bee colony (ABC) algorithm
    """
    class Bee:
        def __init__(self, tabu_num, limits_list, fitness_function):
            self.tabu_num = tabu_num
            self.tabu_time = 0
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            # Initialize position of the bee
            self.position_list = [random.uniform(limits[0], limits[1]) for limits in limits_list]
            self.fitness = fitness_function(self.position_list)

        def update(self):
            # 1- Check whether the new position in the limits/boundary. If exceed the boundary, use the boundary to replace the outliner
            for i in range(len(self.limits_list)):
                if self.position_list[i] < self.limits_list[i][0]:
                    self.position_list[i] = self.limits_list[i][0]
                elif self.position_list[i] > self.limits_list[i][1]:
                    self.position_list[i] = self.limits_list[i][1]
            self.fitness = self.fitness_function(self.position_list)

    def __init__(self, iter_time, bee_num, tabu_num, limits_list, fitness_function):
        """
        :param
            iter_time:
            bee_num:
                Bees are classified as EMPLOYED BEE (bee_num / 2) and ONLOOKER (bee_num / 2).
                bee_num are even number.
            tabu_num:
            limits_list:
            fitness_function:
        """
        self.iter_time = iter_time
        self.bee_num = bee_num
        self.tabu_num = tabu_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.bees_list = [self.Bee(tabu_num, limits_list, fitness_function) for i in range(int(bee_num / 2))]
        self.global_best_bee = self.Bee(tabu_num, limits_list, fitness_function)
        self.global_best_bee.fitness = float('inf')

    def calcu_profit(self):
        # Dealing with the minimum problem, the lesser the value, the better. But the selection chance should be bigger (reversed to the better value)
        profit_list = []
        for bee in self.bees_list:
            fitness = bee.fitness
            profit = 0.0
            if fitness > 0:
                profit = 1 / (fitness + 1)
            else:
                profit = 1 + abs(fitness)
            profit_list.append(profit)
        return profit_list

    def forage(self):
        current_best_bee_list = []
        current_global_best_bee_list = []
        for iter in range(self.iter_time):
            # The first half bees are set as EMPLOYED FORAGERS.
            for bee_i in range(int(self.bee_num / 2)):
                # For each bee, one dimension is randomly selected and modified by a randomly selected bee (EF)
                random_dim_index = random.randint(0, len(self.limits_list) - 1)
                random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)
                while (random_bee_index == bee_i):
                    random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)

                # Get the position and fitness of the current bee
                ef_current_position_list = copy.deepcopy(self.bees_list[bee_i].position_list)
                ef_current_fitness = self.bees_list[bee_i].fitness

                ef_modified_position_list = copy.deepcopy(ef_current_position_list)
                ef_modified_position_list[random_dim_index] = ef_modified_position_list[random_dim_index] + (ef_modified_position_list[random_dim_index] - self.bees_list[random_bee_index].position_list[random_dim_index]) * random.uniform(-1, 1)
                ef_modified_bee = self.Bee(self.tabu_num, self.limits_list, self.fitness_function)
                ef_modified_bee.position_list = ef_modified_position_list
                ef_modified_bee.update()

                # Compare the current bee and the modified one, choose the fitter one
                if ef_modified_bee.fitness < ef_current_fitness:
                    # Newly created modified_bee.tabu_time = 0, So, no need to change the tabu_time as 0
                    self.bees_list[bee_i] = ef_modified_bee
                else:
                    # If the solution i can not be improved, increase its trial counter
                    self.bees_list[bee_i].tabu_time += 1

            # Calculate the profitability of each EMPLOYED FORAGER
            profit_list = self.calcu_profit()
            profit_max = max(profit_list)
            # The selection propability of each profitability
            profit_pop_list = [0.9 * (pf / profit_max) + 0.1 for pf in profit_list]

            # EMPLOYED FORAGERs come back from food sources and dance in front of the ONLOOKERs (UNEMPLOYED FORAGER)
            ef_index = 0
            uef_index = 0
            # Iteration: each ONLOOKER will select an EMPLOYED FORAGER to follow
            while uef_index < (self.bee_num / 2):
                if (random.random() < profit_pop_list[ef_index]):
                    random_dim_index = random.randint(0, len(self.limits_list) - 1)
                    random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)
                    while (random_bee_index == ef_index):
                        random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)

                    # Get the position and fitness of the current bee
                    uef_current_position_list = copy.deepcopy(self.bees_list[ef_index].position_list)
                    uef_current_fitness = self.bees_list[ef_index].fitness

                    uef_modified_position_list = copy.deepcopy(uef_current_position_list)
                    uef_modified_position_list[random_dim_index] = uef_modified_position_list[random_dim_index] + (uef_modified_position_list[random_dim_index] - self.bees_list[random_bee_index].position_list[random_dim_index]) * random.uniform(-1, 1)
                    uef_modified_bee = self.Bee(self.tabu_num, self.limits_list, self.fitness_function)
                    uef_modified_bee.position_list = uef_modified_position_list
                    uef_modified_bee.update()

                    # Compare the current bee and the modified one, choose the fitter one
                    if uef_modified_bee.fitness < uef_current_fitness:
                        # Newly created modified_bee.tabu_time = 0, So, no need to change the tabu_time as 0
                        self.bees_list[ef_index] = uef_modified_bee
                    else:
                        # If the solution i can not be improved, increase its trial counter
                        self.bees_list[ef_index].tabu_time += 1
                    uef_index += 1

                ef_index += 1
                if ef_index > (self.bee_num / 2 - 1):
                    ef_index = 0

            current_best_bee = sorted(self.bees_list[: int(self.bee_num / 2)], key=lambda bee: bee.fitness, reverse=False)[0]
            current_best_bee_list.append(current_best_bee)
            if current_best_bee.fitness < self.global_best_bee.fitness:
                self.global_best_bee = copy.deepcopy(current_best_bee)
            current_global_best_bee_list.append(self.global_best_bee)

            # Check each EF whether they exceed their trial time
            for bee_index in range(int(self.bee_num / 2)):
                if self.bees_list[bee_index].tabu_time > self.tabu_num:
                    self.bees_list[bee_index].__init__(self.tabu_num, self.limits_list, self.fitness_function)
        return current_best_bee_list, current_global_best_bee_list

# if __name__ == '__main__':
#     iter_time = 500
#     bee_num = 10
#     dim = 10
#     tabu_num = 20
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     abc = ABC(iter_time, bee_num, tabu_num, limits_list=f1_limits_list, fitness_function=f1_fitness_function)
#     current_best_bee_list, current_global_best_bee_list = abc.forage()
#     global_best_bee = abc.global_best_bee
#     print('Best bee position:', global_best_bee.position_list)
#     print('Fitness:', global_best_bee.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_time)]
#     cur_fitness_list = [bee.fitness for bee in current_best_bee_list]
#     cur_global_fitness_list = [bee.fitness for bee in current_global_best_bee_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nBee number {1}\nDimension {2}\nTabu num {3}'.format(iter_time, bee_num, dim, tabu_num))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nBee number {1}\nDimension {2}\nTabu num {3}'.format(iter_time, bee_num, dim, tabu_num))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ABC_EIS:
    class Bee:
        def __init__(self, tabu_num, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.tabu_num = tabu_num
            self.tabu_time = 0
            self.fitness_function = fitness_function

            # Initialize position of the bee
            self.position_list = [random.uniform(limits[0], limits[1]) for limits in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update(self):
            # 1- Check whether the new position in the limits/boundary. If exceed the boundary, use the boundary to replace the outliner
            for i in range(len(self.limits_list)):
                if self.position_list[i] < self.limits_list[i][0]:
                    self.position_list[i] = self.limits_list[i][0]
                elif self.position_list[i] > self.limits_list[i][1]:
                    self.position_list[i] = self.limits_list[i][1]
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, bee_num, tabu_num, fitness_function=cal_EIS_WSE_fitness_1):
        """
        :param
            exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试/模拟频率              'f',    fre_list
                    测试/模拟所得复数阻抗       'z_sim' or 'z_raw',     z_simulated/raw_complex_list
            iter_time:
                人为设定的算法最大迭代次数
            bee_num:
                Bees are classified as EMPLOYED BEE (bee_num / 2) and ONLOOKER (bee_num / 2).
                bee_num are even number.
            tabu_num:
            limits_list:
            fitness_function:
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.bee_num = bee_num
        self.tabu_num = tabu_num

        self.fitness_function = fitness_function

        self.bees_list = [self.Bee(tabu_num, self.exp_data_dict, fitness_function) for i in range(int(bee_num / 2))]
        self.global_best_bee = self.Bee(tabu_num, self.exp_data_dict, fitness_function)
        self.global_best_bee.fitness = float('inf')

    def calcu_profit(self):
        # Dealing with the minimum problem, the lesser the value, the better. But the selection chance should be bigger (reversed to the better value)
        profit_list = []
        for bee in self.bees_list:
            fitness = bee.fitness
            profit = 0.0
            if fitness > 0:
                profit = 1 / (fitness + 1)
            else:
                profit = 1 + abs(fitness)
            profit_list.append(profit)
        return profit_list

    def forage(self):
        current_best_bee_list = []
        current_global_best_bee_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # The first half bees are set as EMPLOYED FORAGERS.
            for bee_i in range(int(self.bee_num / 2)):
                # For each bee, one dimension is randomly selected and modified by a randomly selected bee (EF)
                random_dim_index = random.randint(0, len(self.limits_list) - 1)
                random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)
                while (random_bee_index == bee_i):
                    random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)

                # Get the position and fitness of the current bee
                ef_current_position_list = copy.deepcopy(self.bees_list[bee_i].position_list)
                ef_current_fitness = self.bees_list[bee_i].fitness

                ef_modified_position_list = copy.deepcopy(ef_current_position_list)
                ef_modified_position_list[random_dim_index] = ef_modified_position_list[random_dim_index] + (ef_modified_position_list[random_dim_index] - self.bees_list[random_bee_index].position_list[random_dim_index]) * random.uniform(-1, 1)
                ef_modified_bee = self.Bee(self.tabu_num, self.exp_data_dict, self.fitness_function)
                ef_modified_bee.position_list = ef_modified_position_list
                ef_modified_bee.update()

                # Compare the current bee and the modified one, choose the fitter one
                if ef_modified_bee.fitness < ef_current_fitness:
                    # Newly created modified_bee.tabu_time = 0, So, no need to change the tabu_time as 0
                    self.bees_list[bee_i] = ef_modified_bee
                else:
                    # If the solution i can not be improved, increase its trial counter
                    self.bees_list[bee_i].tabu_time += 1

            # Calculate the profitability of each EMPLOYED FORAGER
            profit_list = self.calcu_profit()
            profit_max = max(profit_list)
            # The selection propability of each profitability
            profit_pop_list = [0.9 * (pf / profit_max) + 0.1 for pf in profit_list]

            # EMPLOYED FORAGERs come back from food sources and dance in front of the ONLOOKERs (UNEMPLOYED FORAGER)
            ef_index = 0
            uef_index = 0
            # Iteration: each ONLOOKER will select an EMPLOYED FORAGER to follow
            while uef_index < (self.bee_num / 2):
                if (random.random() < profit_pop_list[ef_index]):
                    random_dim_index = random.randint(0, len(self.limits_list) - 1)
                    random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)
                    while (random_bee_index == ef_index):
                        random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)

                    # Get the position and fitness of the current bee
                    uef_current_position_list = copy.deepcopy(self.bees_list[ef_index].position_list)
                    uef_current_fitness = self.bees_list[ef_index].fitness

                    uef_modified_position_list = copy.deepcopy(uef_current_position_list)
                    uef_modified_position_list[random_dim_index] = uef_modified_position_list[random_dim_index] + (uef_modified_position_list[random_dim_index] - self.bees_list[random_bee_index].position_list[random_dim_index]) * random.uniform(-1, 1)
                    uef_modified_bee = self.Bee(self.tabu_num, self.exp_data_dict, self.fitness_function)
                    uef_modified_bee.position_list = uef_modified_position_list
                    uef_modified_bee.update()

                    # Compare the current bee and the modified one, choose the fitter one
                    if uef_modified_bee.fitness < uef_current_fitness:
                        # Newly created modified_bee.tabu_time = 0, So, no need to change the tabu_time as 0
                        self.bees_list[ef_index] = uef_modified_bee
                    else:
                        # If the solution i can not be improved, increase its trial counter
                        self.bees_list[ef_index].tabu_time += 1
                    uef_index += 1

                ef_index += 1
                if ef_index > (self.bee_num / 2 - 1):
                    ef_index = 0

            current_best_bee = sorted(self.bees_list[: int(self.bee_num / 2)], key=lambda bee: bee.fitness, reverse=False)[0]
            current_best_bee_list.append(current_best_bee)
            if current_best_bee.fitness < self.global_best_bee.fitness:
                self.global_best_bee = copy.deepcopy(current_best_bee)
            current_global_best_bee_list.append(self.global_best_bee)

            # Check each EF whether they exceed their trial time
            for bee_index in range(int(self.bee_num / 2)):
                if self.bees_list[bee_index].tabu_time > self.tabu_num:
                    self.bees_list[bee_index].__init__(self.tabu_num, self.exp_data_dict, self.fitness_function)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [copy.deepcopy(current_global_best_bee_list[-2].position_list),
                                copy.deepcopy(current_global_best_bee_list[-1].position_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_time,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return current_best_bee_list, current_global_best_bee_list, iter, chi_squared

class ABC_EIS_access:
    class Bee:
        def __init__(self, tabu_num, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.tabu_num = tabu_num
            self.tabu_time = 0
            self.fitness_function = fitness_function

            # Initialize position of the bee
            self.position_list = [random.uniform(limits[0], limits[1]) for limits in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update(self):
            # 1- Check whether the new position in the limits/boundary. If exceed the boundary, use the boundary to replace the outliner
            for i in range(len(self.limits_list)):
                if self.position_list[i] < self.limits_list[i][0]:
                    self.position_list[i] = self.limits_list[i][0]
                elif self.position_list[i] > self.limits_list[i][1]:
                    self.position_list[i] = self.limits_list[i][1]
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, bee_num, tabu_num, fitness_function=cal_EIS_WSE_fitness_1):
        """
        :param
            exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试/模拟频率              'f',    fre_list
                    测试/模拟所得复数阻抗       'z_sim' or 'z_raw',     z_simulated/raw_complex_list
            iter_time:
                人为设定的算法最大迭代次数
            bee_num:
                Bees are classified as EMPLOYED BEE (bee_num / 2) and ONLOOKER (bee_num / 2).
                bee_num are even number.
            tabu_num:
            limits_list:
            fitness_function:
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.bee_num = bee_num
        self.tabu_num = tabu_num

        self.fitness_function = fitness_function

        self.bees_list = [self.Bee(tabu_num, self.exp_data_dict, fitness_function) for i in range(int(bee_num / 2))]
        self.global_best_bee = self.Bee(tabu_num, self.exp_data_dict, fitness_function)
        self.global_best_bee.fitness = float('inf')

    def calcu_profit(self):
        # Dealing with the minimum problem, the lesser the value, the better. But the selection chance should be bigger (reversed to the better value)
        profit_list = []
        for bee in self.bees_list:
            fitness = bee.fitness
            profit = 0.0
            if fitness > 0:
                profit = 1 / (fitness + 1)
            else:
                profit = 1 + abs(fitness)
            profit_list.append(profit)
        return profit_list

    def search(self, res_fn, start_time):
        current_best_bee_list = []
        current_global_best_bee_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # The first half bees are set as EMPLOYED FORAGERS.
            for bee_i in range(int(self.bee_num / 2)):
                # For each bee, one dimension is randomly selected and modified by a randomly selected bee (EF)
                random_dim_index = random.randint(0, len(self.limits_list) - 1)
                random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)
                while (random_bee_index == bee_i):
                    random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)

                # Get the position and fitness of the current bee
                ef_current_position_list = copy.deepcopy(self.bees_list[bee_i].position_list)
                ef_current_fitness = self.bees_list[bee_i].fitness

                ef_modified_position_list = copy.deepcopy(ef_current_position_list)
                ef_modified_position_list[random_dim_index] = ef_modified_position_list[random_dim_index]\
                                                              + (ef_modified_position_list[random_dim_index] - self.bees_list[random_bee_index].position_list[random_dim_index]) * random.uniform(-1, 1)
                ef_modified_bee = self.Bee(self.tabu_num, self.exp_data_dict, self.fitness_function)
                ef_modified_bee.position_list = ef_modified_position_list
                ef_modified_bee.update()

                # Compare the current bee and the modified one, choose the fitter one
                if ef_modified_bee.fitness < ef_current_fitness:
                    # Newly created modified_bee.tabu_time = 0, So, no need to change the tabu_time as 0
                    self.bees_list[bee_i] = ef_modified_bee
                else:
                    # If the solution i can not be improved, increase its trial counter
                    self.bees_list[bee_i].tabu_time += 1

            # Calculate the profitability of each EMPLOYED FORAGER
            profit_list = self.calcu_profit()
            profit_max = max(profit_list)
            # The selection propability of each profitability
            profit_pop_list = [0.9 * (pf / profit_max) + 0.1 for pf in profit_list]

            # EMPLOYED FORAGERs come back from food sources and dance in front of the ONLOOKERs (UNEMPLOYED FORAGER)
            ef_index = 0
            uef_index = 0
            # Iteration: each ONLOOKER will select an EMPLOYED FORAGER to follow
            while uef_index < (self.bee_num / 2):
                if (random.random() < profit_pop_list[ef_index]):
                    random_dim_index = random.randint(0, len(self.limits_list) - 1)
                    random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)
                    while (random_bee_index == ef_index):
                        random_bee_index = random.randint(0, int(self.bee_num / 2) - 1)

                    # Get the position and fitness of the current bee
                    uef_current_position_list = copy.deepcopy(self.bees_list[ef_index].position_list)
                    uef_current_fitness = self.bees_list[ef_index].fitness

                    uef_modified_position_list = copy.deepcopy(uef_current_position_list)
                    uef_modified_position_list[random_dim_index] = uef_modified_position_list[random_dim_index] + (uef_modified_position_list[random_dim_index] - self.bees_list[random_bee_index].position_list[random_dim_index]) * random.uniform(-1, 1)
                    uef_modified_bee = self.Bee(self.tabu_num, self.exp_data_dict, self.fitness_function)
                    uef_modified_bee.position_list = uef_modified_position_list
                    uef_modified_bee.update()

                    # Compare the current bee and the modified one, choose the fitter one
                    if uef_modified_bee.fitness < uef_current_fitness:
                        # Newly created modified_bee.tabu_time = 0, So, no need to change the tabu_time as 0
                        self.bees_list[ef_index] = uef_modified_bee
                    else:
                        # If the solution i can not be improved, increase its trial counter
                        self.bees_list[ef_index].tabu_time += 1
                    uef_index += 1

                ef_index += 1
                if ef_index > (self.bee_num / 2 - 1):
                    ef_index = 0

            current_best_bee = sorted(self.bees_list[: int(self.bee_num / 2)], key=lambda bee: bee.fitness, reverse=False)[0]
            current_best_bee_list.append(current_best_bee)
            if current_best_bee.fitness < self.global_best_bee.fitness:
                self.global_best_bee = copy.deepcopy(current_best_bee)
            current_global_best_bee_list.append(self.global_best_bee)

            # Check each EF whether they exceed their trial time
            for bee_index in range(int(self.bee_num / 2)):
                if self.bees_list[bee_index].tabu_time > self.tabu_num:
                    self.bees_list[bee_index].__init__(self.tabu_num, self.exp_data_dict, self.fitness_function)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [copy.deepcopy(current_global_best_bee_list[-2].position_list),
                                copy.deepcopy(current_global_best_bee_list[-1].position_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list = x_lists_list,\
                                                                iter = iter,\
                                                                max_iter_time = self.iter_time,\
                                                                data_dict = self.exp_data_dict,\
                                                                CS_limit = 1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',['\
                           + ','.join([str(para) for para in current_global_best_bee_list[-1].position_list])\
                           + '],' + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_ABC_EIS():
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
            goa = ABC_EIS_access(exp_data_dict=sim_ecm, iter_time=10000, bee_num=10*para_num, tabu_num=3*para_num)
            res_fn = 'abc_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('ABC left: {0}'.format(900 - counter))
# access_ABC_EIS()