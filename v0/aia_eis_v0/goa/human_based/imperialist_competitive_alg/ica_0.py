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

class ICA_0:
    """
    Refer:
        paper:
            Imperialist Competitive Algorithm: An algorithm for Optimization Inspired by Imperialistic Competition
        Webs:
            The author's website:
                http://atashpaz.me/
            Author's Matlab code website
                https://ww2.mathworks.cn/matlabcentral/fileexchange/22046-imperialist-competitive-algorithm-ica
            Java Code
                https://github.com/robinroche/jica
    Attention:
        The stop criterion of ICA is the number of Empire is one, So, the Entity_num should be HUGE, at least 20
    """
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = self.fitness_function(self.x_list)

            # Flag is used to show its identity (Imperialist or colony), 0 == False, 1 == True
            self.imperialist_flag = 0
            self.colony_flag = 0

        def set_imperialist_flag(self, set_flag, colony_index_list):
            if set_flag == True:
                self.imperialist_flag = 1
                self.colony_flag = 0
                # Record the index of its colonies
                self.colony_index_list = colony_index_list
                # The total power is a combination between imperialist and weighted colony, Step D
                self.total_power = 0.0

        def set_colony_flag(self, set_flag, imperialist_index=None):
            if set_flag == True:
                self.colony_flag = 1

                # Remove all ability of imperialist
                self.imperialist_flag = 0
                self.colony_index_list = None
                self.total_power = None

                self.imperialist_index = imperialist_index

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(limits_list, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

        self.empire_num = int(0.1 * self.entity_num)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        # ------------------------------ Initialize imperialists and their colony ------------------------------
        # Sort the self.entity_list by fitness in ascending order
        self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
        # Select the most powerful entity as Imperialist (Its number is default as 1/5 ~ 1/10)
        imperialist_list = self.entity_list[: self.empire_num]
        imperialist_global_index_list = list(range(self.empire_num))
        imperialist_fitness_list = [im.fitness for im in imperialist_list]
        imperialist_power_list = [abs(im_fit / sum(imperialist_fitness_list)) for im_fit in imperialist_fitness_list]

        # Assign colony number to each imperialist according to their power
        imperialist_colony_num_list = []
        colony_num = self.entity_num - len(imperialist_list)
        for index, power in enumerate(imperialist_power_list):
            # Be careful when assign the colony to the last imperialist
            if index == (len(imperialist_power_list) - 1):
                imperialist_colony_num_list.append(colony_num - sum(imperialist_colony_num_list))
            else:
                # The colony number might be 0, so use math.ceil()
                imperialist_colony_num_list.append(int(math.ceil(power * colony_num)))

        # The remaining entity is taken as the colony of some Empire
        # Empire Randomly choose colony
        imperialist_colony_index_list = []
        colony_index_global_set = set(range(len(imperialist_list), self.entity_num))
        left_colony_index_set = colony_index_global_set
        for imperialist_colony_num, imperialist in zip(imperialist_colony_num_list, imperialist_list):
            colony_index_list = random.sample(list(left_colony_index_set), imperialist_colony_num)
            left_colony_index_set = left_colony_index_set - set(colony_index_list)
            imperialist_colony_index_list.append(colony_index_list)
            imperialist.set_imperialist_flag(set_flag=True, colony_index_list=colony_index_list)

        # Count the number of imperialist, if the number is 1, stop the iteration
        while (len(imperialist_list) > 1) and (self.iter_num > 0):
            self.iter_num -= 1
            cur_best_entity = sorted(imperialist_list, key=lambda entity: entity.fitness, reverse=False)[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = cur_best_entity
            cur_best_entity_list.append(cur_best_entity)
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Iterate over each Empire, and move the colony towards the imperialist
            for imperialist in imperialist_list:
                min_colony_fitness = 1000000
                min_colony_fitness_index = None
                for colony_index in imperialist.colony_index_list:
                    colony = self.entity_list[colony_index]
                    tmp_x_list = []
                    for c_x, i_x in zip(colony.x_list, imperialist.x_list):
                        x = c_x + random.uniform(0, 2 * (i_x - c_x))
                        tmp_x_list.append(x)
                    colony.x_list = tmp_x_list
                    colony.update()
                    if colony.fitness < min_colony_fitness:
                        min_colony_fitness_index = colony_index
                        min_colony_fitness = colony.fitness

                # If there is a colony with better fitness than imperialist, exchange their position
                if min_colony_fitness < imperialist.fitness:
                    im_x_list = imperialist.x_list
                    c_x_list = self.entity_list[min_colony_fitness_index].x_list
                    # Exchange position
                    imperialist.x_list = c_x_list
                    self.entity_list[min_colony_fitness_index].x_list = im_x_list
                    # Update fitness
                    imperialist.update()
                    self.entity_list[min_colony_fitness_index].update()

                # Calculate the total power
                tmp_total_power = imperialist.fitness + 0.1 * sum([self.entity_list[c_index].fitness for c_index in imperialist.colony_index_list]) / len(imperialist.colony_index_list)
                imperialist.total_power = tmp_total_power

            #-------------------- Choose the empire to occupy the weakest colony --------------------
            # Calculate the normalized total power
            imperialist_total_power_list = [imperialist.total_power for imperialist in imperialist_list]

            # Choose the weakest Empire
            weakest_imperialist_index = imperialist_total_power_list.index(min(imperialist_total_power_list))
            weakest_imperialist = imperialist_list[weakest_imperialist_index]

            # Choose the weakest colony from the weakest Empire
            weakest_colony_global_index = None
            weakest_colony_in_empire_fitness = 10000000
            for c_global_index in weakest_imperialist.colony_index_list:
                if self.entity_list[c_global_index].fitness < weakest_colony_in_empire_fitness:
                    weakest_colony_global_index = c_global_index
            # Delete the colony index record from one imperialist
            weakest_imperialist.colony_index_list.remove(weakest_colony_global_index)

            imperialist_normalized_total_power_list = [imperialist_total_power - max(imperialist_total_power_list) for imperialist_total_power in imperialist_total_power_list]
            # ZeroDivisionError, this means the fitness is extremely small and the results are good enough. The code can stop
            try:
                imperialist_possession_p_list = [abs(n_t_c / sum(imperialist_normalized_total_power_list)) for n_t_c in imperialist_normalized_total_power_list]
            except ZeroDivisionError as e:
                print('At {} iteration\nCurrent global best entity is {}\nits fitness is {}'.format(self.iter_num, self.global_best_entity.x_list, self.global_best_entity.fitness))
                return cur_best_entity_list, global_best_entity_list
                # break
            decision_p_list = [p - random.random() for p in imperialist_possession_p_list]
            max_dec_p_index = decision_p_list.index(max(decision_p_list))

            # Choose the empire with the highest probability to contain the weakest colony
            # Add the colony index record to another imperialist
            imperialist_list[max_dec_p_index].colony_index_list.append(weakest_colony_global_index)

            # Testify the length of colony_index_list
            for imp, imp_g_index in zip(imperialist_list, imperialist_global_index_list):
                # if the list is empty
                if len(imp.colony_index_list) == 0:
                    # Remove it from imperialist list and index list
                    imperialist_list.remove(imp)
                    imperialist_global_index_list.remove(imp_g_index)
                    # Turn the imperialist into a colony
                    imp.set_colony_flag(set_flag=True)
                    # Add it to another imperialist
                    imperialist_list[0].colony_index_list.append(imp_g_index)
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 200
#     entity_num = 50
#     dim = 8
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     ica = ICA_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     cur_best_entity_list, global_best_entity_list = ica.search()
#     print('Best entity position:', ica.global_best_entity.x_list)
#     print('Fitness:', ica.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ICA_1:
    """
    Refer:
        paper:
            Imperialist Competitive Algorithm: An algorithm for Optimization Inspired by Imperialistic Competition
        Webs:
            The author's website:
                http://atashpaz.me/
            Author's Matlab code website
                https://ww2.mathworks.cn/matlabcentral/fileexchange/22046-imperialist-competitive-algorithm-ica
            Java Code
                https://github.com/robinroche/jica
    Attention:
        The stop criterion of ICA is the number of Empire is one, So, the Entity_num should be HUGE, at least 20
    """
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = self.fitness_function(self.x_list)

            # Flag is used to show its identity (Imperialist or colony), 0 == False, 1 == True
            self.imperialist_flag = 0
            self.colony_flag = 0

        def set_imperialist_flag(self, set_flag, colony_index_list):
            if set_flag == True:
                self.imperialist_flag = 1
                self.colony_flag = 0
                # Record the index of its colonies
                self.colony_index_list = colony_index_list
                # The total power is a combination between imperialist and weighted colony, Step D
                self.total_power = 0.0

        def set_colony_flag(self, set_flag, imperialist_index=None):
            if set_flag == True:
                self.colony_flag = 1

                # Remove all ability of imperialist
                self.imperialist_flag = 0
                self.colony_index_list = None
                self.total_power = None

                self.imperialist_index = imperialist_index

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(limits_list, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

        self.empire_num = int(0.1 * self.entity_num)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        # ------------------------------ Initialize imperialists and their colony ------------------------------
        # Sort the self.entity_list by fitness in ascending order
        self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
        # Select the most powerful entity as Imperialist (Its number is default as 1/5 ~ 1/10)
        imperialist_list = self.entity_list[: self.empire_num]
        imperialist_global_index_list = list(range(self.empire_num))
        imperialist_inv_fitness_list = [1.0 / im.fitness for im in imperialist_list]
        imperialist_power_list = [abs(im_inv_fit / sum(imperialist_inv_fitness_list)) for im_inv_fit in imperialist_inv_fitness_list]

        # Assign colony number to each imperialist according to their power
        imperialist_colony_num_list = []
        colony_num = self.entity_num - len(imperialist_list)
        for index, power in enumerate(imperialist_power_list):
            # Be careful when assign the colony to the last imperialist
            if index == (len(imperialist_power_list) - 1):
                imperialist_colony_num_list.append(colony_num - sum(imperialist_colony_num_list))
            else:
                # The colony number might be 0, so use math.ceil()
                imperialist_colony_num_list.append(int(math.ceil(power * colony_num)))

        # The remaining entity is taken as the colony of some Empire
        # Empire Randomly choose colony
        imperialist_colony_index_list = []
        colony_index_global_set = set(range(len(imperialist_list), self.entity_num))
        left_colony_index_set = colony_index_global_set
        for imperialist_colony_num, imperialist in zip(imperialist_colony_num_list, imperialist_list):
            colony_index_list = random.sample(list(left_colony_index_set), imperialist_colony_num)
            left_colony_index_set = left_colony_index_set - set(colony_index_list)
            imperialist_colony_index_list.append(colony_index_list)
            imperialist.set_imperialist_flag(set_flag=True, colony_index_list=colony_index_list)

        # Count the number of imperialist, if the number is 1, stop the iteration
        while (len(imperialist_list) > 1) and (self.iter_num > 0):
            self.iter_num -= 1
            cur_best_entity = sorted(imperialist_list, key=lambda entity: entity.fitness, reverse=False)[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Iterate over each Empire, and move the colony towards the imperialist
            for imperialist in imperialist_list:
                min_colony_fitness = 1000000
                min_colony_fitness_index = None
                for colony_index in imperialist.colony_index_list:
                    colony = self.entity_list[colony_index]
                    tmp_x_list = []
                    for c_x, i_x in zip(colony.x_list, imperialist.x_list):
                        x = c_x + random.uniform(0, 2 * (i_x - c_x))
                        tmp_x_list.append(x)
                    colony.x_list = tmp_x_list
                    colony.update()
                    if colony.fitness < min_colony_fitness:
                        min_colony_fitness_index = colony_index
                        min_colony_fitness = colony.fitness

                # If there is a colony with better fitness than imperialist, exchange their position
                if min_colony_fitness < imperialist.fitness:
                    im_x_list = imperialist.x_list
                    c_x_list = self.entity_list[min_colony_fitness_index].x_list
                    # Exchange position
                    imperialist.x_list = c_x_list
                    self.entity_list[min_colony_fitness_index].x_list = im_x_list
                    # Update fitness
                    imperialist.update()
                    self.entity_list[min_colony_fitness_index].update()

                # Calculate the total power
                tmp_total_power = imperialist.fitness + 0.1 * sum([self.entity_list[c_index].fitness\
                                                                   for c_index in imperialist.colony_index_list]) / len(imperialist.colony_index_list)
                imperialist.total_power = tmp_total_power

            #-------------------- Choose the empire to occupy the weakest colony --------------------
            # Calculate the normalized total power
            imperialist_total_power_list = [imperialist.total_power for imperialist in imperialist_list]

            # Choose the weakest Empire
            weakest_imperialist_index = imperialist_total_power_list.index(min(imperialist_total_power_list))
            weakest_imperialist = imperialist_list[weakest_imperialist_index]

            # Choose the weakest colony from the weakest Empire
            weakest_colony_global_index = None
            weakest_colony_in_empire_fitness = -10000000
            for c_global_index in weakest_imperialist.colony_index_list:
                if self.entity_list[c_global_index].fitness > weakest_colony_in_empire_fitness:
                    weakest_colony_global_index = c_global_index
            # Delete the colony index record from one imperialist
            weakest_imperialist.colony_index_list.remove(weakest_colony_global_index)

            imperialist_normalized_total_power_list = [imperialist_total_power - max(imperialist_total_power_list)\
                                                       for imperialist_total_power in imperialist_total_power_list]
            # ZeroDivisionError, this means the fitness is extremely small and the results are good enough.
            # The code can stop
            try:
                imperialist_possession_p_list = [abs(n_t_c / sum(imperialist_normalized_total_power_list))\
                                                 for n_t_c in imperialist_normalized_total_power_list]
            except ZeroDivisionError as e:
                print('At {} iteration\nCurrent global best entity is {}\nits fitness is {}'.format(self.iter_num,\
                                                                                                    self.global_best_entity.x_list,\
                                                                                                    self.global_best_entity.fitness))
                return cur_best_entity_list, global_best_entity_list
                # break
            decision_p_list = [p - random.random() for p in imperialist_possession_p_list]
            max_dec_p_index = decision_p_list.index(max(decision_p_list))

            # Choose the empire with the highest probability to contain the weakest colony
            # Add the colony index record to another imperialist
            imperialist_list[max_dec_p_index].colony_index_list.append(weakest_colony_global_index)

            # Testify the length of colony_index_list
            for imp, imp_g_index in zip(imperialist_list, imperialist_global_index_list):
                # if the list is empty
                if len(imp.colony_index_list) == 0:
                    # Remove it from imperialist list and index list
                    imperialist_list.remove(imp)
                    imperialist_global_index_list.remove(imp_g_index)
                    # Turn the imperialist into a colony
                    imp.set_colony_flag(set_flag=True)
                    # Add it to another imperialist
                    imperialist_list[0].colony_index_list.append(imp_g_index)
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 200
#     entity_num = 50
#     dim = 8
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     ica = ICA_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     cur_best_entity_list, global_best_entity_list = ica.search()
#     print('Best entity position:', ica.global_best_entity.x_list)
#     print('Fitness:', ica.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ICA_EIS:
    """
    Refer:
        paper:
            Imperialist Competitive Algorithm: An algorithm for Optimization Inspired by Imperialistic Competition
        Webs:
            The author's website:
                http://atashpaz.me/
            Author's Matlab code website
                https://ww2.mathworks.cn/matlabcentral/fileexchange/22046-imperialist-competitive-algorithm-ica
            Java Code
                https://github.com/robinroche/jica
    Attention:
        The stop criterion of ICA is the number of Empire is one, So, the Entity_num should be HUGE, at least 20
    """
    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

            # Flag is used to show its identity (Imperialist or colony), 0 == False, 1 == True
            self.imperialist_flag = 0
            self.colony_flag = 0

        def set_imperialist_flag(self, set_flag, colony_index_list):
            if set_flag == True:
                self.imperialist_flag = 1
                self.colony_flag = 0
                # Record the index of its colonies
                self.colony_index_list = colony_index_list
                # The total power is a combination between imperialist and weighted colony, Step D
                self.total_power = 0.0

        def set_colony_flag(self, set_flag, imperialist_index=None):
            if set_flag == True:
                self.colony_flag = 1

                # Remove all ability of imperialist
                self.imperialist_flag = 0
                self.colony_index_list = None
                self.total_power = None

                self.imperialist_index = imperialist_index

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

        self.empire_num = int(0.1 * self.entity_num)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        # ------------------------------ Initialize imperialists and their colony ------------------------------
        # Sort the self.entity_list by fitness in ascending order
        self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
        # Select the most powerful entity as Imperialist (Its number is default as 1/5 ~ 1/10)
        imperialist_list = self.entity_list[: self.empire_num]
        imperialist_global_index_list = list(range(self.empire_num))
        imperialist_inv_fitness_list = [1.0 / im.fitness for im in imperialist_list]
        imperialist_power_list = [abs(im_inv_fit / sum(imperialist_inv_fitness_list)) for im_inv_fit in imperialist_inv_fitness_list]

        # Assign colony number to each imperialist according to their power
        imperialist_colony_num_list = []
        colony_num = self.entity_num - len(imperialist_list)
        for index, power in enumerate(imperialist_power_list):
            # Be careful when assign the colony to the last imperialist
            if index == (len(imperialist_power_list) - 1):
                imperialist_colony_num_list.append(colony_num - sum(imperialist_colony_num_list))
            else:
                # The colony number might be 0, so use math.ceil()
                # imperialist_colony_num_list.append(int(math.ceil(power * colony_num)))
                """
                    一开始在此处担心math.floor(power * colony_num)可能会四舍五入向下取值取到0，但这种情况只会发生在最后一两个占比较小的情况
                    当entity_num = 70, imperialist_num = 7, 剩余63个当作殖民地
                        7个帝国的实力占比为：[0.3406801419679475, 0.3217737094889076, 0.14547412284601466, 0.1295313554746719, 0.03302637351142717, 0.019999081302042195, 0.009515215408988987]
                        相应分得的殖民地数量为：实力 * 63 = [21.462848943980692, 20.271743697801178, 9.164869739298924, 8.16047539490433, 2.0806615312199117, 1.2599421220286584, 0.5994585707663062]
                        如果用math.ceil,四舍五入向上取值：[22(0), 21(1), 10(2), 9(3), 3(4), 2(5), *(6)] = 67 + * > 63
                        故此，采用math.floor, 四舍五入向下取值：[21(0), 20(1), 9(2), 8(3), 2(4), 1(5), 0(6)] = 61 < 63
                """
                imperialist_colony_num_list.append(int(math.floor(power * colony_num)))

        # The remaining entity is taken as the colony of some Empire
        # Empire Randomly choose colony
        imperialist_colony_index_list = []
        colony_index_global_set = set(range(len(imperialist_list), self.entity_num))
        left_colony_index_set = colony_index_global_set
        for imperialist_colony_num, imperialist in zip(imperialist_colony_num_list, imperialist_list):
            try:
                # ValueError: Sample larger than population or is negative
                colony_index_list = random.sample(list(left_colony_index_set), imperialist_colony_num)
            except ValueError as e:
                colony_index_list = list(left_colony_index_set)
            left_colony_index_set = left_colony_index_set - set(colony_index_list)
            imperialist_colony_index_list.append(colony_index_list)
            imperialist.set_imperialist_flag(set_flag=True, colony_index_list=colony_index_list)

        continue_criterion = True
        iter = 0
        # Count the number of imperialist, if the number is 1, stop the iteration
        while (len(imperialist_list) > 1) and continue_criterion:
            # self.iter_num -= 1
            # 默认是reverse=False 升序
            cur_best_entity = sorted(imperialist_list, key=lambda entity: entity.fitness, reverse=False)[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Iterate over each Empire, and move the colony towards the imperialist
            for imperialist in imperialist_list:
                # min_colony_fitness = 1000000
                # min_colony_fitness_index = None
                max_colony_fitness = 0
                max_colony_fitness_index = None
                for colony_index in imperialist.colony_index_list:
                    colony = self.entity_list[colony_index]
                    tmp_x_list = []
                    for c_x, i_x in zip(colony.x_list, imperialist.x_list):
                        x = c_x + random.uniform(0, 2 * (i_x - c_x))
                        tmp_x_list.append(x)
                    colony.x_list = tmp_x_list
                    colony.update()
                    if colony.fitness > max_colony_fitness:
                        max_colony_fitness_index = colony_index
                        max_colony_fitness = colony.fitness

                # If there is a colony with [better fitness] = [bigger power] = [1/cost] than imperialist, exchange their position
                if max_colony_fitness > imperialist.fitness:
                    im_x_list = copy.deepcopy(imperialist.x_list)
                    try:
                        c_x_list = self.entity_list[max_colony_fitness_index].x_list
                    except TypeError as e:
                        print(e)
                    # Exchange position
                    imperialist.x_list = c_x_list
                    self.entity_list[max_colony_fitness_index].x_list = im_x_list
                    # Update fitness
                    imperialist.update()
                    self.entity_list[max_colony_fitness_index].update()

                # Calculate the total power
                # ZeroDivisionError: float division by zero
                try:
                    tmp_total_power = imperialist.fitness + 0.1 * sum([self.entity_list[c_index].fitness\
                                                                   for c_index in imperialist.colony_index_list]) / len(imperialist.colony_index_list)
                except ZeroDivisionError as e:
                    tmp_total_power = imperialist.fitness
                    print('We got a imperialist that has no colony', e)
                imperialist.total_power = tmp_total_power

            #-------------------- Choose the empire to occupy the weakest colony --------------------
            # Calculate the normalized total power
            imperialist_total_power_list = [imperialist.total_power for imperialist in imperialist_list]

            # Choose the weakest Empire
            weakest_imperialist_index = imperialist_total_power_list.index(min(imperialist_total_power_list))
            weakest_imperialist = imperialist_list[weakest_imperialist_index]

            # Choose the weakest colony from the weakest Empire
            weakest_colony_global_index = None
            weakest_colony_in_empire_fitness = -10000000
            for c_global_index in weakest_imperialist.colony_index_list:
                if self.entity_list[c_global_index].fitness > weakest_colony_in_empire_fitness:
                    weakest_colony_global_index = c_global_index
            # Delete the colony index record from one imperialist
            weakest_imperialist.colony_index_list.remove(weakest_colony_global_index)

            imperialist_normalized_total_power_list = [imperialist_total_power - max(imperialist_total_power_list)\
                                                       for imperialist_total_power in imperialist_total_power_list]
            # ZeroDivisionError, this means the fitness is extremely small and the results are good enough.
            # The code can stop
            try:
                imperialist_possession_p_list = [abs(n_t_c / sum(imperialist_normalized_total_power_list))\
                                                 for n_t_c in imperialist_normalized_total_power_list]
            except ZeroDivisionError as e:
                print('At {} iteration\nCurrent global best entity is {}\nits fitness is {}'.format(self.iter_num,\
                                                                                                    self.global_best_entity.x_list,\
                                                                                                    self.global_best_entity.fitness))
                # return cur_best_entity_list, global_best_entity_list
                x_lists_list = [global_best_entity_list[-2].x_list, global_best_entity_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                return cur_best_entity_list, global_best_entity_list, iter, chi_squared

            decision_p_list = [p - random.random() for p in imperialist_possession_p_list]
            max_dec_p_index = decision_p_list.index(max(decision_p_list))

            # Choose the empire with the highest probability to contain the weakest colony
            # Add the colony index record to another imperialist
            imperialist_list[max_dec_p_index].colony_index_list.append(weakest_colony_global_index)

            # Testify the length of colony_index_list
            for imp, imp_g_index in zip(imperialist_list, imperialist_global_index_list):
                # if the list is empty
                if len(imp.colony_index_list) == 0:
                    # Remove it from imperialist list and index list
                    imperialist_list.remove(imp)
                    imperialist_global_index_list.remove(imp_g_index)
                    # Turn the imperialist into a colony
                    imp.set_colony_flag(set_flag=True)
                    # Add it to another imperialist
                    imperialist_list[0].colony_index_list.append(imp_g_index)

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

class ICA_EIS_access:
    """
    Refer:
        paper:
            Imperialist Competitive Algorithm: An algorithm for Optimization Inspired by Imperialistic Competition
        Webs:
            The author's website:
                http://atashpaz.me/
            Author's Matlab code website
                https://ww2.mathworks.cn/matlabcentral/fileexchange/22046-imperialist-competitive-algorithm-ica
            Java Code
                https://github.com/robinroche/jica
    Attention:
        The stop criterion of ICA is the number of Empire is one, So, the Entity_num should be HUGE, at least 20
    """
    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

            # Flag is used to show its identity (Imperialist or colony), 0 == False, 1 == True
            self.imperialist_flag = 0
            self.colony_flag = 0

        def set_imperialist_flag(self, set_flag, colony_index_list):
            if set_flag == True:
                self.imperialist_flag = 1
                self.colony_flag = 0
                # Record the index of its colonies
                self.colony_index_list = colony_index_list
                # The total power is a combination between imperialist and weighted colony, Step D
                self.total_power = 0.0

        def set_colony_flag(self, set_flag, imperialist_index=None):
            if set_flag == True:
                self.colony_flag = 1

                # Remove all ability of imperialist
                self.imperialist_flag = 0
                self.colony_index_list = None
                self.total_power = None

                self.imperialist_index = imperialist_index

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

        self.empire_num = int(0.1 * self.entity_num)

    def search(self, res_fn, start_time):
        cur_best_entity_list = []
        global_best_entity_list = []

        # ------------------------------ Initialize imperialists and their colony ------------------------------
        # Sort the self.entity_list by fitness in ascending order
        self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
        # Select the most powerful entity as Imperialist (Its number is default as 1/5 ~ 1/10)
        imperialist_list = self.entity_list[: self.empire_num]
        imperialist_global_index_list = list(range(self.empire_num))
        imperialist_inv_fitness_list = [1.0 / im.fitness for im in imperialist_list]
        imperialist_power_list = [abs(im_inv_fit / sum(imperialist_inv_fitness_list)) for im_inv_fit in imperialist_inv_fitness_list]

        # Assign colony number to each imperialist according to their power
        imperialist_colony_num_list = []
        colony_num = self.entity_num - len(imperialist_list)
        for index, power in enumerate(imperialist_power_list):
            # Be careful when assign the colony to the last imperialist
            if index == (len(imperialist_power_list) - 1):
                left_colony_num = colony_num - sum(imperialist_colony_num_list)
                imperialist_colony_num_list.append(left_colony_num)
                # if left_colony_num > 0:
                #     imperialist_colony_num_list.append(left_colony_num)
                # else:
                    # Take one colony from the first imperialist and give it to the last imperialist
                    # imperialist_colony_num_list[0] -= 1
                    # imperialist_colony_num_list.append(1)
            else:
                # The colony number might be 0, so use math.ceil()
                # imperialist_colony_num_list.append(int(math.ceil(power * colony_num)))
                """
                    一开始在此处担心math.floor(power * colony_num)可能会四舍五入向下取值取到0，但这种情况只会发生在最后一两个占比较小的情况
                    当entity_num = 70, imperialist_num = 7, 剩余63个当作殖民地
                        7个帝国的实力占比为：[0.3406801419679475, 0.3217737094889076, 0.14547412284601466, 0.1295313554746719, 0.03302637351142717, 0.019999081302042195, 0.009515215408988987]
                        相应分得的殖民地数量为：实力 * 63 = [21.462848943980692, 20.271743697801178, 9.164869739298924, 8.16047539490433, 2.0806615312199117, 1.2599421220286584, 0.5994585707663062]
                        如果用math.ceil,四舍五入向上取值：[22(0), 21(1), 10(2), 9(3), 3(4), 2(5), *(6)] = 67 + * > 63
                        故此，采用math.floor, 四舍五入向下取值：[21(0), 20(1), 9(2), 8(3), 2(4), 1(5), 0(6)] = 61 < 63
                """
                imperialist_colony_num_list.append(int(math.floor(power * colony_num)))

        # The remaining entity is taken as the colony of some Empire
        # Empire Randomly choose colony
        imperialist_colony_index_list = []
        colony_index_global_set = set(range(len(imperialist_list), self.entity_num))
        left_colony_index_set = colony_index_global_set
        for imperialist_colony_num, imperialist in zip(imperialist_colony_num_list, imperialist_list):
            try:
                colony_index_list = random.sample(list(left_colony_index_set), imperialist_colony_num)
            except ValueError as e:
                print('The number of left colony: {0}\n Need to pick {1} colony from it'.format(len(left_colony_index_set), imperialist_colony_num))
                print(e)
                colony_index_list = list(left_colony_index_set)
                # sys.exit(1)
            left_colony_index_set = left_colony_index_set - set(colony_index_list)
            imperialist_colony_index_list.append(colony_index_list)
            imperialist.set_imperialist_flag(set_flag=True, colony_index_list=colony_index_list)

        continue_criterion = True
        iter = 0
        # Count the number of imperialist, if the number is 1, stop the iteration
        while (len(imperialist_list) > 1) and continue_criterion:
            # self.iter_num -= 1
            cur_best_entity = sorted(imperialist_list, key=lambda entity: entity.fitness, reverse=False)[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Iterate over each Empire, and move the colony towards the imperialist
            for imperialist in imperialist_list:
                min_colony_fitness = 1000000
                min_colony_fitness_index = None
                for colony_index in imperialist.colony_index_list:
                    colony = self.entity_list[colony_index]
                    tmp_x_list = []
                    for c_x, i_x in zip(colony.x_list, imperialist.x_list):
                        x = c_x + random.uniform(0, 2 * (i_x - c_x))
                        tmp_x_list.append(x)
                    colony.x_list = tmp_x_list
                    colony.update()
                    if colony.fitness < min_colony_fitness:
                        min_colony_fitness_index = colony_index
                        min_colony_fitness = colony.fitness

                # If there is a colony with better fitness than imperialist, exchange their position
                # 我怀疑这一段822-848写错了，现在是最差的colony和帝国交换位置，论文中是最好的colony和帝国交换位置
                if min_colony_fitness < imperialist.fitness:
                    im_x_list = imperialist.x_list
                    c_x_list = self.entity_list[min_colony_fitness_index].x_list
                    # Exchange position
                    imperialist.x_list = c_x_list
                    self.entity_list[min_colony_fitness_index].x_list = im_x_list
                    # Update fitness
                    imperialist.update()
                    self.entity_list[min_colony_fitness_index].update()

                # Calculate the total power
                try:
                    tmp_total_power = imperialist.fitness \
                                      + 0.1 * sum([self.entity_list[c_index].fitness for c_index in imperialist.colony_index_list]) / len(imperialist.colony_index_list)
                except ZeroDivisionError as e:
                    tmp_total_power = imperialist.fitness
                    print('We got a imperialist that has no colony', e)

                imperialist.total_power = tmp_total_power

            #-------------------- Choose the empire to occupy the weakest colony --------------------
            # Calculate the normalized total power
            imperialist_total_power_list = [imperialist.total_power for imperialist in imperialist_list]

            # Choose the weakest Empire
            weakest_imperialist_index = imperialist_total_power_list.index(min(imperialist_total_power_list))
            weakest_imperialist = imperialist_list[weakest_imperialist_index]

            # Choose the weakest colony from the weakest Empire
            weakest_colony_global_index = None
            weakest_colony_in_empire_fitness = -10000000
            for c_global_index in weakest_imperialist.colony_index_list:
                if self.entity_list[c_global_index].fitness > weakest_colony_in_empire_fitness:
                    weakest_colony_global_index = c_global_index
            # Delete the colony index record from one imperialist
            weakest_imperialist.colony_index_list.remove(weakest_colony_global_index)

            imperialist_normalized_total_power_list = [imperialist_total_power - max(imperialist_total_power_list)\
                                                       for imperialist_total_power in imperialist_total_power_list]
            # ZeroDivisionError, this means the fitness is extremely small and the results are good enough.
            # The code can stop
            try:
                imperialist_possession_p_list = [abs(n_t_c / sum(imperialist_normalized_total_power_list))\
                                                 for n_t_c in imperialist_normalized_total_power_list]
            except ZeroDivisionError as e:
                print('At {} iteration\nCurrent global best entity is {}\nits fitness is {}'.format(self.iter_num,\
                                                                                                    self.global_best_entity.x_list,\
                                                                                                    self.global_best_entity.fitness))
                return cur_best_entity_list, global_best_entity_list
                # break
            decision_p_list = [p - random.random() for p in imperialist_possession_p_list]
            max_dec_p_index = decision_p_list.index(max(decision_p_list))

            # Choose the empire with the highest probability to contain the weakest colony
            # Add the colony index record to another imperialist
            imperialist_list[max_dec_p_index].colony_index_list.append(weakest_colony_global_index)

            # Testify the length of colony_index_list
            for imp, imp_g_index in zip(imperialist_list, imperialist_global_index_list):
                # if the list is empty
                if len(imp.colony_index_list) == 0:
                    # Remove it from imperialist list and index list
                    imperialist_list.remove(imp)
                    imperialist_global_index_list.remove(imp_g_index)
                    # Turn the imperialist into a colony
                    imp.set_colony_flag(set_flag=True)
                    # Add it to another imperialist
                    imperialist_list[0].colony_index_list.append(imp_g_index)

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

def access_ICA_EIS(ecm_num, start_num):
    counter = 0
    # Iterate on 9 ECMs
    # for i in range(1, 10):
    # # for i in range(2, 10):
    #     print('ICA starts to fit ECM{0}'.format(i))
    #     ecm_sim_folder = '../../../datasets/goa_datasets/simulated'
    #     ecm_num = i
    #     ecm_num_str = get_ecm_num_str(ecm_num)
    #     file_path = os.path.join(ecm_sim_folder, 'ecm_' + ecm_num_str)
    #     sim_ecm = load_sim_ecm_para_config_dict(ecm_num, file_path)
    #     para_num = len(sim_ecm['para'])
    #
    #     # Iterate for 100 times
    #     for j in range(100):
    #         t_start = perf_counter()
    #         # ------------------------------  Change GOA name ------------------------------
    #         goa = ICA_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
    #         res_fn = 'ica_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
    #         # ------------------------------  Change GOA name ------------------------------
    #         goa.search(res_fn, start_time=t_start)
    #
    #         counter += 1
    #         print('ICA left: {0}'.format(900 - counter))

    print('ICA starts to fit ECM{0}'.format(ecm_num))
    ecm_sim_folder = '../../../datasets/goa_datasets/simulated'
    ecm_num_str = get_ecm_num_str(ecm_num)
    file_path = os.path.join(ecm_sim_folder, 'ecm_' + ecm_num_str)
    sim_ecm = load_sim_ecm_para_config_dict(ecm_num, file_path)
    para_num = len(sim_ecm['para'])

    # Iterate for 100 times
    for j in range(start_num, 100):
        t_start = perf_counter()
        # ------------------------------  Change GOA name ------------------------------
        goa = ICA_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
        res_fn = 'ica_ecm{0}_'.format(ecm_num) + get_Num_len(num=j, length=2) + '.txt'
        # ------------------------------  Change GOA name ------------------------------
        goa.search(res_fn, start_time=t_start)

        counter += 1
        print('ICA on ECM{0} left: {1}'.format(ecm_num, len(range(start_num, 100)) - counter))
# access_ICA_EIS(ecm_num=9, start_num=7)