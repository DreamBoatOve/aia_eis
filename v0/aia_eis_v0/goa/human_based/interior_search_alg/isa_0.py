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

def element_rank(element):
    return element.fitness

class ISA_0:
    class Element:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self, tmp_limit_list, cur_best_x_list):
            # ISA stresses its own Boundary constraint handling method
            for i, x in enumerate(self.x_list):
                if x < tmp_limit_list[i][0]:
                    r4 = random.random()
                    self.x_list[i] = r4 * tmp_limit_list[i][0] + (1 - r4) * cur_best_x_list[i]
                elif x > tmp_limit_list[i][1]:
                    r5 = random.random()
                    self.x_list[i] = r5 * tmp_limit_list[i][1] + (1 - r5) * cur_best_x_list[i]
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, element_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.element_num = element_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.element_list = [self.Element(self.limits_list, self.fitness_function) for i in range(self.element_num)]
        self.global_best_element = self.Element(self.limits_list, self.fitness_function)

        # Unique and only parameter used in this algorithm
        self.alpha = 0.2

    def search(self):
        cur_best_element_list = []
        global_best_element_list = []

        for iter_index in range(self.iter_num):
            self.element_list.sort(key=element_rank, reverse=False)
            cur_best_element_list.append(copy.deepcopy(self.element_list[0]))
            if self.element_list[0].fitness < self.global_best_element.fitness:
                self.global_best_element = copy.deepcopy(self.element_list[0])
            global_best_element_list.append(self.global_best_element)

            # Update current limitation for composition group
            # The upper/lower bounds is the extreme value of x in each dimension
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_x_list = []
                for element in self.element_list:
                    tmp_x_list.append(element.x_list[i])
                tmp_limit_list.append([min(tmp_x_list), max(tmp_x_list)])

            # Randomly divide all the elements into composition/mirror group
            mirror_list = []
            composition_list = []
            for i in range(1, self.element_num):
                if random.random() < self.alpha:
                    mirror_list.append(self.element_list[i])
                else:
                    composition_list.append(self.element_list[i])

            # Update all the elements, update the composition and mirror group first, because the global best is used
            # Update the Mirror group
            for m_e in mirror_list:
                r3 = random.random()
                m_x_list = [r3 * x + (1 - r3) * x_best for x, x_best in zip(m_e.x_list, self.element_list[0].x_list)]
                tmp_m_x_list = [2 * m_x - x for m_x, x in zip(m_x_list, m_e.x_list)]
                tmp_m_fitness = self.fitness_function(tmp_m_x_list)
                if tmp_m_fitness < m_e.fitness:
                    m_e.x_list = tmp_m_x_list
                    m_e.update(tmp_limit_list, self.element_list[0].x_list)

            # Update the Composition group
            for c_e in composition_list:
                r2 = random.random()
                tmp_c_x_list = [tmp_limit[0] + (tmp_limit[1] - tmp_limit[0]) * r2 for tmp_limit in tmp_limit_list]
                tmp_c_fitness = self.fitness_function(tmp_c_x_list)
                if tmp_c_fitness < c_e.fitness:
                    c_e.x_list = tmp_c_x_list
                    c_e.update(tmp_limit_list, self.element_list[0].x_list)

            # Update the current best element
            r_list = [random.gauss(mu=0, sigma=1) for i in range(len(self.limits_list))]
            tmp_best_element_x_list = [x + r * 0.01 for x, r in zip(self.element_list[0].x_list, r_list)]
            tmp_best_element_fitness = self.fitness_function(tmp_best_element_x_list)
            if tmp_best_element_fitness < self.element_list[0].fitness:
                self.element_list[0].x_list = tmp_best_element_x_list
                self.element_list[0].update(tmp_limit_list, self.element_list[0].x_list)
        return cur_best_element_list, global_best_element_list

# if __name__ == '__main__':
#     iter_num = 1000
#     element_num = 10
#     dim = 8
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     isa = ISA_0(iter_num, element_num, f1_limits_list, f1_fitness_function)
#     cur_best_element_list, global_best_element_list = isa.search()
#     print('Best element position:', isa.global_best_element.x_list)
#     print('Fitness:', isa.global_best_element.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_element_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_element_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, element_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, element_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ISA_1:
    """
    Refer:
        Paper:
            paper0: Interior search algorithm (ISA): A novel approach for global optimization
            paper1: Engineering Optimization using Interior Search Algorithm
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            alpha
                default 0.2
    Attention:
    """
    class Element:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self, tmp_limit_list, cur_best_x_list):
            # ISA stresses its own Boundary constraint handling method
            for i, x in enumerate(self.x_list):
                if x < tmp_limit_list[i][0]:
                    r4 = random.random()
                    self.x_list[i] = r4 * tmp_limit_list[i][0] + (1 - r4) * cur_best_x_list[i]
                elif x > tmp_limit_list[i][1]:
                    r5 = random.random()
                    self.x_list[i] = r5 * tmp_limit_list[i][1] + (1 - r5) * cur_best_x_list[i]
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, element_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.element_num = element_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.element_list = [self.Element(self.limits_list, self.fitness_function) for i in range(self.element_num)]
        self.global_best_element = self.Element(self.limits_list, self.fitness_function)

        # Unique and only parameter used in this algorithm
        self.alpha = 0.2

    def search(self):
        cur_best_element_list = []
        global_best_element_list = []

        for iter_index in range(self.iter_num):
            self.element_list.sort(key=lambda element : element.fitness, reverse=False)
            if self.element_list[0].fitness < self.global_best_element.fitness:
                self.global_best_element = copy.deepcopy(self.element_list[0])
            cur_best_element_list.append(copy.deepcopy(self.element_list[0]))
            global_best_element_list.append(copy.deepcopy(self.global_best_element))

            """
            paper1: II. INTERIOR SEARCH ALGORITHM step 5:
                LBj and UBj are respectively lower and upper bounds of the elements in jth iteration and they are, 
                respectively, the minimum and maximum values of all elements in the (j-1)th iteration.
                
                Update current limitation for composition group
                The upper/lower bounds is the extreme value of x in each dimension
            """
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_x_list = []
                for element in self.element_list:
                    tmp_x_list.append(element.x_list[i])
                tmp_limit_list.append([min(tmp_x_list), max(tmp_x_list)])

            # Randomly divide all the elements into composition/mirror group
            mirror_list = []
            composition_list = []
            for i in range(1, self.element_num):
                if random.random() < self.alpha:
                    mirror_list.append(self.element_list[i])
                else:
                    composition_list.append(self.element_list[i])

            # ------------------------------- Update all the elements -------------------------------
            # ------------------- Update the Mirror group -------------------
            # update the composition and mirror group first, because the global best is used
            for m_e in mirror_list:
                r3 = random.random()
                m_x_list = [r3 * x + (1 - r3) * x_best for x, x_best in zip(m_e.x_list, self.element_list[0].x_list)]
                tmp_m_x_list = [2 * m_x - x for m_x, x in zip(m_x_list, m_e.x_list)]
                tmp_m_fitness = self.fitness_function(tmp_m_x_list)
                if tmp_m_fitness < m_e.fitness:
                    m_e.x_list = tmp_m_x_list
                    m_e.update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the Mirror group -------------------

            # ------------------- Update the Composition group -------------------
            for c_e in composition_list:
                r2 = random.random()
                tmp_c_x_list = [tmp_limit[0] + (tmp_limit[1] - tmp_limit[0]) * r2 for tmp_limit in tmp_limit_list]
                tmp_c_fitness = self.fitness_function(tmp_c_x_list)
                if tmp_c_fitness < c_e.fitness:
                    c_e.x_list = tmp_c_x_list
                    c_e.update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the Composition group -------------------

            # ------------------- Update the current best element -------------------
            r_list = [random.gauss(mu=0, sigma=1) for i in range(len(self.limits_list))]
            tmp_best_element_x_list = [x + r * 0.01 * (limit[1] - limit[0])\
                                       for x, r, limit in zip(self.element_list[0].x_list, r_list, self.limits_list)]
            tmp_best_element_fitness = self.fitness_function(tmp_best_element_x_list)
            if tmp_best_element_fitness < self.element_list[0].fitness:
                self.element_list[0].x_list = tmp_best_element_x_list
                self.element_list[0].update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the current best element -------------------
            # ------------------------------- Update all the elements -------------------------------
        return cur_best_element_list, global_best_element_list

# if __name__ == '__main__':
#     iter_num = 1000
#     element_num = 10
#     dim = 8
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     isa = ISA_1(iter_num, element_num, f1_limits_list, f1_fitness_function)
#     cur_best_element_list, global_best_element_list = isa.search()
#     print('Best element position:', isa.global_best_element.x_list)
#     print('Fitness:', isa.global_best_element.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_element_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_element_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, element_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, element_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ISA_EIS:
    """
    Refer:
        Paper:
            paper0: Interior search algorithm (ISA): A novel approach for global optimization
            paper1: Engineering Optimization using Interior Search Algorithm
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            alpha
                default 0.2
    Attention:
    """

    class Element:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self, tmp_limit_list, cur_best_x_list):
            """
            Boundary constraint handling method
                paper1:
                    II. INTERIOR SEARCH ALGORITHM
                        B. Boundary constraint handling
            """
            for i, x in enumerate(self.x_list):
                if x < tmp_limit_list[i][0]:
                    r4 = random.random()
                    self.x_list[i] = r4 * tmp_limit_list[i][0] + (1 - r4) * cur_best_x_list[i]
                elif x > tmp_limit_list[i][1]:
                    r5 = random.random()
                    self.x_list[i] = r5 * tmp_limit_list[i][1] + (1 - r5) * cur_best_x_list[i]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, element_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.element_num = element_num
        self.fitness_function = fitness_function

        self.element_list = [self.Element(self.exp_data_dict, self.fitness_function) for i in range(self.element_num)]
        self.global_best_element = self.Element(self.exp_data_dict, self.fitness_function)

        # Unique and only parameter used in this algorithm
        self.alpha = 0.2

    def search(self):
        cur_best_element_list = []
        global_best_element_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.element_list.sort(key=lambda element: element.fitness, reverse=False)
            if self.element_list[0].fitness < self.global_best_element.fitness:
                self.global_best_element = copy.deepcopy(self.element_list[0])
            cur_best_element_list.append(copy.deepcopy(self.element_list[0]))
            global_best_element_list.append(copy.deepcopy(self.global_best_element))

            """
            paper1: II. INTERIOR SEARCH ALGORITHM step 5:
                LBj and UBj are respectively lower and upper bounds of the elements in jth iteration and they are, 
                respectively, the minimum and maximum values of all elements in the (j-1)th iteration.

                Update current limitation for composition group
                The upper/lower bounds is the extreme value of x in each dimension
            """
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_x_list = []
                for element in self.element_list:
                    tmp_x_list.append(element.x_list[i])
                tmp_limit_list.append([min(tmp_x_list), max(tmp_x_list)])

            # Randomly divide all the elements into composition/mirror group
            mirror_list = []
            composition_list = []
            for i in range(1, self.element_num):
                if random.random() < self.alpha:
                    mirror_list.append(self.element_list[i])
                else:
                    composition_list.append(self.element_list[i])

            # ------------------------------- Update all the elements -------------------------------
            # ------------------- Update the Mirror group -------------------
            # update the composition and mirror group first, because the global best is used
            for m_e in mirror_list:
                r3 = random.random()
                m_x_list = [r3 * x + (1 - r3) * x_best for x, x_best in zip(m_e.x_list, self.element_list[0].x_list)]
                tmp_m_x_list = [2 * m_x - x for m_x, x in zip(m_x_list, m_e.x_list)]
                tmp_m_fitness = self.fitness_function(self.exp_data_dict, tmp_m_x_list)
                if tmp_m_fitness < m_e.fitness:
                    m_e.x_list = tmp_m_x_list
                    m_e.update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the Mirror group -------------------

            # ------------------- Update the Composition group -------------------
            for c_e in composition_list:
                r2 = random.random()
                tmp_c_x_list = [tmp_limit[0] + (tmp_limit[1] - tmp_limit[0]) * r2 for tmp_limit in tmp_limit_list]
                tmp_c_fitness = self.fitness_function(self.exp_data_dict, tmp_c_x_list)
                if tmp_c_fitness < c_e.fitness:
                    c_e.x_list = tmp_c_x_list
                    c_e.update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the Composition group -------------------

            # ------------------- Update the current best element -------------------
            r_list = [random.gauss(mu=0, sigma=1) for i in range(len(self.limits_list))]
            tmp_best_element_x_list = [x + r * 0.01 * (limit[1] - limit[0]) \
                                       for x, r, limit in zip(self.element_list[0].x_list, r_list, self.limits_list)]
            tmp_best_element_fitness = self.fitness_function(self.exp_data_dict, tmp_best_element_x_list)
            if tmp_best_element_fitness < self.element_list[0].fitness:
                self.element_list[0].x_list = tmp_best_element_x_list
                self.element_list[0].update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the current best element -------------------
            # ------------------------------- Update all the elements -------------------------------

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_element_list[-2].x_list, global_best_element_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return cur_best_element_list, global_best_element_list, iter, chi_squared

class ISA_EIS_access:
    """
    Refer:
        Paper:
            paper0: Interior search algorithm (ISA): A novel approach for global optimization
            paper1: Engineering Optimization using Interior Search Algorithm
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            alpha
                default 0.2
    Attention:
    """

    class Element:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self, tmp_limit_list, cur_best_x_list):
            """
            Boundary constraint handling method
                paper1:
                    II. INTERIOR SEARCH ALGORITHM
                        B. Boundary constraint handling
            """
            for i, x in enumerate(self.x_list):
                if x < tmp_limit_list[i][0]:
                    r4 = random.random()
                    self.x_list[i] = r4 * tmp_limit_list[i][0] + (1 - r4) * cur_best_x_list[i]
                elif x > tmp_limit_list[i][1]:
                    r5 = random.random()
                    self.x_list[i] = r5 * tmp_limit_list[i][1] + (1 - r5) * cur_best_x_list[i]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, element_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.element_num = element_num
        self.fitness_function = fitness_function

        self.element_list = [self.Element(self.exp_data_dict, self.fitness_function) for i in range(self.element_num)]
        self.global_best_element = self.Element(self.exp_data_dict, self.fitness_function)

        # Unique and only parameter used in this algorithm
        self.alpha = 0.2

    def search(self, res_fn, start_time):
        cur_best_element_list = []
        global_best_element_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.element_list.sort(key=lambda element: element.fitness, reverse=False)
            if self.element_list[0].fitness < self.global_best_element.fitness:
                self.global_best_element = copy.deepcopy(self.element_list[0])
            cur_best_element_list.append(copy.deepcopy(self.element_list[0]))
            global_best_element_list.append(copy.deepcopy(self.global_best_element))

            """
            paper1: II. INTERIOR SEARCH ALGORITHM step 5:
                LBj and UBj are respectively lower and upper bounds of the elements in jth iteration and they are, 
                respectively, the minimum and maximum values of all elements in the (j-1)th iteration.

                Update current limitation for composition group
                The upper/lower bounds is the extreme value of x in each dimension
            """
            tmp_limit_list = []
            for i in range(len(self.limits_list)):
                tmp_x_list = []
                for element in self.element_list:
                    tmp_x_list.append(element.x_list[i])
                tmp_limit_list.append([min(tmp_x_list), max(tmp_x_list)])

            # Randomly divide all the elements into composition/mirror group
            mirror_list = []
            composition_list = []
            for i in range(1, self.element_num):
                if random.random() < self.alpha:
                    mirror_list.append(self.element_list[i])
                else:
                    composition_list.append(self.element_list[i])

            # ------------------------------- Update all the elements -------------------------------
            # ------------------- Update the Mirror group -------------------
            # update the composition and mirror group first, because the global best is used
            for m_e in mirror_list:
                r3 = random.random()
                m_x_list = [r3 * x + (1 - r3) * x_best for x, x_best in zip(m_e.x_list, self.element_list[0].x_list)]
                tmp_m_x_list = [2 * m_x - x for m_x, x in zip(m_x_list, m_e.x_list)]
                tmp_m_fitness = self.fitness_function(self.exp_data_dict, tmp_m_x_list)
                if tmp_m_fitness < m_e.fitness:
                    m_e.x_list = tmp_m_x_list
                    m_e.update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the Mirror group -------------------

            # ------------------- Update the Composition group -------------------
            for c_e in composition_list:
                r2 = random.random()
                tmp_c_x_list = [tmp_limit[0] + (tmp_limit[1] - tmp_limit[0]) * r2 for tmp_limit in tmp_limit_list]
                tmp_c_fitness = self.fitness_function(self.exp_data_dict, tmp_c_x_list)
                if tmp_c_fitness < c_e.fitness:
                    c_e.x_list = tmp_c_x_list
                    c_e.update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the Composition group -------------------

            # ------------------- Update the current best element -------------------
            r_list = [random.gauss(mu=0, sigma=1) for i in range(len(self.limits_list))]
            tmp_best_element_x_list = [x + r * 0.01 * (limit[1] - limit[0]) \
                                       for x, r, limit in zip(self.element_list[0].x_list, r_list, self.limits_list)]
            tmp_best_element_fitness = self.fitness_function(self.exp_data_dict, tmp_best_element_x_list)
            if tmp_best_element_fitness < self.element_list[0].fitness:
                self.element_list[0].x_list = tmp_best_element_x_list
                self.element_list[0].update(tmp_limit_list, self.element_list[0].x_list)
            # ------------------- Update the current best element -------------------
            # ------------------------------- Update all the elements -------------------------------

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_element_list[-2].x_list, global_best_element_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter, \
                                                                max_iter_time=self.iter_num, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_element_list[-1].x_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_ISA_EIS():
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
            goa = ISA_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, element_num=10*para_num)
            res_fn = 'isa_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('ISA left: {0}'.format(900 - counter))
access_ISA_EIS()