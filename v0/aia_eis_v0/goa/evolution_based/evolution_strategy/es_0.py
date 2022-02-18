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
(u+u, λ) ES
    U parent
    U offsprings
    select the U entity from U+U (parents and child)
"""
class ES_0:
    """
    Refer:
        Book:
            Computational intelligence An introduction
                part 3 EVOLUTIONARY COMPUTATION
                    ch12 Evolution Strategies
                        12.1 (1 + 1)-ES
                        12.2 Generic Evolution Strategy Algorithm
                        12.3 Strategy Parameters and Self-Adaptation
                            12.3.1 Strategy Parameter Types
        Paper:
        Web:
            什么是进化策略
                https://morvanzhou.github.io/tutorials/machine-learning/evolutionary-algorithm/3-01-evolution-strategy/
        Code
            Matlab
                Evolution strategies (es) in matlab
                    http://freesourcecode.net/matlabprojects/64999/evolution-strategies-(es)-in-matlab#.XYQ0UnakFbo
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

            self.sigma_ceil = 5
            self.sigma_list = [random.uniform(0, self.sigma_ceil) for i in range(len(limits_list))]

            self.fitness = fitness_function(self.x_list)

        def update(self):
            for index in range(len(self.limits_list)):
                if (self.x_list[index] < self.limits_list[index][0]) or (self.x_list[index] > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            for index in range(len(self.sigma_list)):
                if self.sigma_list[index] > self.sigma_ceil:
                    self.sigma_list[index] = random.uniform(0, self.sigma_ceil)
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize parents entities
        self.entities_list = [self.Entity(limits_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity and maxmize its fitness
        self.global_best_entity = self.Entity(limits_list, fitness_function)

    def evolve(self):
        current_best_entity_list = []
        global_best_entity_list = []
        for iter in range(self.iter_num):
            # Select the current/global best entity
            current_best_entity = sorted(self.entities_list, key = lambda entity: entity.fitness, reverse=False)[0]
            if self.global_best_entity.fitness > current_best_entity.fitness:
                self.global_best_entity = current_best_entity
            global_best_entity_list.append(self.global_best_entity)
            current_best_entity_list.append(current_best_entity)
            # Generate child
            child_entities_list = []
            for index in range(self.entity_num):
                # Select two parents randomly
                random_parent_indexes = random.sample(range(self.entity_num), 2)
                father = self.entities_list[random_parent_indexes[0]]
                mother = self.entities_list[random_parent_indexes[1]]

                # Crossover at multiple points according to random number
                child_x_list = []
                child_sigma_list = []
                for x_index in range(len(self.limits_list)):
                    r = random.random()
                    if r > 0.5:
                        # child_x_list.append(father.x_list[x_index])
                        # child_sigma_list.append(father.sigma_list[x_index])
                        x = father.x_list[x_index]
                        sigma = father.sigma_list[x_index]
                    else:
                        # child_x_list.append(mother.x_list[x_index])
                        # child_sigma_list.append(mother.sigma_list[x_index])
                        x = mother.x_list[x_index]
                        sigma = mother.sigma_list[x_index]
                    sigma = max(sigma + random.uniform(0, 1) - 0.5, 0.001)
                    x = x + math.sqrt(sigma) * random.gauss(mu=0, sigma=1)
                    child_x_list.append(x)
                    child_sigma_list.append(sigma)
                child_entity = self.Entity(self.limits_list, self.fitness_function)
                child_entity.x_list = child_x_list
                child_entity.sigma_list = child_sigma_list
                child_entity.update()
                child_entities_list.append(child_entity)

            # Rank parent + child and save the first U entity by their fitness
            all_entity = self.entities_list + child_entities_list
            self.entities_list = sorted(all_entity, key=lambda entity: entity.fitness, reverse=False)[:self.entity_num]
        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 4
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     es = ES_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = es.evolve()
#     print('Best entity position:', es.global_best_entity.x_list)
#     print('Fitness:', es.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ES_1:
    """
    Refer:
        Book:
            Computational intelligence An introduction
                part 3 EVOLUTIONARY COMPUTATION
                    12 Evolution Strategies
                        12.1 (1 + 1)-ES
                        12.2 Generic Evolution Strategy Algorithm
                        12.3 Strategy Parameters and Self-Adaptation
                            12.3.1 Strategy Parameter Types
        Paper:
        Web:
            什么是进化策略
                https://morvanzhou.github.io/tutorials/machine-learning/evolutionary-algorithm/3-01-evolution-strategy/
        Code
            Matlab
                Evolution strategies (es) in matlab
                    http://freesourcecode.net/matlabprojects/64999/evolution-strategies-(es)-in-matlab#.XYQ0UnakFbo
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
        def __init__(self, limits_list, sigma_limit_list, fitness_function):
            self.limits_list = limits_list
            self.sigma_limit_list = sigma_limit_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.sigma_list = [random.uniform(0.0, s_limit) for s_limit in self.sigma_limit_list]

            self.fitness = fitness_function(self.x_list)

        def update(self):
            for index in range(len(self.limits_list)):
                if (self.x_list[index] < self.limits_list[index][0]) or (self.x_list[index] > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            for index in range(len(self.sigma_list)):
                if self.sigma_list[index] > self.sigma_limit_list[index]:
                    self.sigma_list[index] = random.uniform(0, self.sigma_limit_list[index])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        # The scale of limitation of sigma in different dimensions
        # should be consistent with the limitation of each dimension
        self.sigma_limit_list = [(limit[1] - limit[0]) / 3 for limit in self.limits_list]
        self.fitness_function = fitness_function

        # Initialize parents entities
        self.entities_list = [self.Entity(limits_list, self.sigma_limit_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity and maximize its fitness
        self.global_best_entity = self.Entity(limits_list, self.sigma_limit_list, fitness_function)

    def evolve(self):
        current_best_entity_list = []
        global_best_entity_list = []

        for iter in range(self.iter_num):
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            # Select the current/global best entity
            current_best_entity = self.entities_list[0]
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))
            current_best_entity_list.append(copy.deepcopy(current_best_entity))

            # Generate child
            child_entities_list = []
            for index in range(self.entity_num):
                # Select two parents randomly
                random_parent_indexes = random.sample(range(self.entity_num), 2)
                father = self.entities_list[random_parent_indexes[0]]
                mother = self.entities_list[random_parent_indexes[1]]

                # Crossover at multiple points according to random number
                child_x_list = []
                child_sigma_list = []
                for x_index in range(len(self.limits_list)):
                    r = random.random()
                    if r > 0.5:
                        # child_x_list.append(father.x_list[x_index])
                        # child_sigma_list.append(father.sigma_list[x_index])
                        x = father.x_list[x_index]
                        sigma = father.sigma_list[x_index]
                    else:
                        # child_x_list.append(mother.x_list[x_index])
                        # child_sigma_list.append(mother.sigma_list[x_index])
                        x = mother.x_list[x_index]
                        sigma = mother.sigma_list[x_index]
                    sigma = max(sigma + random.uniform(0, 1) - 0.5, self.sigma_limit_list[x_index] * 0.01)
                    x = x + random.gauss(mu=0, sigma=sigma)
                    child_x_list.append(x)
                    child_sigma_list.append(sigma)
                child_entity = self.Entity(self.limits_list, self.sigma_limit_list, self.fitness_function)
                child_entity.x_list = child_x_list
                child_entity.sigma_list = child_sigma_list
                child_entity.update()
                child_entities_list.append(child_entity)

            # Rank parent + child and save the first U entity by their fitness
            all_entity = self.entities_list + child_entities_list
            self.entities_list = sorted(all_entity, key=lambda entity: entity.fitness, reverse=False)[:self.entity_num]
        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 4
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     es = ES_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = es.evolve()
#     print('Best entity position:', es.global_best_entity.x_list)
#     print('Fitness:', es.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class ES_EIS:
    """
    Refer:
        Book:
            Computational intelligence An introduction
                part 3 EVOLUTIONARY COMPUTATION
                    12 Evolution Strategies
                        12.1 (1 + 1)-ES
                        12.2 Generic Evolution Strategy Algorithm
                        12.3 Strategy Parameters and Self-Adaptation
                            12.3.1 Strategy Parameter Types
        Paper:
        Web:
            什么是进化策略
                https://morvanzhou.github.io/tutorials/machine-learning/evolutionary-algorithm/3-01-evolution-strategy/
        Code
            Matlab
                Evolution strategies (es) in matlab
                    http://freesourcecode.net/matlabprojects/64999/evolution-strategies-(es)-in-matlab#.XYQ0UnakFbo
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
        def __init__(self, exp_data_dict, sigma_limit_list, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.sigma_limit_list = sigma_limit_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.sigma_list = [random.uniform(0.0, s_limit) for s_limit in self.sigma_limit_list]

            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for index in range(len(self.limits_list)):
                if (self.x_list[index] < self.limits_list[index][0]) or (self.x_list[index] > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            for index in range(len(self.sigma_list)):
                if self.sigma_list[index] > self.sigma_limit_list[index]:
                    self.sigma_list[index] = random.uniform(0, self.sigma_limit_list[index])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num

        # The scale of limitation of sigma in different dimensions
        # should be consistent with the limitation of each dimension
        self.sigma_limit_list = [(limit[1] - limit[0]) / 3 for limit in self.limits_list]
        self.fitness_function = fitness_function

        # Initialize parents entities
        self.entities_list = [self.Entity(self.exp_data_dict, self.sigma_limit_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity and maximize its fitness
        self.global_best_entity = self.Entity(self.exp_data_dict, self.sigma_limit_list, fitness_function)

    def search(self):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            # Select the current/global best entity
            current_best_entity = self.entities_list[0]
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))
            current_best_entity_list.append(copy.deepcopy(current_best_entity))

            # Generate child
            child_entities_list = []
            for index in range(self.entity_num):
                # Select two parents randomly
                random_parent_indexes = random.sample(range(self.entity_num), 2)
                father = self.entities_list[random_parent_indexes[0]]
                mother = self.entities_list[random_parent_indexes[1]]

                # Crossover at multiple points according to random number
                child_x_list = []
                child_sigma_list = []
                for x_index in range(len(self.limits_list)):
                    r = random.random()
                    if r > 0.5:
                        # child_x_list.append(father.x_list[x_index])
                        # child_sigma_list.append(father.sigma_list[x_index])
                        x = father.x_list[x_index]
                        sigma = father.sigma_list[x_index]
                    else:
                        # child_x_list.append(mother.x_list[x_index])
                        # child_sigma_list.append(mother.sigma_list[x_index])
                        x = mother.x_list[x_index]
                        sigma = mother.sigma_list[x_index]
                    sigma = max(sigma + random.uniform(0, 1) - 0.5, self.sigma_limit_list[x_index] * 0.01)
                    x = x + random.gauss(mu=0, sigma=sigma)
                    child_x_list.append(x)
                    child_sigma_list.append(sigma)
                child_entity = self.Entity(self.exp_data_dict, self.sigma_limit_list, self.fitness_function)
                child_entity.x_list = child_x_list
                child_entity.sigma_list = child_sigma_list
                child_entity.update()
                child_entities_list.append(child_entity)

            # Rank parent + child and save the first U entity by their fitness
            all_entity = self.entities_list + child_entities_list
            self.entities_list = sorted(all_entity, key=lambda entity: entity.fitness, reverse=False)[:self.entity_num]

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

class ES_EIS_access:
    """
    Refer:
        Book:
            Computational intelligence An introduction
                part 3 EVOLUTIONARY COMPUTATION
                    12 Evolution Strategies
                        12.1 (1 + 1)-ES
                        12.2 Generic Evolution Strategy Algorithm
                        12.3 Strategy Parameters and Self-Adaptation
                            12.3.1 Strategy Parameter Types
        Paper:
        Web:
            什么是进化策略
                https://morvanzhou.github.io/tutorials/machine-learning/evolutionary-algorithm/3-01-evolution-strategy/
        Code
            Matlab
                Evolution strategies (es) in matlab
                    http://freesourcecode.net/matlabprojects/64999/evolution-strategies-(es)-in-matlab#.XYQ0UnakFbo
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
        def __init__(self, exp_data_dict, sigma_limit_list, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.sigma_limit_list = sigma_limit_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.sigma_list = [random.uniform(0.0, s_limit) for s_limit in self.sigma_limit_list]

            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for index in range(len(self.limits_list)):
                if (self.x_list[index] < self.limits_list[index][0]) or (self.x_list[index] > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            for index in range(len(self.sigma_list)):
                if self.sigma_list[index] > self.sigma_limit_list[index]:
                    self.sigma_list[index] = random.uniform(0, self.sigma_limit_list[index])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num

        # The scale of limitation of sigma in different dimensions
        # should be consistent with the limitation of each dimension
        self.sigma_limit_list = [(limit[1] - limit[0]) / 3 for limit in self.limits_list]
        self.fitness_function = fitness_function

        # Initialize parents entities
        self.entities_list = [self.Entity(self.exp_data_dict, self.sigma_limit_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity and maximize its fitness
        self.global_best_entity = self.Entity(self.exp_data_dict, self.sigma_limit_list, fitness_function)

    def search(self, res_fn, start_time):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            # Select the current/global best entity
            current_best_entity = self.entities_list[0]
            if current_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))
            current_best_entity_list.append(copy.deepcopy(current_best_entity))

            # Generate child
            child_entities_list = []
            for index in range(self.entity_num):
                # Select two parents randomly
                random_parent_indexes = random.sample(range(self.entity_num), 2)
                father = self.entities_list[random_parent_indexes[0]]
                mother = self.entities_list[random_parent_indexes[1]]

                # Crossover at multiple points according to random number
                child_x_list = []
                child_sigma_list = []
                for x_index in range(len(self.limits_list)):
                    r = random.random()
                    if r > 0.5:
                        # child_x_list.append(father.x_list[x_index])
                        # child_sigma_list.append(father.sigma_list[x_index])
                        x = father.x_list[x_index]
                        sigma = father.sigma_list[x_index]
                    else:
                        # child_x_list.append(mother.x_list[x_index])
                        # child_sigma_list.append(mother.sigma_list[x_index])
                        x = mother.x_list[x_index]
                        sigma = mother.sigma_list[x_index]
                    sigma = max(sigma + random.uniform(0, 1) - 0.5, self.sigma_limit_list[x_index] * 0.01)
                    x = x + random.gauss(mu=0, sigma=sigma)
                    child_x_list.append(x)
                    child_sigma_list.append(sigma)
                child_entity = self.Entity(self.exp_data_dict, self.sigma_limit_list, self.fitness_function)
                child_entity.x_list = child_x_list
                child_entity.sigma_list = child_sigma_list
                child_entity.update()
                child_entities_list.append(child_entity)

            # Rank parent + child and save the first U entity by their fitness
            all_entity = self.entities_list + child_entities_list
            self.entities_list = sorted(all_entity, key=lambda entity: entity.fitness, reverse=False)[:self.entity_num]

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

def access_ES_EIS():
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
            goa = ES_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'es_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('ES left: {0}'.format(900 - counter))
access_ES_EIS()