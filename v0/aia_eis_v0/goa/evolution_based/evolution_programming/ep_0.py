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

class EP_0:
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]

            # Set the value boundary for sigma
            self.sigma_abs_ceil = 0.5
            # random.gauss(mu, sigma)  # 随机生成符合高斯分布的随机数，mu,sigma为高斯分布的两个参数
            self.sigma_list = [random.uniform(0, self.sigma_abs_ceil) for i in range(len(limits_list))]

            self.fitness = fitness_function(self.x_list)
            self.q_score = 0

        def update(self):
            # Restrain the x in its boundary
            for i in range(len(self.limits_list)):
                if self.x_list[i] > self.limits_list[i][1]:
                    self.x_list[i] = self.limits_list[i][1]
                if self.x_list[i] < self.limits_list[i][0]:
                    self.x_list[i] = self.limits_list[i][0]
            # Restrain sigma in its boundary
            for s_index, sigma in enumerate(self.sigma_list):
                if abs(sigma) > self.sigma_abs_ceil:
                    # self.sigma_list[s_index] = (sigma / abs(sigma)) * self.sigma_abs_ceil
                    self.sigma_list[s_index] = random.uniform(0, self.sigma_abs_ceil)
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize entities
        self.entities_list = [self.Entity(limits_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity
        self.global_best_entity = self.Entity(limits_list, fitness_function)
        self.global_best_entity.fitness = float('inf')

        # Q-selection setting
        self.q_len = int(self.entity_num / 10)

    def evolve(self):
        current_best_entity_list = []
        global_best_entity_list = []
        for iter in range(self.iter_num):
            # Get the current&global best entity
            current_best_entity = sorted(self.entities_list, key=lambda entity: entity.fitness, reverse=False)[0]
            # print('Current best x:', current_best_entity.x_list)
            # print('Current best fitness:', current_best_entity.fitness)
            if self.global_best_entity.fitness > current_best_entity.fitness:
                self.global_best_entity = current_best_entity
            current_best_entity_list.append(current_best_entity)
            global_best_entity_list.append(self.global_best_entity)

            father_child_entities_list = copy.deepcopy(self.entities_list)
            for father_entity in self.entities_list:
                child_entity = self.Entity(self.limits_list, self.fitness_function)
                child_x_list = []
                child_sigma_list = []
                for f_x, f_sigma in zip(father_entity.x_list, father_entity.sigma_list):
                    # Mutation
                    c_x = f_x + f_sigma * random.gauss(mu=0, sigma=1)
                    c_sigma = f_sigma * (1 + 0.2 * random.gauss(mu=0, sigma=1))
                    child_x_list.append(c_x)
                    child_sigma_list.append(c_sigma)
                child_entity.x_list = copy.deepcopy(child_x_list)
                child_entity.sigma_list = copy.deepcopy(child_sigma_list)
                child_entity.update()
                father_child_entities_list.append(child_entity)
            # Select Q entities randomly
            q_entities_list = [copy.deepcopy(father_child_entities_list[i]) for i in random.sample(range(len(father_child_entities_list)), self.q_len)]

            # Score every entity
            for en_index in range(len(father_child_entities_list)):
                for q_entity in q_entities_list:
                    if father_child_entities_list[en_index].fitness < q_entity.fitness:
                        father_child_entities_list[en_index].q_score += 1

            # 1-Rank every entity by their score
            # 2-Select the first N entities
            # 3-Replace entities_list with the selected N entities
            # sorted(reverse=False)--Ascending order
            # sorted(reverse=True)--Descending order
            self.entities_list = sorted(father_child_entities_list, key=lambda entity: entity.q_score, reverse=True)[:len(self.entities_list)]
        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 50
#     entity_num = 20
#     dim = 2
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     ep = EP_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = ep.evolve()
#     print('Best entity position:', ep.global_best_entity.x_list)
#     print('Fitness:', ep.global_best_entity.fitness)
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

class EP_1:
    """
    Refer:
        Book:
            Book0: Computational Intelligence: An Introduction, Second Edition
                ch 11 Evolutionary Programming
                    11.1 Basic Evolutionary Programming
                    11.2 Evolutionary Programming Operators
                        11.2.1 Mutation Operators
                            non-adaptive EP
                            dynamic EP
                            *self-adaptive EP

                            *Noise probability distribution
                                Uniform
                                *Gaussian
                                Cauchy
                                Levy
                                Exponential
                                Chaos
                                Combined distributions
                        11.2.2 Selection Operators
                            Elitism
                            Tournament selection
                            Proportional selection
                            Nonlinear ranking selection
                    11.3 Strategy Parameters
                        11.3.1 Static Strategy Parameters
                        11.3.2 Dynamic Strategies
                        11.3.3 Self-Adaptation
                            *Additive
                            Multiplicative
                            Lognormal
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        In Evolution Programming, its offspring is first generated, and then strategy parameters (σ, sigma) are updated
        In Evolution Strategy, its strategy is updated first, and then offspring is generated
    Version:
        1
            Mutation:
            Selection:
    """
    class Entity:
        def __init__(self, limits_list, sigma_limit_list, fitness_function):
            self.limits_list = limits_list
            self.sigma_limit_list = sigma_limit_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            # random.gauss(mu, sigma)  # 随机生成符合高斯分布的随机数，mu,sigma为高斯分布的两个参数
            self.sigma_list = sigma_limit_list

            self.fitness = fitness_function(self.x_list)
            self.q_score = 0

        def update(self):
            # Restrain the x in its boundary
            for i in range(len(self.limits_list)):
                limit = self.limits_list[i]
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            # Restrain sigma in its boundary
            for s_index, sigma in enumerate(self.sigma_list):
                s_limit = self.sigma_limit_list[s_index]
                if (sigma <= 0.0) or (sigma > s_limit):
                    self.sigma_list[s_index] = random.uniform(0, s_limit)
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function
        # Initial Sigma limitation list
        self.I_sigma_limit_list = [(limit[1] - limit[0])/3 for limit in self.limits_list]

        # Initialize entities
        self.entities_list = [self.Entity(limits_list, self.I_sigma_limit_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity
        self.global_best_entity = self.Entity(limits_list, self.I_sigma_limit_list, fitness_function)

        # Q-selection setting
        self.q_len = int(self.entity_num / 5)

    def evolve(self):
        current_best_entity_list = []
        global_best_entity_list = []

        for iter in range(self.iter_num):
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            # Get the current & global best entity
            current_best_entity = self.entities_list[0]
            if self.global_best_entity.fitness > current_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            child_entity_list = []
            for father_entity in self.entities_list:
                child_entity = self.Entity(self.limits_list, self.I_sigma_limit_list, self.fitness_function)
                child_x_list = []
                child_sigma_list = []
                for i, f_x, f_sigma in zip(range(len(self.limits_list)), father_entity.x_list, father_entity.sigma_list):
                    # Mutation
                    c_x = f_x + random.gauss(mu=0, sigma=f_sigma)
                    # 仅靠下面一行sigma的更新会使得sigma始终上下飘忽，较大
                    c_sigma = f_sigma + math.sqrt(max(f_sigma, 0.001 * self.I_sigma_limit_list[i])) * random.uniform(0, 1)
                    # 人为逐渐缩小sigma，会使得算法快速收敛，且效果不错
                    c_sigma = c_sigma * (self.iter_num - iter) / self.iter_num
                    child_x_list.append(c_x)
                    child_sigma_list.append(c_sigma)
                child_entity.x_list = copy.deepcopy(child_x_list)
                child_entity.sigma_list = copy.deepcopy(child_sigma_list)
                child_entity.update()
                child_entity_list.append(child_entity)

            father_and_child_entity_list = self.entities_list + child_entity_list
            for en in father_and_child_entity_list:
                # Randomly select Q entities to be compared
                q_entity_index_list = random.sample(range(len(father_and_child_entity_list)), self.q_len)
                q_score = 0
                for q_i in q_entity_index_list:
                    if en.fitness < father_and_child_entity_list[q_i].fitness:
                        q_score += 1
                en.q_score = q_score

            if random.random() < 0.5:
                # ------------------ Use roulette wheel selection to select the next generation ------------------
                next_generation_index_set = set()
                fitness_list = [en.fitness for en in father_and_child_entity_list]
                inv_fitness_list = [max(fitness_list) - fit for fit in fitness_list]
                while len(next_generation_index_set) < self.entity_num:
                    r = random.uniform(0, sum(inv_fitness_list))
                    inv_fit_sum = 0.0
                    for i, inv_fit in enumerate(inv_fitness_list):
                        inv_fit_sum += inv_fit
                        if inv_fit_sum > r:
                            next_generation_index_set.add(i)
                            break
                self.entities_list = [copy.deepcopy(father_and_child_entity_list[n_i]) for n_i in list(next_generation_index_set)]
                # ------------------ Use roulette wheel selection to select the next generation ------------------
            else:
                # ------------------ Use Elitism to select the next generation ------------------
                self.entities_list = copy.deepcopy(sorted(father_and_child_entity_list, key=lambda en:en.fitness, reverse=False)[:self.entity_num])
                # ------------------ Use Elitism to select the next generation ------------------
        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 5
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     ep = EP_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = ep.evolve()
#     print('Best entity position:', ep.global_best_entity.x_list)
#     print('Fitness:', ep.global_best_entity.fitness)
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

class EP_EIS:
    """
    Refer:
        Book:
            Book0: Computational Intelligence: An Introduction, Second Edition
                ch 11 Evolutionary Programming
                    11.1 Basic Evolutionary Programming
                    11.2 Evolutionary Programming Operators
                        11.2.1 Mutation Operators
                            non-adaptive EP
                            dynamic EP
                            *self-adaptive EP

                            *Noise probability distribution
                                Uniform
                                *Gaussian
                                Cauchy
                                Levy
                                Exponential
                                Chaos
                                Combined distributions
                        11.2.2 Selection Operators
                            Elitism
                            Tournament selection
                            Proportional selection
                            Nonlinear ranking selection
                    11.3 Strategy Parameters
                        11.3.1 Static Strategy Parameters
                        11.3.2 Dynamic Strategies
                        11.3.3 Self-Adaptation
                            *Additive
                            Multiplicative
                            Lognormal
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        In Evolution Programming, its offspring is first generated, and then strategy parameters (σ, sigma) are updated
        In Evolution Strategy, its strategy is updated first, and then offspring is generated
    Version:
        1
            Mutation:
            Selection:
    """
    class Entity:
        def __init__(self, exp_data_dict, sigma_limit_list, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.sigma_limit_list = sigma_limit_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # random.gauss(mu, sigma)  # 随机生成符合高斯分布的随机数，mu,sigma为高斯分布的两个参数
            self.sigma_list = sigma_limit_list
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)
            self.q_score = 0

        def update(self):
            # Restrain the x in its boundary
            for i in range(len(self.limits_list)):
                limit = self.limits_list[i]
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            # Restrain sigma in its boundary
            for s_index, sigma in enumerate(self.sigma_list):
                s_limit = self.sigma_limit_list[s_index]
                if (sigma <= 0.0) or (sigma > s_limit):
                    self.sigma_list[s_index] = random.uniform(0, s_limit)
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num

        self.fitness_function = fitness_function
        # Initial Sigma limitation list
        self.I_sigma_limit_list = [(limit[1] - limit[0])/3 for limit in self.limits_list]

        # Initialize entities
        self.entities_list = [self.Entity(self.exp_data_dict, self.I_sigma_limit_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity
        self.global_best_entity = self.Entity(self.exp_data_dict, self.I_sigma_limit_list, fitness_function)

        # Q-selection setting
        self.q_len = int(self.entity_num / 5)

    def search(self):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            # Get the current & global best entity
            current_best_entity = self.entities_list[0]
            if self.global_best_entity.fitness > current_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            child_entity_list = []
            for father_entity in self.entities_list:
                child_entity = self.Entity(self.exp_data_dict, self.I_sigma_limit_list, self.fitness_function)
                child_x_list = []
                child_sigma_list = []
                for i, f_x, f_sigma in zip(range(len(self.limits_list)), father_entity.x_list, father_entity.sigma_list):
                    # Mutation
                    c_x = f_x + random.gauss(mu=0, sigma=f_sigma)
                    # 仅靠下面一行sigma的更新会使得sigma始终上下飘忽，较大
                    c_sigma = f_sigma + math.sqrt(max(f_sigma, 0.001 * self.I_sigma_limit_list[i])) * random.uniform(0, 1)
                    # 人为逐渐缩小sigma，会使得算法快速收敛，且效果不错
                    c_sigma = c_sigma * (self.iter_num - iter) / self.iter_num
                    child_x_list.append(c_x)
                    child_sigma_list.append(c_sigma)
                child_entity.x_list = copy.deepcopy(child_x_list)
                child_entity.sigma_list = copy.deepcopy(child_sigma_list)
                child_entity.update()
                child_entity_list.append(child_entity)

            father_and_child_entity_list = self.entities_list + child_entity_list
            for en in father_and_child_entity_list:
                # Randomly select Q entities to be compared
                q_entity_index_list = random.sample(range(len(father_and_child_entity_list)), self.q_len)
                q_score = 0
                for q_i in q_entity_index_list:
                    if en.fitness < father_and_child_entity_list[q_i].fitness:
                        q_score += 1
                en.q_score = q_score

            if random.random() < 0.5:
                # ------------------ Use roulette wheel selection to select the next generation ------------------
                next_generation_index_set = set()
                fitness_list = [en.fitness for en in father_and_child_entity_list]
                inv_fitness_list = [max(fitness_list) - fit for fit in fitness_list]
                while len(next_generation_index_set) < self.entity_num:
                    r = random.uniform(0, sum(inv_fitness_list))
                    inv_fit_sum = 0.0
                    for i, inv_fit in enumerate(inv_fitness_list):
                        inv_fit_sum += inv_fit
                        if inv_fit_sum > r:
                            next_generation_index_set.add(i)
                            break
                self.entities_list = [copy.deepcopy(father_and_child_entity_list[n_i]) for n_i in list(next_generation_index_set)]
                # ------------------ Use roulette wheel selection to select the next generation ------------------
            else:
                # ------------------ Use Elitism to select the next generation ------------------
                self.entities_list = copy.deepcopy(sorted(father_and_child_entity_list, key=lambda en:en.fitness, reverse=False)[:self.entity_num])
                # ------------------ Use Elitism to select the next generation ------------------

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

class EP_EIS_access:
    """
    Refer:
        Book:
            Book0: Computational Intelligence: An Introduction, Second Edition
                ch 11 Evolutionary Programming
                    11.1 Basic Evolutionary Programming
                    11.2 Evolutionary Programming Operators
                        11.2.1 Mutation Operators
                            non-adaptive EP
                            dynamic EP
                            *self-adaptive EP

                            *Noise probability distribution
                                Uniform
                                *Gaussian
                                Cauchy
                                Levy
                                Exponential
                                Chaos
                                Combined distributions
                        11.2.2 Selection Operators
                            Elitism
                            Tournament selection
                            Proportional selection
                            Nonlinear ranking selection
                    11.3 Strategy Parameters
                        11.3.1 Static Strategy Parameters
                        11.3.2 Dynamic Strategies
                        11.3.3 Self-Adaptation
                            *Additive
                            Multiplicative
                            Lognormal
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
        In Evolution Programming, its offspring is first generated, and then strategy parameters (σ, sigma) are updated
        In Evolution Strategy, its strategy is updated first, and then offspring is generated
    Version:
        1
            Mutation:
            Selection:
    """
    class Entity:
        def __init__(self, exp_data_dict, sigma_limit_list, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.sigma_limit_list = sigma_limit_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # random.gauss(mu, sigma)  # 随机生成符合高斯分布的随机数，mu,sigma为高斯分布的两个参数
            self.sigma_list = sigma_limit_list
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)
            self.q_score = 0

        def update(self):
            # Restrain the x in its boundary
            for i in range(len(self.limits_list)):
                limit = self.limits_list[i]
                if (self.x_list[i] < limit[0]) or (self.x_list[i] > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            # Restrain sigma in its boundary
            for s_index, sigma in enumerate(self.sigma_list):
                s_limit = self.sigma_limit_list[s_index]
                if (sigma <= 0.0) or (sigma > s_limit):
                    self.sigma_list[s_index] = random.uniform(0, s_limit)
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num

        self.fitness_function = fitness_function
        # Initial Sigma limitation list
        self.I_sigma_limit_list = [(limit[1] - limit[0])/3 for limit in self.limits_list]

        # Initialize entities
        self.entities_list = [self.Entity(self.exp_data_dict, self.I_sigma_limit_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity
        self.global_best_entity = self.Entity(self.exp_data_dict, self.I_sigma_limit_list, fitness_function)

        # Q-selection setting
        self.q_len = int(self.entity_num / 5)

    def search(self, res_fn, start_time):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entities_list.sort(key=lambda en:en.fitness, reverse=False)
            # Get the current & global best entity
            current_best_entity = self.entities_list[0]
            if self.global_best_entity.fitness > current_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            child_entity_list = []
            for father_entity in self.entities_list:
                child_entity = self.Entity(self.exp_data_dict, self.I_sigma_limit_list, self.fitness_function)
                child_x_list = []
                child_sigma_list = []
                for i, f_x, f_sigma in zip(range(len(self.limits_list)), father_entity.x_list, father_entity.sigma_list):
                    # Mutation
                    c_x = f_x + random.gauss(mu=0, sigma=f_sigma)
                    # 仅靠下面一行sigma的更新会使得sigma始终上下飘忽，较大
                    c_sigma = f_sigma + math.sqrt(max(f_sigma, 0.001 * self.I_sigma_limit_list[i])) * random.uniform(0, 1)
                    # 人为逐渐缩小sigma，会使得算法快速收敛，且效果不错
                    c_sigma = c_sigma * (self.iter_num - iter) / self.iter_num
                    child_x_list.append(c_x)
                    child_sigma_list.append(c_sigma)
                child_entity.x_list = copy.deepcopy(child_x_list)
                child_entity.sigma_list = copy.deepcopy(child_sigma_list)
                child_entity.update()
                child_entity_list.append(child_entity)

            father_and_child_entity_list = self.entities_list + child_entity_list
            for en in father_and_child_entity_list:
                # Randomly select Q entities to be compared
                q_entity_index_list = random.sample(range(len(father_and_child_entity_list)), self.q_len)
                q_score = 0
                for q_i in q_entity_index_list:
                    if en.fitness < father_and_child_entity_list[q_i].fitness:
                        q_score += 1
                en.q_score = q_score

            if random.random() < 0.5:
                # ------------------ Use roulette wheel selection to select the next generation ------------------
                next_generation_index_set = set()
                fitness_list = [en.fitness for en in father_and_child_entity_list]
                inv_fitness_list = [max(fitness_list) - fit for fit in fitness_list]
                while len(next_generation_index_set) < self.entity_num:
                    r = random.uniform(0, sum(inv_fitness_list))
                    inv_fit_sum = 0.0
                    for i, inv_fit in enumerate(inv_fitness_list):
                        inv_fit_sum += inv_fit
                        if inv_fit_sum > r:
                            next_generation_index_set.add(i)
                            break
                self.entities_list = [copy.deepcopy(father_and_child_entity_list[n_i]) for n_i in list(next_generation_index_set)]
                # ------------------ Use roulette wheel selection to select the next generation ------------------
            else:
                # ------------------ Use Elitism to select the next generation ------------------
                self.entities_list = copy.deepcopy(sorted(father_and_child_entity_list, key=lambda en:en.fitness, reverse=False)[:self.entity_num])
                # ------------------ Use Elitism to select the next generation ------------------

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

def access_EP_EIS():
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
            goa = EP_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'ep_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('EP left: {0}'.format(900 - counter))
# access_EP_EIS()