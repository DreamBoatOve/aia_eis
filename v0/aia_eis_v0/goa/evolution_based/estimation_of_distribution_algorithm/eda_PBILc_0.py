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

class EDA_PBILc_0:
    """
    Estimation of distribution algorithm 分布估计算法
        Extending population-based incremental learning to continuous search spaces
    Refer:
        Paper:
            paper0: 分布估计算法综述
                ch02 一个简单的分布估计算法
                    给了一个简单易懂的案例
                ch04 连续域的分布估计算法
            paper1: Estimation of Distribution Algorithms: A New Evolutionary Computation Approach for Graph Matching Problems
                ch02: Estimation Distribution Algorithms
                    2.3 EDAs in Continuous Domains
            paper2: **Extending population-based incremental learning to continuous search spaces
                ch03: Continuous PBIL
                    3.1 Continuous PBIL with dichotomic distributions
                    3.2 Continuous PBIL with Gausssian distributions
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Relaxation factor, alpha, the only adjustable parameter
    Attention:
    Version:
        0
    """
    class Entity:
        def __init__(self, limits_list, fitness_function, x_avg_list, x_sigma_list):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_avg_list = x_avg_list
            self.x_sigma_list = x_sigma_list

            self.x_list = [random.gauss(x_avg_list[i], x_sigma_list[i]) for i in range(len(limits_list))]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            # Check whether the value is in the boundary
            for i in range(len(self.limits_list)):
                # If not, give them a random value from the doable range
                if (self.x_list[i] < self.limits_list[i][0]) or (self.x_list[i] > self.limits_list[i][1]):
                    # Constrain the value in the boundary
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])

            # Update its fitness
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        # Load setting
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize average list,
        self.avg_list = [0.5 * (limit[1] + limit[0]) for limit in limits_list]

        # Initialize sigma_list
        # Default as the length of (0.25 ~ 0.5) * range
        self.sigma_list = [0.25 * (limit[1] - limit[0]) for limit in limits_list]

        # Initialize the global best entity
        self.global_best_entity = self.Entity(limits_list, fitness_function, self.avg_list, self.sigma_list)
        self.global_best_entity.fitness = float('inf')

    def evolve(self, alpha=0.5):
        current_best_entity_list = []
        global_best_entity_list = []
        for iter_index in range(self.iter_num):
            # Create new generation
            # Initialize a population
            self.entities_list = [self.Entity(self.limits_list, self.fitness_function, self.avg_list, self.sigma_list) for i in range(self.entity_num)]

            # Select the first two optimal and the worst entities according to their fitness
            sorted_entities_list = sorted(self.entities_list, key=lambda entity : entity.fitness, reverse=False)
            current_1_best_entity, current_2_best_entity = sorted_entities_list[:2]
            current_worst_entity = sorted_entities_list[-1]

            # Compare the global best and the current best, if the current is better, replace global
            if current_1_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = current_1_best_entity
            current_best_entity_list.append(current_1_best_entity)
            global_best_entity_list.append(self.global_best_entity)

            # Calculate the average and variance of each dimension
            tmp_x_avg_list = []
            tmp_sigma_list = []
            for x_index in range(len(self.limits_list)):
                tmp_list = []
                for entity_index in range(self.entity_num):
                    x = self.entities_list[entity_index].x_list[x_index]
                    tmp_list.append(x)
                tmp_x_avg = sum(tmp_list)/len(tmp_list)
                tmp_x_avg_list.append(tmp_x_avg)
                tmp_sigma_list.append(math.sqrt(sum([math.pow((x0 - tmp_x_avg), 2) for x0 in tmp_list])/len(tmp_list)))

            # Update the average and sigma list
            self.avg_list = [(1 - alpha) * parent_avg + alpha * (current_1_best_entity.x_list[index] + current_2_best_entity.x_list[index] - current_worst_entity.x_list[index]) for index, parent_avg in enumerate(self.avg_list)]
            self.sigma_list = [(1 - alpha) * parent_sigma + alpha * tmp_sigma for parent_sigma, tmp_sigma in zip(self.sigma_list, tmp_sigma_list)]
        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 500
#     entity_num = 20
#     dim = 10
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     eda = EDA_PBILc_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = eda.evolve()
#     print('Best entity position:', eda.global_best_entity.x_list)
#     print('Fitness:', eda.global_best_entity.fitness)
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
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class EDA_PBILc_1:
    """
    Estimation of distribution algorithm 分布估计算法
        Extending population-based incremental learning to continuous search spaces
    Refer:
        Paper:
            paper0: 分布估计算法综述
                ch02 一个简单的分布估计算法
                    给了一个简单易懂的案例
                ch04 连续域的分布估计算法
            paper1: Estimation of Distribution Algorithms: A New Evolutionary Computation Approach for Graph Matching Problems
                ch02: Estimation Distribution Algorithms
                    2.3 EDAs in Continuous Domains
            paper2: **Extending population-based incremental learning to continuous search spaces
                ch03: Continuous PBIL
                    3.1 Continuous PBIL with dichotomic distributions
                    3.2 Continuous PBIL with Gausssian distributions
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Relaxation factor, alpha
                the only adjustable parameter, is taken as 1e-2
                (paper2-ch3-penultimate paragraph倒数第二段提到alpha很小，缓慢减小父母的值，起到保持父母记忆的作用)

    Attention:
    Version:
        0
    """
    class Entity:
        def __init__(self, limits_list, fitness_function, x_avg_list, x_sigma_list):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_avg_list = x_avg_list
            self.x_sigma_list = x_sigma_list

            self.x_list = [random.gauss(x_avg_list[i], x_sigma_list[i]) for i in range(len(limits_list))]

            # each x generated from Gaussian function might beyond its boundary,
            # have to check its boundary, then calculate its fitness
            # self.fitness = fitness_function(self.x_list)
            self.update()

        def update(self):
            # Check whether the value is in the boundary
            for i in range(len(self.limits_list)):
                # If not, give them a random value from the doable range
                if (self.x_list[i] < self.limits_list[i][0]) or (self.x_list[i] > self.limits_list[i][1]):
                    # Constrain the value in the boundary
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])

            # Update its fitness
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        # Load setting
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize average list,
        self.avg_list = [0.5 * (limit[1] + limit[0]) for limit in limits_list]

        # Initialize sigma_list
        # Default as the length of (0.25 ~ 0.5) * range
        self.sigma_list = [0.25 * (limit[1] - limit[0]) for limit in limits_list]
        """
        Update sigma:
            refer: 
                paper2-ch03-3.2-C
                    Content:
                        Adjust sigma depending on the diversity of the current best offspring;
                        sigma_t (t: current iteration time) is set to the variance of the K best current offspring
                        sigma_t_C = sqrt(sum([Xi - x_mean for i in range(K)]) / K), X = [x0, x1, x2, ..., xn-1]
                        select the K (0.5, half) best entities
                    Result: 
                        the entity becomes premature too early, abandon this method
                paper2-ch03-3.2-D
                    Content:
                        Sigma can be learned in the same way as X itself, 
                        by memorizing the diversity of the K best offspring
                    Result:
                        sigma_t+1_D = sigma_t_D
        """
        self.K = int(0.5 * self.entity_num)
        self.alpha = 1e-2

        # Initialize the global best entity
        self.global_best_entity = self.Entity(limits_list, fitness_function, self.avg_list, self.sigma_list)
        self.global_best_entity.fitness = float('inf')

    def evolve(self):
        current_best_entity_list = []
        global_best_entity_list = []

        for iter_index in range(self.iter_num):
            # Create new generation
            # Initialize a population
            self.entities_list = [self.Entity(self.limits_list, self.fitness_function, self.avg_list, self.sigma_list) \
                                  for i in range(self.entity_num)]
            self.entities_list.sort(key=lambda entity: entity.fitness, reverse=False)
            # Select the first two optimal and the worst entities according to their fitness
            current_1_best_entity, current_2_best_entity = self.entities_list[:2]
            current_worst_entity = self.entities_list[-1]

            # Compare the global best and the current best, if the current is better, replace global
            if current_1_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_1_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_1_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Calculate the average and variance of each dimension
            tmp_x_avg_list = []
            tmp_sigma_list = []
            for x_index in range(len(self.limits_list)):
                tmp_list = []
                for entity_index in range(self.K):
                    x = self.entities_list[entity_index].x_list[x_index]
                    tmp_list.append(x)
                tmp_x_avg = sum(tmp_list)/self.K
                tmp_x_avg_list.append(tmp_x_avg)
                tmp_sigma_list.append( math.sqrt(sum([math.pow((x0 - tmp_x_avg), 2) for x0 in tmp_list]) / self.K) )

            # Update the average and sigma list
            self.avg_list = [(1 - self.alpha) * parent_avg \
                             + self.alpha * (current_1_best_entity.x_list[index] + current_2_best_entity.x_list[index] - current_worst_entity.x_list[index]) \
                             for index, parent_avg in enumerate(self.avg_list)]
            # self.sigma_list = copy.deepcopy(tmp_sigma_list)
            self.sigma_list = [(1 - self.alpha) * parent_sigma + self.alpha * tmp_sigma \
                               for parent_sigma, tmp_sigma in zip(self.sigma_list, tmp_sigma_list)]
        return current_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 2000
#     entity_num = 20
#     dim = 10
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     eda = EDA_PBILc_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_entity_list, global_best_entity_list = eda.evolve()
#     print('Best entity position:', eda.global_best_entity.x_list)
#     print('Fitness:', eda.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
    # for i, c_en, g_en in zip(range(iter_num), current_best_entity_list, global_best_entity_list):
    #     print(i, c_en.fitness, g_en.fitness)

    # import matplotlib.pyplot as plt
    # fig, ax = plt.subplots()
    # line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
    # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    # line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
    # line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    # ax.legend()
    # plt.xlabel('Iteration times')
    # plt.ylabel('Error rate')
    # plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
    # plt.show()

class EDA_PBILc_EIS:
    """
    Estimation of distribution algorithm 分布估计算法
        Extending population-based incremental learning to continuous search spaces
    Refer:
        Paper:
            paper0: 分布估计算法综述
                ch02 一个简单的分布估计算法
                    给了一个简单易懂的案例
                ch04 连续域的分布估计算法
            paper1: Estimation of Distribution Algorithms: A New Evolutionary Computation Approach for Graph Matching Problems
                ch02: Estimation Distribution Algorithms
                    2.3 EDAs in Continuous Domains
            paper2: **Extending population-based incremental learning to continuous search spaces
                ch03: Continuous PBIL
                    3.1 Continuous PBIL with dichotomic distributions
                    3.2 Continuous PBIL with Gausssian distributions
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Relaxation factor, alpha
                the only adjustable parameter, is taken as 1e-2
                (paper2-ch3-penultimate paragraph倒数第二段提到alpha很小，缓慢减小父母的值，起到保持父母记忆的作用)
    Attention:
    Version:
        0
    """
    class Entity:
        def __init__(self, exp_data_dict, fitness_function, x_avg_list, x_sigma_list):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_avg_list = x_avg_list
            self.x_sigma_list = x_sigma_list

            self.x_list = [random.gauss(x_avg_list[i], x_sigma_list[i]) for i in range(len(self.limits_list))]

            # each x generated from Gaussian function might beyond its boundary,
            # have to check its boundary, then calculate its fitness
            # self.fitness = fitness_function(self.x_list)
            self.update()

        def update(self):
            # Check whether the value is in the boundary
            for i in range(len(self.limits_list)):
                # If not, give them a random value from the doable range
                if (self.x_list[i] < self.limits_list[i][0]) or (self.x_list[i] > self.limits_list[i][1]):
                    # Constrain the value in the boundary
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])

            # Update its fitness
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        # Load setting
        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        # Initialize average list,
        self.avg_list = [0.5 * (limit[1] + limit[0]) for limit in self.limits_list]

        # Initialize sigma_list
        # Default as the length of (0.25 ~ 0.5) * range
        self.sigma_list = [0.25 * (limit[1] - limit[0]) for limit in self.limits_list]
        """
        Update sigma:
            refer: 
                paper2-ch03-3.2-C
                    Content:
                        Adjust sigma depending on the diversity of the current best offspring;
                        sigma_t (t: current iteration time) is set to the variance of the K best current offspring
                        sigma_t_C = sqrt(sum([Xi - x_mean for i in range(K)]) / K), X = [x0, x1, x2, ..., xn-1]
                        select the K (0.5, half) best entities
                    Result: 
                        the entity becomes premature too early, abandon this method
                paper2-ch03-3.2-D
                    Content:
                        Sigma can be learned in the same way as X itself, 
                        by memorizing the diversity of the K best offspring
                    Result:
                        sigma_t+1_D = sigma_t_D
        """
        self.K = int(0.5 * self.entity_num)
        self.alpha = 1e-2

        # Initialize the global best entity
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function, self.avg_list, self.sigma_list)

    def search(self):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Create new generation
            # Initialize a population
            self.entities_list = [self.Entity(self.exp_data_dict, self.fitness_function, self.avg_list, self.sigma_list) \
                                  for i in range(self.entity_num)]
            self.entities_list.sort(key=lambda entity: entity.fitness, reverse=False)
            # Select the first two optimal and the worst entities according to their fitness
            current_1_best_entity, current_2_best_entity = self.entities_list[:2]
            current_worst_entity = self.entities_list[-1]

            # Compare the global best and the current best, if the current is better, replace global
            if current_1_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_1_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_1_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Calculate the average and variance of each dimension
            tmp_x_avg_list = []
            tmp_sigma_list = []
            for x_index in range(len(self.limits_list)):
                tmp_list = []
                for entity_index in range(self.K):
                    x = self.entities_list[entity_index].x_list[x_index]
                    tmp_list.append(x)
                tmp_x_avg = sum(tmp_list)/self.K
                tmp_x_avg_list.append(tmp_x_avg)
                tmp_sigma_list.append( math.sqrt(sum([math.pow((x0 - tmp_x_avg), 2) for x0 in tmp_list]) / self.K) )

            # Update the average and sigma list
            self.avg_list = [(1 - self.alpha) * parent_avg \
                             + self.alpha * (current_1_best_entity.x_list[index] + current_2_best_entity.x_list[index] - current_worst_entity.x_list[index]) \
                             for index, parent_avg in enumerate(self.avg_list)]
            # self.sigma_list = copy.deepcopy(tmp_sigma_list)
            self.sigma_list = [(1 - self.alpha) * parent_sigma + self.alpha * tmp_sigma \
                               for parent_sigma, tmp_sigma in zip(self.sigma_list, tmp_sigma_list)]

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

class EDA_PBILc_EIS_access:
    """
    Estimation of distribution algorithm 分布估计算法
        Extending population-based incremental learning to continuous search spaces
    Refer:
        Paper:
            paper0: 分布估计算法综述
                ch02 一个简单的分布估计算法
                    给了一个简单易懂的案例
                ch04 连续域的分布估计算法
            paper1: Estimation of Distribution Algorithms: A New Evolutionary Computation Approach for Graph Matching Problems
                ch02: Estimation Distribution Algorithms
                    2.3 EDAs in Continuous Domains
            paper2: **Extending population-based incremental learning to continuous search spaces
                ch03: Continuous PBIL
                    3.1 Continuous PBIL with dichotomic distributions
                    3.2 Continuous PBIL with Gausssian distributions
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Relaxation factor, alpha
                the only adjustable parameter, is taken as 1e-2
                (paper2-ch3-penultimate paragraph倒数第二段提到alpha很小，缓慢减小父母的值，起到保持父母记忆的作用)
    Attention:
    Version:
        0
    """
    class Entity:
        def __init__(self, exp_data_dict, fitness_function, x_avg_list, x_sigma_list):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_avg_list = x_avg_list
            self.x_sigma_list = x_sigma_list

            self.x_list = [random.gauss(x_avg_list[i], x_sigma_list[i]) for i in range(len(self.limits_list))]

            # each x generated from Gaussian function might beyond its boundary,
            # have to check its boundary, then calculate its fitness
            # self.fitness = fitness_function(self.x_list)
            self.update()

        def update(self):
            # Check whether the value is in the boundary
            for i in range(len(self.limits_list)):
                # If not, give them a random value from the doable range
                if (self.x_list[i] < self.limits_list[i][0]) or (self.x_list[i] > self.limits_list[i][1]):
                    # Constrain the value in the boundary
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])

            # Update its fitness
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        # Load setting
        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        # Initialize average list,
        self.avg_list = [0.5 * (limit[1] + limit[0]) for limit in self.limits_list]

        # Initialize sigma_list
        # Default as the length of (0.25 ~ 0.5) * range
        self.sigma_list = [0.25 * (limit[1] - limit[0]) for limit in self.limits_list]
        """
        Update sigma:
            refer: 
                paper2-ch03-3.2-C
                    Content:
                        Adjust sigma depending on the diversity of the current best offspring;
                        sigma_t (t: current iteration time) is set to the variance of the K best current offspring
                        sigma_t_C = sqrt(sum([Xi - x_mean for i in range(K)]) / K), X = [x0, x1, x2, ..., xn-1]
                        select the K (0.5, half) best entities
                    Result: 
                        the entity becomes premature too early, abandon this method
                paper2-ch03-3.2-D
                    Content:
                        Sigma can be learned in the same way as X itself, 
                        by memorizing the diversity of the K best offspring
                    Result:
                        sigma_t+1_D = sigma_t_D
        """
        self.K = int(0.5 * self.entity_num)
        self.alpha = 1e-2

        # Initialize the global best entity
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function, self.avg_list, self.sigma_list)

    def search(self, res_fn, start_time):
        current_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Create new generation
            # Initialize a population
            self.entities_list = [self.Entity(self.exp_data_dict, self.fitness_function, self.avg_list, self.sigma_list) \
                                  for i in range(self.entity_num)]
            self.entities_list.sort(key=lambda entity: entity.fitness, reverse=False)
            # Select the first two optimal and the worst entities according to their fitness
            current_1_best_entity, current_2_best_entity = self.entities_list[:2]
            current_worst_entity = self.entities_list[-1]

            # Compare the global best and the current best, if the current is better, replace global
            if current_1_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_1_best_entity)
            current_best_entity_list.append(copy.deepcopy(current_1_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Calculate the average and variance of each dimension
            tmp_x_avg_list = []
            tmp_sigma_list = []
            for x_index in range(len(self.limits_list)):
                tmp_list = []
                for entity_index in range(self.K):
                    x = self.entities_list[entity_index].x_list[x_index]
                    tmp_list.append(x)
                tmp_x_avg = sum(tmp_list)/self.K
                tmp_x_avg_list.append(tmp_x_avg)
                tmp_sigma_list.append( math.sqrt(sum([math.pow((x0 - tmp_x_avg), 2) for x0 in tmp_list]) / self.K) )

            # Update the average and sigma list
            self.avg_list = [(1 - self.alpha) * parent_avg \
                             + self.alpha * (current_1_best_entity.x_list[index] + current_2_best_entity.x_list[index] - current_worst_entity.x_list[index]) \
                             for index, parent_avg in enumerate(self.avg_list)]
            # self.sigma_list = copy.deepcopy(tmp_sigma_list)
            self.sigma_list = [(1 - self.alpha) * parent_sigma + self.alpha * tmp_sigma \
                               for parent_sigma, tmp_sigma in zip(self.sigma_list, tmp_sigma_list)]

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

def access_EDA_PBILc_EIS():
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
            goa = EDA_PBILc_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'eda_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('EDA left: {0}'.format(900 - counter))
access_EDA_PBILc_EIS()