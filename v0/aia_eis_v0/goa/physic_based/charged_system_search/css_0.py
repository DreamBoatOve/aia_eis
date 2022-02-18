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

def css_particle_distance(x1_list, x2_list, xbest_list):
    numerator = math.sqrt(sum([pow(x1 - x2, 2) for x1, x2 in zip(x1_list, x2_list)]))
    denominator = 1e-5 + math.sqrt(sum([pow((x1 + x2) / 2 - xb, 2) for x1, x2, xb in zip(x1_list, x2_list, xbest_list)]))
    return numerator / denominator

class CSS:
    """
    Refer:
        paper: A novel heuristic optimization method: charged system search
    Version;
        First version
    """
    class Particle:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.v_list = [0.0 for i in range(len(self.limits_list))]

            self.fitness = fitness_function(self.x_list)

        def update(self, x_pack_list):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    # The update rule for x that exceed boundaries is unique and complicated.
                    # See Eq 27, the related parameters are refered from harmony search article.
                    # change rate = 0.5, pitch adjusting rate = 0.1
                    if random.random() < 0.5:
                        x_candidate_list = [x_list[i] for x_list in x_pack_list]
                        tmp_x = random.sample(x_candidate_list, 1)[0]
                        if random.random() < 0.1:
                            dim_range = self.limits_list[i][1] - self.limits_list[i][0]
                            up_boundary = tmp_x + dim_range * 0.05
                            if up_boundary > self.limits_list[i][1]:
                                up_boundary = self.limits_list[i][1]
                            down_boundary = tmp_x - dim_range * 0.05
                            if down_boundary < self.limits_list[i][0]:
                                down_boundary = self.limits_list[i][0]
                            self.x_list[i] = random.uniform(down_boundary, up_boundary)
                        else:
                            self.x_list[i] = tmp_x
                    else:
                        # Randomly generate x
                        self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, particle_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.particle_num = particle_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.particle_list = [self.Particle(limits_list, fitness_function) for i in range(self.particle_num)]
        self.global_best_particle = self.Particle(limits_list, fitness_function)

        # Each dimension has the same particle radius, and the radius is determined by Eq 21
        self.particle_radius = 0.1 * max([limit[1] - limit[0] for limit in self.limits_list])
        # Ke = Coulomb constant
        self.ke = 10000000

    def search(self):
        cur_best_particle_list = []
        global_best_particle_list = []
        for iter_index in range(self.iter_num):
            self.ke = self.ke * (1 - iter_index / self.iter_num)
            Kv = 0.5 * (1 - iter_index / self.iter_num)
            Ka = 0.5 * (1 + iter_index / self.iter_num)

            sorted_particle_list = sorted(self.particle_list, key=lambda particle: particle.fitness, reverse=False)
            charged_memory_list = sorted_particle_list[:int(self.particle_num / 4)]
            charged_memory_x_pack_list = [c_m.x_list for c_m in charged_memory_list]

            cur_best_particle = sorted_particle_list[0]
            if cur_best_particle.fitness < self.global_best_particle.fitness:
                self.global_best_particle = cur_best_particle
            cur_best_particle_list.append(cur_best_particle)
            global_best_particle_list.append(self.global_best_particle)

            # Calculate the quality of all solution
            sorted_q_list = [(s_p.fitness - sorted_particle_list[-1].fitness) / (sorted_particle_list[0].fitness - sorted_particle_list[-1].fitness) for s_p in sorted_particle_list]
            sorted_q_list[-1] = sorted_q_list[-2] / 2

            tmp_particle_list = []
            for p_i, pa_i in enumerate(sorted_particle_list):
                a_pack_list = []
                for p_j, pa_j in enumerate(sorted_particle_list):
                    if (p_i != p_j) and (pa_j.fitness < pa_i.fitness):
                        d = css_particle_distance(pa_i.x_list, pa_j.x_list, self.global_best_particle.x_list)

                        a_list = []
                        # Calculate the force in each dimension by Eq 5
                        for dim_i in range(len(self.limits_list)):
                            # Choose Force equation according to the distance
                            try:
                                if d > self.particle_radius:
                                    # force = self.ke * sorted_q_list[p_i] * sorted_q_list[p_j] * (pa_j.x_list[dim_i] - pa_i.x_list[dim_i]) / (pow(d, 2) * abs(pa_j.x_list[dim_i] - pa_i.x_list[dim_i]))
                                    a = self.ke * sorted_q_list[p_j] * (pa_j.x_list[dim_i] - pa_i.x_list[dim_i]) / (pow(d, 2) * abs(pa_j.x_list[dim_i] - pa_i.x_list[dim_i]))
                                else:
                                    # force = self.ke * sorted_q_list[p_i] * sorted_q_list[p_j] * (pa_j.x_list[dim_i] - pa_i.x_list[dim_i]) / (pow(self.particle_radius, 3) * abs(pa_j.x_list[dim_i] - pa_i.x_list[dim_i]))
                                    a = self.ke * sorted_q_list[p_j] * (pa_j.x_list[dim_i] - pa_i.x_list[dim_i]) / (pow(self.particle_radius, 3) * abs(pa_j.x_list[dim_i] - pa_i.x_list[dim_i]))
                                a_list.append(a)
                            except ZeroDivisionError as e:
                                a_list.append(0.0)
                        a_pack_list.append(a_list)

                # Cumulate the acceleration
                a_sum_list = [sum([al[i] for al in a_pack_list]) for i in range(len(self.limits_list))]
                tmp_x_list = [random.random() * Ka * a + random.random() * Kv * v + x for a, v, x in zip(a_sum_list, pa_i.v_list, pa_i.x_list)]
                tmp_v_list = [tmp_x - x for tmp_x, x in zip(tmp_x_list, pa_i.x_list)]
                tmp_particle = self.Particle(self.limits_list, self.fitness_function)
                tmp_particle.x_list = tmp_x_list
                tmp_particle.v_list = tmp_v_list
                tmp_particle.update(charged_memory_x_pack_list)
                tmp_particle_list.append(tmp_particle)
            self.particle_list = copy.deepcopy(tmp_particle_list)
        return cur_best_particle_list, global_best_particle_list

# if __name__ == '__main__':
#     iter_num = 500
#     particle_num = 4
#     dim = 4
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     css = CSS(iter_num, particle_num, f1_limits_list, f1_fitness_function)
#     cur_best_particle_list, global_best_particle_list = css.search()
#     print('Best particle position:', css.global_best_particle.x_list)
#     print('Fitness:', css.global_best_particle.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_particle_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_particle_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

def css_particle_distance_on_1_dim(x1_list, x2_list, xbest_list, dim):
    x1, x2, xb = x1_list[dim], x2_list[dim], xbest_list[dim]
    d = abs(x1 - x2) / (abs((x1+x2)/2.0 - xb) + 1e-10)
    return d

class CSS_1:
    """
    Refer:
        paper: A novel heuristic optimization method: charged system search
    Version;
        First Version
        Second version
            1- Setting of radius in each dimension
                The first version is implemented according to the above paper,
                it does not consider that the scales of different dimensions are different.
                Like the Eq 21 setting the radius, if one dimension, d1 [0~1], another dimension,
                d2 [10 ~ 100], the radius will be r = max(0.1 * 1, 0.1 * 90) = 9, r is obviously not suitable on d1.
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Ke: Coulomb constant, very IMPORTANT
            CMCR: Charged Memory Changing Rate
            PAR: Pitch Adjusting Rate
            CMS: Size of Charged Memory
    """
    class Particle:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.v_list = [0.0 for i in range(len(self.limits_list))]

            self.fitness = fitness_function(self.x_list)

        def cal_q(self, extreme_fit_pair):
            # extreme_fit_pair: [the best fitness in all iterations, the worst fitness in all iterations]
            best_fit, worst_fit = extreme_fit_pair[0], extreme_fit_pair[1]
            self.q = (self.fitness - worst_fit) / (best_fit - worst_fit)

        def update(self, x_pack_list):
            # The update rule for x that exceed boundaries is unique and complicated.
            # See Eq 27, the related parameters are refered from harmony search article.
            CMCR = 0.95
            # CMCR = 0.75
            # CMCR = 0.5
            # CMCR = 0.25
            PAR = 0.1

            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    if random.random() < CMCR:
                        x_candidate_list = [x_list[i] for x_list in x_pack_list]
                        tmp_x = random.sample(x_candidate_list, 1)[0]
                        if random.random() < PAR:
                            dim_range = self.limits_list[i][1] - self.limits_list[i][0]
                            up_boundary = tmp_x + dim_range * 0.05
                            if up_boundary > self.limits_list[i][1]:
                                up_boundary = self.limits_list[i][1]
                            down_boundary = tmp_x - dim_range * 0.05
                            if down_boundary < self.limits_list[i][0]:
                                down_boundary = self.limits_list[i][0]
                            self.x_list[i] = random.uniform(down_boundary, up_boundary)
                        else:
                            self.x_list[i] = tmp_x
                    else:
                        # Randomly generate x
                        self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, particle_num, limits_list, fitness_function):
        """
        :param
            iter_num:
            particle_num:
            limits_list:
            fitness_function:
            CMS:
            CM_list:
        """
        self.iter_num = iter_num
        self.particle_num = particle_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Each dimension has the same particle radius, and the radius is determined by Eq 21
        # In order to handle the situation that different dimensions have different scale, the radius is varied
        self.particle_radius_list = [0.1 * m for m in [limit[1] - limit[0] for limit in self.limits_list]]

        # Size of Charged Memory
        self.CMS = int(self.particle_num / 4.0)

        # Ke = Coulomb constant = 9 * 1e9
        self.ke = 10000000

        self.particle_list = [self.Particle(limits_list, fitness_function) for i in range(self.particle_num)]
        self.particle_list.sort(key=lambda p: p.fitness, reverse=False)
        self.CM_list = copy.deepcopy(self.particle_list[: self.CMS])

        self.global_best_particle = copy.deepcopy(self.particle_list[0])
        self.global_worst_particle = copy.deepcopy(self.particle_list[-1])

    def search(self):
        cur_best_particle_list = []
        global_best_particle_list = []

        for iter_index in range(self.iter_num):
            # self.ke = self.ke * (1 - iter_index / self.iter_num)
            Kv = 0.5 * (1 - iter_index / self.iter_num)
            Ka = 0.5 * (1 + iter_index / self.iter_num)

            if self.particle_list[0].fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(self.particle_list[0])
            cur_best_particle_list.append(copy.deepcopy(self.particle_list[0]))
            global_best_particle_list.append(copy.deepcopy(self.global_best_particle))

            if self.particle_list[-1].fitness > self.global_worst_particle.fitness:
                self.global_worst_particle = copy.deepcopy(self.particle_list[-1])
            extreme_fit_pair = [self.global_best_particle.fitness, self.global_worst_particle.fitness]
            # Calculate charge of each particle
            for i in range(self.particle_num):
                self.particle_list[i].cal_q(extreme_fit_pair)

            for i in range(self.particle_num):
                p_i = self.particle_list[i]
                a_pack_list = []
                for j in range(self.particle_num):
                    p_j = self.particle_list[j]
                    a_list = []
                    if (i != j):
                        # Good attracts bad in most time
                        if p_j.fitness <= p_i.fitness:
                            for dim_i in range(len(self.limits_list)):
                                # calculate distance between two particles
                                d = css_particle_distance_on_1_dim(p_i.x_list, p_j.x_list, self.particle_list[0].x_list, dim_i)
                                q_j = self.particle_list[j].q
                                # Judge d > particle_radius or not
                                if d >= self.particle_radius_list[dim_i]:
                                    # a = q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                    a = self.ke * q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                else:
                                    # a = q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                    a = self.ke * q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                a_list.append(a)
                        # Bad attracts good in rare situation, p_j.fitness > p_i.fitness
                        else:
                            chance = (p_i.fitness - self.global_best_particle.fitness) / (p_j.fitness - p_i.fitness)
                            if chance > random.random():
                                for dim_i in range(len(self.limits_list)):
                                    # calculate distance between two particles
                                    d = css_particle_distance_on_1_dim(p_i.x_list, p_j.x_list, self.particle_list[0].x_list, dim_i)
                                    q_j = self.particle_list[j].q
                                    # Judge d > particle_radius or not
                                    if d >= self.particle_radius_list[dim_i]:
                                        # a = q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                        a = self.ke * q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                    else:
                                        # a = q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                        a = self.ke * q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                    a_list.append(a)
                            # else:
                                # a_list = [0 for i in range(len(self.limits_list))]
                    # else:
                    #     a_list = [0 for i in range(len(self.limits_list))]
                    # 在对粒子i不产生加速度的情况下，a_list为空，添加进a_pack_list没有意义
                    if len(a_list) == len(self.limits_list):
                        a_pack_list.append(a_list)
                try:
                    a_sum_list = [sum([tmp_a[i] for tmp_a in a_pack_list]) for i in range(len(self.limits_list))]
                except IndexError as e:
                    print(e)
                # Update position and velocity of a particle according to acceleration
                tmp_x_list = [random.random() * Ka * a + random.random() * Kv * v + x\
                              for a, v, x in zip(a_sum_list, p_i.v_list, p_i.x_list)]
                memory_x_pack_list = [p.x_list for p in self.CM_list]
                x_old_list = copy.deepcopy(p_i.x_list)
                p_i.x_list = tmp_x_list
                p_i.update(x_pack_list=memory_x_pack_list)
                p_i.v_list = [x_new - x_old for x_new, x_old in zip(p_i.x_list, x_old_list)]

                # Update Memory only when a better particle appears
                if p_i.fitness < self.CM_list[-1].fitness:
                    self.CM_list[-1] = copy.deepcopy(p_i)
                    self.CM_list.sort(key=lambda p : p.fitness, reverse=False)
            self.particle_list.sort(key=lambda p: p.fitness, reverse=False)
        return cur_best_particle_list, global_best_particle_list

# if __name__ == '__main__':
#     iter_num = 200
#     particle_num = 8
#     dim = 4
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     css = CSS_1(iter_num, particle_num, f1_limits_list, f1_fitness_function)
#     cur_best_particle_list, global_best_particle_list = css.search()
#     print('Best particle position:', css.global_best_particle.x_list)
#     print('Fitness:', css.global_best_particle.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_particle_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_particle_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class CSS_EIS:
    """
    Refer:
        paper: A novel heuristic optimization method: charged system search
    Version;
        First Version
        Second version
            1- Setting of radius in each dimension
                The first version is implemented according to the above paper,
                it does not consider that the scales of different dimensions are different.
                Like the Eq 21 setting the radius, if one dimension, d1 [0~1], another dimension,
                d2 [10 ~ 100], the radius will be r = max(0.1 * 1, 0.1 * 90) = 9, r is obviously not suitable on d1.
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Ke: Coulomb constant, very IMPORTANT
            CMCR: Charged Memory Changing Rate
            PAR: Pitch Adjusting Rate
            CMS: Size of Charged Memory
    """
    class Particle:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.v_list = [0.0 for i in range(len(self.limits_list))]

            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def cal_q(self, extreme_fit_pair):
            # extreme_fit_pair: [the best fitness in all iterations, the worst fitness in all iterations]
            best_fit, worst_fit = extreme_fit_pair[0], extreme_fit_pair[1]
            self.q = (self.fitness - worst_fit) / (best_fit - worst_fit)

        def update(self, x_pack_list):
            # The update rule for x that exceed boundaries is unique and complicated.
            # See Eq 27, the related parameters are refered from harmony search article.
            CMCR = 0.95
            # CMCR = 0.75
            # CMCR = 0.5
            # CMCR = 0.25
            PAR = 0.1

            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    if random.random() < CMCR:
                        x_candidate_list = [x_list[i] for x_list in x_pack_list]
                        tmp_x = random.sample(x_candidate_list, 1)[0]
                        if random.random() < PAR:
                            dim_range = self.limits_list[i][1] - self.limits_list[i][0]
                            up_boundary = tmp_x + dim_range * 0.05
                            if up_boundary > self.limits_list[i][1]:
                                up_boundary = self.limits_list[i][1]
                            down_boundary = tmp_x - dim_range * 0.05
                            if down_boundary < self.limits_list[i][0]:
                                down_boundary = self.limits_list[i][0]
                            self.x_list[i] = random.uniform(down_boundary, up_boundary)
                        else:
                            self.x_list[i] = tmp_x
                    else:
                        # Randomly generate x
                        self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, particle_num, fitness_function=cal_EIS_WSE_fitness_1):
        """
        :param
            iter_num:
            particle_num:
            limits_list:
            fitness_function:
            CMS:
            CM_list:
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.particle_num = particle_num

        self.fitness_function = fitness_function

        # Each dimension has the same particle radius, and the radius is determined by Eq 21
        # In order to handle the situation that different dimensions have different scale, the radius is varied
        self.particle_radius_list = [0.1 * m for m in [limit[1] - limit[0] for limit in self.limits_list]]

        # Size of Charged Memory
        self.CMS = int(self.particle_num / 4.0)

        # Ke = Coulomb constant = 9 * 1e9
        self.ke = 10000000

        self.particle_list = [self.Particle(self.exp_data_dict, fitness_function) for i in range(self.particle_num)]
        self.particle_list.sort(key=lambda p: p.fitness, reverse=False)
        self.CM_list = copy.deepcopy(self.particle_list[: self.CMS])

        self.global_best_particle = copy.deepcopy(self.particle_list[0])
        self.global_worst_particle = copy.deepcopy(self.particle_list[-1])

    def search(self):
        cur_best_particle_list = []
        global_best_particle_list = []

        continue_criterion = True
        iter_index = 0
        while continue_criterion:
            self.ke = self.ke * (1 - iter_index / self.iter_num)
            Kv = 0.5 * (1 - iter_index / self.iter_num)
            Ka = 0.5 * (1 + iter_index / self.iter_num)

            if self.particle_list[0].fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(self.particle_list[0])
            cur_best_particle_list.append(copy.deepcopy(self.particle_list[0]))
            global_best_particle_list.append(copy.deepcopy(self.global_best_particle))

            if self.particle_list[-1].fitness > self.global_worst_particle.fitness:
                self.global_worst_particle = copy.deepcopy(self.particle_list[-1])
            extreme_fit_pair = [self.global_best_particle.fitness, self.global_worst_particle.fitness]
            # Calculate charge of each particle
            for i in range(self.particle_num):
                self.particle_list[i].cal_q(extreme_fit_pair)

            for i in range(self.particle_num):
                p_i = self.particle_list[i]
                a_pack_list = []
                for j in range(self.particle_num):
                    p_j = self.particle_list[j]
                    a_list = []
                    if (i != j):
                        # Good attracts bad in most time
                        if p_j.fitness <= p_i.fitness:
                            for dim_i in range(len(self.limits_list)):
                                # calculate distance between two particles
                                d = css_particle_distance_on_1_dim(p_i.x_list, p_j.x_list, self.particle_list[0].x_list, dim_i)
                                q_j = self.particle_list[j].q
                                # Judge d > particle_radius or not
                                if d >= self.particle_radius_list[dim_i]:
                                    # a = q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                    a = self.ke * q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                else:
                                    # a = q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                    a = self.ke * q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                a_list.append(a)
                        # Bad attracts good in rare situation, p_j.fitness > p_i.fitness
                        else:
                            chance = (p_i.fitness - self.global_best_particle.fitness) / (p_j.fitness - p_i.fitness)
                            if chance > random.random():
                                for dim_i in range(len(self.limits_list)):
                                    # calculate distance between two particles
                                    d = css_particle_distance_on_1_dim(p_i.x_list, p_j.x_list, self.particle_list[0].x_list, dim_i)
                                    q_j = self.particle_list[j].q
                                    # Judge d > particle_radius or not
                                    if d >= self.particle_radius_list[dim_i]:
                                        # a = q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                        a = self.ke * q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                    else:
                                        # a = q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                        a = self.ke * q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                    a_list.append(a)
                            # else:
                                # a_list = [0 for i in range(len(self.limits_list))]
                    # else:
                    #     a_list = [0 for i in range(len(self.limits_list))]
                    # 在对粒子i不产生加速度的情况下，a_list为空，添加进a_pack_list没有意义
                    if len(a_list) == len(self.limits_list):
                        a_pack_list.append(a_list)
                try:
                    a_sum_list = [sum([tmp_a[i] for tmp_a in a_pack_list]) for i in range(len(self.limits_list))]
                except IndexError as e:
                    print(e)
                # Update position and velocity of a particle according to acceleration
                tmp_x_list = [random.random() * Ka * a + random.random() * Kv * v + x\
                              for a, v, x in zip(a_sum_list, p_i.v_list, p_i.x_list)]
                memory_x_pack_list = [p.x_list for p in self.CM_list]
                x_old_list = copy.deepcopy(p_i.x_list)
                p_i.x_list = tmp_x_list
                p_i.update(x_pack_list=memory_x_pack_list)
                p_i.v_list = [x_new - x_old for x_new, x_old in zip(p_i.x_list, x_old_list)]

                # Update Memory only when a better particle appears
                if p_i.fitness < self.CM_list[-1].fitness:
                    self.CM_list[-1] = copy.deepcopy(p_i)
                    self.CM_list.sort(key=lambda p : p.fitness, reverse=False)
            self.particle_list.sort(key=lambda p: p.fitness, reverse=False)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter_index >= 1:
                x_lists_list = [global_best_particle_list[-2].x_list, global_best_particle_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter_index,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter_index += 1
        return cur_best_particle_list, global_best_particle_list, iter_index, chi_squared

class CSS_EIS_access:
    """
    Refer:
        paper: A novel heuristic optimization method: charged system search
    Version;
        First Version
        Second version
            1- Setting of radius in each dimension
                The first version is implemented according to the above paper,
                it does not consider that the scales of different dimensions are different.
                Like the Eq 21 setting the radius, if one dimension, d1 [0~1], another dimension,
                d2 [10 ~ 100], the radius will be r = max(0.1 * 1, 0.1 * 90) = 9, r is obviously not suitable on d1.
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
            Ke: Coulomb constant, very IMPORTANT
            CMCR: Charged Memory Changing Rate
            PAR: Pitch Adjusting Rate
            CMS: Size of Charged Memory
    """
    class Particle:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.v_list = [0.0 for i in range(len(self.limits_list))]

            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def cal_q(self, extreme_fit_pair):
            # extreme_fit_pair: [the best fitness in all iterations, the worst fitness in all iterations]
            best_fit, worst_fit = extreme_fit_pair[0], extreme_fit_pair[1]
            self.q = (self.fitness - worst_fit) / (best_fit - worst_fit)

        def update(self, x_pack_list):
            # The update rule for x that exceed boundaries is unique and complicated.
            # See Eq 27, the related parameters are refered from harmony search article.
            CMCR = 0.95
            # CMCR = 0.75
            # CMCR = 0.5
            # CMCR = 0.25
            PAR = 0.1

            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    if random.random() < CMCR:
                        x_candidate_list = [x_list[i] for x_list in x_pack_list]
                        tmp_x = random.sample(x_candidate_list, 1)[0]
                        if random.random() < PAR:
                            dim_range = self.limits_list[i][1] - self.limits_list[i][0]
                            up_boundary = tmp_x + dim_range * 0.05
                            if up_boundary > self.limits_list[i][1]:
                                up_boundary = self.limits_list[i][1]
                            down_boundary = tmp_x - dim_range * 0.05
                            if down_boundary < self.limits_list[i][0]:
                                down_boundary = self.limits_list[i][0]
                            self.x_list[i] = random.uniform(down_boundary, up_boundary)
                        else:
                            self.x_list[i] = tmp_x
                    else:
                        # Randomly generate x
                        self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, particle_num, fitness_function=cal_EIS_WSE_fitness_1):
        """
        :param
            iter_num:
            particle_num:
            limits_list:
            fitness_function:
            CMS:
            CM_list:
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.particle_num = particle_num

        self.fitness_function = fitness_function

        # Each dimension has the same particle radius, and the radius is determined by Eq 21
        # In order to handle the situation that different dimensions have different scale, the radius is varied
        self.particle_radius_list = [0.1 * m for m in [limit[1] - limit[0] for limit in self.limits_list]]

        # Size of Charged Memory
        self.CMS = int(self.particle_num / 4.0)

        # Ke = Coulomb constant = 9 * 1e9
        self.ke = 10000000

        self.particle_list = [self.Particle(self.exp_data_dict, fitness_function) for i in range(self.particle_num)]
        self.particle_list.sort(key=lambda p: p.fitness, reverse=False)
        self.CM_list = copy.deepcopy(self.particle_list[: self.CMS])

        self.global_best_particle = copy.deepcopy(self.particle_list[0])
        self.global_worst_particle = copy.deepcopy(self.particle_list[-1])

    def search(self, res_fn, start_time):
        cur_best_particle_list = []
        global_best_particle_list = []

        continue_criterion = True
        iter_index = 0
        while continue_criterion:
            self.ke = self.ke * (1 - iter_index / self.iter_num)
            Kv = 0.5 * (1 - iter_index / self.iter_num)
            Ka = 0.5 * (1 + iter_index / self.iter_num)

            if self.particle_list[0].fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(self.particle_list[0])
            cur_best_particle_list.append(copy.deepcopy(self.particle_list[0]))
            global_best_particle_list.append(copy.deepcopy(self.global_best_particle))

            if self.particle_list[-1].fitness > self.global_worst_particle.fitness:
                self.global_worst_particle = copy.deepcopy(self.particle_list[-1])
            extreme_fit_pair = [self.global_best_particle.fitness, self.global_worst_particle.fitness]
            # Calculate charge of each particle
            for i in range(self.particle_num):
                self.particle_list[i].cal_q(extreme_fit_pair)

            for i in range(self.particle_num):
                p_i = self.particle_list[i]
                a_pack_list = []
                for j in range(self.particle_num):
                    p_j = self.particle_list[j]
                    a_list = []
                    if (i != j):
                        # Good attracts bad in most time
                        if p_j.fitness <= p_i.fitness:
                            for dim_i in range(len(self.limits_list)):
                                # calculate distance between two particles
                                d = css_particle_distance_on_1_dim(p_i.x_list, p_j.x_list, self.particle_list[0].x_list, dim_i)
                                q_j = self.particle_list[j].q
                                # Judge d > particle_radius or not
                                if d >= self.particle_radius_list[dim_i]:
                                    # a = q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                    a = self.ke * q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                else:
                                    # a = q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                    a = self.ke * q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                a_list.append(a)
                        # Bad attracts good in rare situation, p_j.fitness > p_i.fitness
                        else:
                            chance = (p_i.fitness - self.global_best_particle.fitness) / (p_j.fitness - p_i.fitness)
                            if chance > random.random():
                                for dim_i in range(len(self.limits_list)):
                                    # calculate distance between two particles
                                    d = css_particle_distance_on_1_dim(p_i.x_list, p_j.x_list, self.particle_list[0].x_list, dim_i)
                                    q_j = self.particle_list[j].q
                                    # Judge d > particle_radius or not
                                    if d >= self.particle_radius_list[dim_i]:
                                        # a = q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                        a = self.ke * q_j * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(d, 2)
                                    else:
                                        # a = q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                        a = self.ke * q_j * d * (p_j.x_list[dim_i] - p_i.x_list[dim_i]) / pow(self.particle_radius_list[dim_i], 3)
                                    a_list.append(a)
                            # else:
                                # a_list = [0 for i in range(len(self.limits_list))]
                    # else:
                    #     a_list = [0 for i in range(len(self.limits_list))]
                    # 在对粒子i不产生加速度的情况下，a_list为空，添加进a_pack_list没有意义
                    if len(a_list) == len(self.limits_list):
                        a_pack_list.append(a_list)
                try:
                    a_sum_list = [sum([tmp_a[i] for tmp_a in a_pack_list]) for i in range(len(self.limits_list))]
                except IndexError as e:
                    print(e)
                # Update position and velocity of a particle according to acceleration
                tmp_x_list = [random.random() * Ka * a + random.random() * Kv * v + x\
                              for a, v, x in zip(a_sum_list, p_i.v_list, p_i.x_list)]
                memory_x_pack_list = [p.x_list for p in self.CM_list]
                x_old_list = copy.deepcopy(p_i.x_list)
                p_i.x_list = tmp_x_list
                p_i.update(x_pack_list=memory_x_pack_list)
                p_i.v_list = [x_new - x_old for x_new, x_old in zip(p_i.x_list, x_old_list)]

                # Update Memory only when a better particle appears
                if p_i.fitness < self.CM_list[-1].fitness:
                    self.CM_list[-1] = copy.deepcopy(p_i)
                    self.CM_list.sort(key=lambda p : p.fitness, reverse=False)
            self.particle_list.sort(key=lambda p: p.fitness, reverse=False)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter_index >= 1:
                x_lists_list = [global_best_particle_list[-2].x_list, global_best_particle_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter_index, \
                                                                max_iter_time=self.iter_num, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter_index) + ',[' \
                           + ','.join([str(para) for para in global_best_particle_list[-1].x_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter_index += 1

def access_CSS_EIS():
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
            goa = CSS_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, particle_num=10*para_num)
            res_fn = 'css_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('CSS left: {0}'.format(900 - counter))
# access_CSS_EIS()