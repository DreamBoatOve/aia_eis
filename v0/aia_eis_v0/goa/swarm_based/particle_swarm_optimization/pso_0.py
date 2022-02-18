import copy
import math
import random

from time import perf_counter
import os
import sys
sys.path.append('../../../')
from utils.file_utils.filename_utils import get_ecm_num_str, get_Num_len
from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict

from global_optimizations.GOA_criterions import goa_criterion_pack
from GA_pack.fittness_functions.eis_fitness import cal_EIS_WSE_fitness_1

class PSO:
    """
    Refer:
        paper:
            0- Particle Swarm Optimization
                Authors: James Kennedy and Russell Eberhart
                    The finally applied equation of updating particle's velocity is in 3.6 Current Simplified Version
                    vx[][] = vx[][]
                            + 2 * rand() * (pbestx[][] - presentx[][])
                            + 2 * rand() * (pbestx[][gbest] - presentx[][])
                    x[][] = x[][] + vx[][]
                        pbestx[][]: the best particle in current iteration
                        pbestx[][gbest]: the best global particle the iteration so far
                    The authors did try several other ideals to improve the version in 3.6, but nothing useful is found
            1- A Modified Particle Swarm Optimizer
                Authors: Yuhui Shi and Russell Eberhart
                    Add inertia weight(w) into the above original PSO
                    Eq 2a
                        Vid = w * Vid
                              + c1 * rand() * (Pid - Xid)
                              + c2 * rand() * (Pgd - Xid)
                    Eq 2b
                        Xid = Xid + Vid
    """
    class Particle:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            # Velocity is assigned as one third of the dimension length
            self.v = [(limit[1] - limit[0]) / 3 for limit in limits_list]
            self.fitness = fitness_function(self.position_list)

        def update(self, Vmax_list):
            # 1-The Velocity should be smaller than the Vmax in each dimension
            for v_index in range(len(self.v)):
                if abs(self.v[v_index]) > Vmax_list[v_index]:
                    if self.v[v_index] > 0:
                        self.v[v_index] = Vmax_list[v_index]
                    else:
                        self.v[v_index] = - Vmax_list[v_index]

            # 2-Update the position; The position should be in the boundary
            for pos_index in range(len(self.position_list)):
                self.position_list[pos_index] += self.v[pos_index]
                if self.position_list[pos_index] < self.limits_list[pos_index][0]:
                    self.position_list[pos_index] = self.limits_list[pos_index][0]
                if self.position_list[pos_index] > self.limits_list[pos_index][1]:
                    self.position_list[pos_index] = self.limits_list[pos_index][1]

            # 3-Calculate the new fitness according to the new position
            self.fitness = self.fitness_function(self.position_list)

    def __init__(self, iter_num, particle_num, limits_list, fitness_function, c1 = 2, c2 = 2):
        self.iter_num = iter_num
        self.particle_num = particle_num
        self.limits_list = limits_list

        # Set the Vmax in each dimension, assign the Vmax = Dimension_length / 3
        self.Vmax_list = [(limit[1] - limit[0]) / 3.0 for limit in limits_list]

        self.fitness_function = fitness_function

        # Linearly decreasing inertia weight 0.9 -> 0.4
        self.inertia_weight = 0.9

        self.c1 = c1
        self.c2 = c2

        # Initialize the particle swarm
        self.particles_list = [self.Particle(limits_list, fitness_function) for i in range(self.particle_num)]

        self.global_best_particle = self.Particle(limits_list, fitness_function)
        self.global_best_particle.fitness = float('inf')

    def search(self):
        cur_best_particle_list = []
        global_best_particle_list = []
        for iter_index in range(self.iter_num):
            # Find the current and global best particles
            cur_best_particle = sorted(self.particles_list, key=lambda particle:particle.fitness, reverse=False)[0]
            if cur_best_particle.fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(cur_best_particle)
            cur_best_particle_list.append(cur_best_particle)
            global_best_particle_list.append(self.global_best_particle)

            # Calculate inertia weight
            w = 0.9 - (0.9 - 0.4) * iter_index / self.iter_num

            for p_index in range(self.particle_num):
                for v_index in range(len(self.limits_list)):
                    self.particles_list[p_index].v[v_index] = w * self.particles_list[p_index].v[v_index] \
                    + self.c1 * random.random() * (cur_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index]) \
                    + self.c1 * random.random() * (self.global_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index])
                self.particles_list[p_index].update(self.Vmax_list)
        return cur_best_particle_list, global_best_particle_list

# ---------------- Test PSO on f1 function (f1(x) = x**2) ----------------
# if __name__ == '__main__':
#     iter_num = 2000
#     particle_num = 20
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     pso = PSO(iter_num, particle_num, f1_limits_list, f1_fitness_function)
#     cur_best_particle_list, global_best_particle_list = pso.search()
#     print('Best particle position:', pso.global_best_particle.position_list)
#     print('Fitness:', pso.global_best_particle.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [particle.fitness for particle in cur_best_particle_list]
#     cur_global_fitness_list = [particle.fitness for particle in global_best_particle_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nParticle number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nParticle number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()
# ---------------- Test PSO on f1 function (f1(x) = x**2) ----------------

class PSO_1:
    """
    Refer:
        paper:
            Particle Swarm Optimization
                Authors: James Kennedy and Russell Eberhart
                    The finally applied equation of updating particle's velocity is in 3.6 Current Simplified Version
                    vx[][] = vx[][]
                            + 2 * rand() * (pbestx[][] - presentx[][])
                            + 2 * rand() * (pbestx[][gbest] - presentx[][])
                    x[][] = x[][] + vx[][]
                        pbestx[][]: the best particle in current iteration
                        pbestx[][gbest]: the best global particle the iteration so far
                    The authors did try several other ideals to improve the version in 3.6, but nothing useful is found
            A Modified Particle Swarm Optimizer
                Authors: Yuhui Shi and Russell Eberhart
                    Add inertia weight(w) into the above original PSO
                    Eq 2a
                        Vid = w * Vid
                              + c1 * rand() * (Pid - Xid)
                              + c2 * rand() * (Pgd - Xid)
                    Eq 2b
                        Xid = Xid + Vid
    """
    class Particle:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            # Velocity is assigned as one third of the dimension length
            self.v_list = [(limit[1] - limit[0]) / 3.0 for limit in limits_list]
            self.fitness = fitness_function(self.position_list)

        def update(self, Vmax_list):
            # 1-The Velocity should be smaller than the Vmax in each dimension
            for v_index in range(len(self.v_list)):
                if abs(self.v_list[v_index]) > Vmax_list[v_index]:
                    if self.v_list[v_index] > 0:
                        self.v_list[v_index] = random.uniform(0, Vmax_list[v_index])
                    else:
                        self.v_list[v_index] = - random.uniform(0, Vmax_list[v_index])

            # 2-Update the position; The position should be in the boundary
            for pos_index in range(len(self.position_list)):
                self.position_list[pos_index] += self.v_list[pos_index]
                if (self.position_list[pos_index] < self.limits_list[pos_index][0]) or (self.position_list[pos_index] > self.limits_list[pos_index][1]):
                    self.position_list[pos_index] = random.uniform(self.limits_list[pos_index][0], self.limits_list[pos_index][1])

            # 3-Calculate the new fitness according to the new position
            self.fitness = self.fitness_function(self.position_list)

    def __init__(self, iter_num, particle_num, limits_list, fitness_function, c1 = 2, c2 = 2):
        self.iter_num = iter_num
        self.particle_num = particle_num
        self.limits_list = limits_list

        # Set the Vmax in each dimension, assign the Vmax = Dimension_length / 3
        self.Vmax_list = [(limit[1] - limit[0]) / 3.0 for limit in limits_list]
        self.fitness_function = fitness_function

        """
        inertia_weight, w
            this concept is introduced in paper <A Modified Particle Swarm Optimizer>
                when w > 1, particles tend to explore the search space, global search,
                when w < 1, particles tend to exploit the search space, local search
            Linearly decreasing inertia weight 1.2 -> 0.4
        """
        self.inertia_weight = 1.2

        self.c1 = c1
        self.c2 = c2

        # Initialize the particle swarm
        self.particles_list = [self.Particle(limits_list, fitness_function) for i in range(self.particle_num)]

        self.global_best_particle = self.Particle(limits_list, fitness_function)
        self.global_best_particle.fitness = float('inf')

    def search(self):
        cur_best_particle_list = []
        global_best_particle_list = []
        for iter_index in range(self.iter_num):
            # Find the current and global best particles
            cur_best_particle = sorted(self.particles_list, key=lambda particle:particle.fitness, reverse=False)[0]
            cur_best_particle_list.append(copy.deepcopy(cur_best_particle))

            if cur_best_particle.fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(cur_best_particle)
            global_best_particle_list.append(copy.deepcopy(self.global_best_particle))

            # Calculate inertia weight
            w = self.inertia_weight - 0.8 * iter_index / self.iter_num

            for p_index in range(self.particle_num):
                for v_index in range(len(self.limits_list)):
                    self.particles_list[p_index].v_list[v_index] = w * self.particles_list[p_index].v_list[v_index] \
                                                                    + self.c1 * random.random() * (cur_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index]) \
                                                                    + self.c2 * random.random() * (self.global_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index])
                self.particles_list[p_index].update(self.Vmax_list)
        return cur_best_particle_list, global_best_particle_list

# ---------------- Test PSO_1 on f1 function (f1(x) = x**2) ----------------
# if __name__ == '__main__':
#     iter_num = 1000
#     particle_num = 20
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     pso = PSO_1(iter_num, particle_num, f1_limits_list, f1_fitness_function)
#     cur_best_particle_list, global_best_particle_list = pso.search()
#     print('Best particle position:', pso.global_best_particle.position_list)
#     print('Fitness:', pso.global_best_particle.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [particle.fitness for particle in cur_best_particle_list]
#     cur_global_fitness_list = [particle.fitness for particle in global_best_particle_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nParticle number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nParticle number {1}\nDimension {2}'.format(iter_num, particle_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()
# ---------------- Test PSO_1 on f1 function (f1(x) = x**2) ----------------

class PSO_EIS:
    """
    Refer:
        paper:
            Particle Swarm Optimization
                Authors: James Kennedy and Russell Eberhart
                    The finally applied equation of updating particle's velocity is in 3.6 Current Simplified Version
                    vx[][] = vx[][]
                            + 2 * rand() * (pbestx[][] - presentx[][])
                            + 2 * rand() * (pbestx[][gbest] - presentx[][])
                    x[][] = x[][] + vx[][]
                        pbestx[][]: the best particle in current iteration
                        pbestx[][gbest]: the best global particle the iteration so far
                    The authors did try several other ideals to improve the version in 3.6, but nothing useful is found
            A Modified Particle Swarm Optimizer
                Authors: Yuhui Shi and Russell Eberhart
                    Add inertia weight(w) into the above original PSO
                    Eq 2a
                        Vid = w * Vid
                              + c1 * rand() * (Pid - Xid)
                              + c2 * rand() * (Pgd - Xid)
                    Eq 2b
                        Xid = Xid + Vid
    """
    class Particle:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Velocity is assigned as one third of the dimension length
            self.v_list = [random.uniform(0, (limit[1] - limit[0]) / 3) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update(self, Vmax_list):
            # 1-The Velocity should be smaller than the Vmax in each dimension
            for v_index in range(len(self.v_list)):
                if abs(self.v_list[v_index]) > Vmax_list[v_index]:
                    if self.v_list[v_index] > 0:
                        self.v_list[v_index] = random.uniform(0, Vmax_list[v_index])
                    else:
                        self.v_list[v_index] = - random.uniform(0, Vmax_list[v_index])

            # 2-Update the position; The position should be in the boundary
            for pos_index in range(len(self.position_list)):
                self.position_list[pos_index] += self.v_list[pos_index]
                if (self.position_list[pos_index] < self.limits_list[pos_index][0]) or (self.position_list[pos_index] > self.limits_list[pos_index][1]):
                    self.position_list[pos_index] = random.uniform(self.limits_list[pos_index][0], self.limits_list[pos_index][1])

            # 3-Calculate the new fitness according to the new position
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, particle_num, fitness_function=cal_EIS_WSE_fitness_1, c1 = 2, c2 = 2):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.particle_num = particle_num

        # Set the Vmax in each dimension, assign the Vmax = Dimension_length / 3
        self.Vmax_list = [(limit[1] - limit[0]) / 3 for limit in self.limits_list]
        self.fitness_function = fitness_function

        """
        inertia_weight, w
            this concept is introduced in paper <A Modified Particle Swarm Optimizer>
                when w > 1, particles tend to explore the search space, global search,
                when w < 1, particles tend to exploit the search space, local search
            Linearly decreasing inertia weight 1.2 -> 0.4
        """
        self.inertia_weight = 1.2

        self.c1 = c1
        self.c2 = c2

        # Initialize the particle swarm
        self.particles_list = [self.Particle(self.exp_data_dict, fitness_function) for i in range(self.particle_num)]

        self.global_best_particle = self.Particle(self.exp_data_dict, fitness_function)
        self.global_best_particle.fitness = float('inf')

    def search(self):
        cur_best_particle_list = []
        global_best_particle_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Find the current and global best particles
            cur_best_particle = sorted(self.particles_list, key=lambda particle: particle.fitness, reverse=False)[0]
            cur_best_particle_list.append(copy.deepcopy(cur_best_particle))

            if cur_best_particle.fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(cur_best_particle)
            global_best_particle_list.append(copy.deepcopy(self.global_best_particle))

            # Calculate inertia weight
            w = self.inertia_weight - 0.8 * iter / self.iter_time

            for p_index in range(self.particle_num):
                for v_index in range(len(self.limits_list)):
                    self.particles_list[p_index].v_list[v_index] = w * self.particles_list[p_index].v_list[v_index] \
                                                                   + self.c1 * random.random() * (cur_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index]) \
                                                                   + self.c2 * random.random() * (self.global_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index])
                self.particles_list[p_index].update(self.Vmax_list)
            if iter >= 1:
                x_lists_list = [copy.deepcopy(global_best_particle_list[-2].position_list), copy.deepcopy(global_best_particle_list[-1].position_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter, max_iter_time=self.iter_time, data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return cur_best_particle_list, global_best_particle_list, iter, chi_squared

class PSO_EIS_access:
    """
    Refer:
        paper:
            Particle Swarm Optimization
                Authors: James Kennedy and Russell Eberhart
                    The finally applied equation of updating particle's velocity is in 3.6 Current Simplified Version
                    vx[][] = vx[][]
                            + 2 * rand() * (pbestx[][] - presentx[][])
                            + 2 * rand() * (pbestx[][gbest] - presentx[][])
                    x[][] = x[][] + vx[][]
                        pbestx[][]: the best particle in current iteration
                        pbestx[][gbest]: the best global particle the iteration so far
                    The authors did try several other ideals to improve the version in 3.6, but nothing useful is found
            A Modified Particle Swarm Optimizer
                Authors: Yuhui Shi and Russell Eberhart
                    Add inertia weight(w) into the above original PSO
                    Eq 2a
                        Vid = w * Vid
                              + c1 * rand() * (Pid - Xid)
                              + c2 * rand() * (Pgd - Xid)
                    Eq 2b
                        Xid = Xid + Vid
    """
    class Particle:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            # Velocity is assigned as one third of the dimension length
            self.v_list = [random.uniform(0, (limit[1] - limit[0]) / 3) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update(self, Vmax_list):
            # 1-The Velocity should be smaller than the Vmax in each dimension
            for v_index in range(len(self.v_list)):
                if abs(self.v_list[v_index]) > Vmax_list[v_index]:
                    if self.v_list[v_index] > 0:
                        self.v_list[v_index] = random.uniform(0, Vmax_list[v_index])
                    else:
                        self.v_list[v_index] = - random.uniform(0, Vmax_list[v_index])

            # 2-Update the position; The position should be in the boundary
            for pos_index in range(len(self.position_list)):
                self.position_list[pos_index] += self.v_list[pos_index]
                if (self.position_list[pos_index] < self.limits_list[pos_index][0]) or (self.position_list[pos_index] > self.limits_list[pos_index][1]):
                    self.position_list[pos_index] = random.uniform(self.limits_list[pos_index][0], self.limits_list[pos_index][1])

            # 3-Calculate the new fitness according to the new position
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, particle_num, fitness_function=cal_EIS_WSE_fitness_1, c1 = 2, c2 = 2):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.particle_num = particle_num

        # Set the Vmax in each dimension, assign the Vmax = Dimension_length / 3
        self.Vmax_list = [(limit[1] - limit[0]) / 3 for limit in self.limits_list]
        self.fitness_function = fitness_function

        """
        inertia_weight, w
            this concept is introduced in paper <A Modified Particle Swarm Optimizer>
                when w > 1, particles tend to explore the search space, global search,
                when w < 1, particles tend to exploit the search space, local search
            Linearly decreasing inertia weight 1.2 -> 0.4
        """
        self.inertia_weight = 1.2

        self.c1 = c1
        self.c2 = c2

        # Initialize the particle swarm
        self.particles_list = [self.Particle(self.exp_data_dict, fitness_function) for i in range(self.particle_num)]

        self.global_best_particle = self.Particle(self.exp_data_dict, fitness_function)
        self.global_best_particle.fitness = float('inf')

    def search(self, res_fn, start_time):
        cur_best_particle_list = []
        global_best_particle_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Find the current and global best particles
            cur_best_particle = sorted(self.particles_list, key=lambda particle: particle.fitness, reverse=False)[0]
            cur_best_particle_list.append(copy.deepcopy(cur_best_particle))

            if cur_best_particle.fitness < self.global_best_particle.fitness:
                self.global_best_particle = copy.deepcopy(cur_best_particle)
            global_best_particle_list.append(copy.deepcopy(self.global_best_particle))

            # Calculate inertia weight
            w = self.inertia_weight - 0.8 * iter / self.iter_time

            for p_index in range(self.particle_num):
                for v_index in range(len(self.limits_list)):
                    self.particles_list[p_index].v_list[v_index] = w * self.particles_list[p_index].v_list[v_index] \
                                                                   + self.c1 * random.random() * (cur_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index]) \
                                                                   + self.c2 * random.random() * (self.global_best_particle.position_list[v_index] - self.particles_list[p_index].position_list[v_index])
                self.particles_list[p_index].update(self.Vmax_list)
            if iter >= 1:
                x_lists_list = [copy.deepcopy(global_best_particle_list[-2].position_list),\
                                copy.deepcopy(global_best_particle_list[-1].position_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list,\
                                                                iter=iter, \
                                                                max_iter_time=self.iter_time, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_particle_list[-1].position_list]) \
                           + '],' + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_PSO_EIS():
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
            goa = PSO_EIS_access(exp_data_dict=sim_ecm, iter_time=10000, particle_num=10*para_num)
            res_fn = 'pso_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('PSO left: {0}'.format(900 - counter))
access_PSO_EIS()