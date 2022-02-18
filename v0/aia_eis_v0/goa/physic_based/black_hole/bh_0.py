import copy
import random

from time import perf_counter
import os
import sys
sys.path.append('../../../')
from utils.file_utils.filename_utils import get_ecm_num_str, get_Num_len
from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict

from goa.GOA_criterions import goa_criterion_pack
from GA_pack.fittness_functions.eis_fitness import cal_EIS_WSE_fitness_1

class BH_0:
    """
    Refer:
        Paper:
            paper0: Black hole: A new heuristic optimization approach for data clustering
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

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = self.fitness_function(self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                limit = self.limits_list[i]
                if (x < limit[0]) or (x > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.limits_list, self.fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.limits_list, self.fitness_function)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        for iter in range(self.iter_num):
            self.entity_list.sort(key=lambda en:en.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Determine the radius of the Black Hole
            fitness_list = [en.fitness for en in self.entity_list]
            self.radius_list = [(limit[1] - limit[0]) * cur_best_entity.fitness / sum(fitness_list) for limit in self.limits_list]

            # Black hole attracts the other stars
            bh_x_list = cur_best_entity.x_list
            in_radius_star_index_list = []
            for i in range(1, self.entity_num):
                star = self.entity_list[i]
                s_x_list = star.x_list
                for j in range(len(self.limits_list)):
                    s_x_list[j] = s_x_list[j] + random.random() * (bh_x_list[j] - s_x_list[j])
                star.update()

                # Check whether this star in the radius of the black hole
                for a in range(len(self.limits_list)):
                    r = self.radius_list[a]
                    if abs(s_x_list[a] - bh_x_list[a]) < r:
                        in_radius_star_index_list.append(i)
                        break

            # Check whether any star is absorded by the black hole, if it does, add new star in next population
            if len(in_radius_star_index_list) > 0:
                for i in reversed(in_radius_star_index_list):
                    del self.entity_list[i]
                self.entity_list = self.entity_list + [self.Entity(self.limits_list, self.fitness_function) for i in range(len(in_radius_star_index_list))]
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 5
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     bh = BH_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     current_best_atom_list, global_best_atom_list = bh.search()
#     print('Best entity position:', bh.global_best_entity.x_list)
#     print('Fitness:', bh.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_atom_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_atom_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, linestyle='-', label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, linestyle='--', label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     # line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class BH_EIS:
    """
    Refer:
        Paper:
            paper0: Black hole: A new heuristic optimization approach for data clustering
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
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                limit = self.limits_list[i]
                if (x < limit[0]) or (x > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num

        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.exp_data_dict, self.fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, self.fitness_function)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entity_list.sort(key=lambda en:en.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Determine the radius of the Black Hole
            fitness_list = [en.fitness for en in self.entity_list]
            self.radius_list = [(limit[1] - limit[0]) * cur_best_entity.fitness / sum(fitness_list) for limit in self.limits_list]

            # Black hole attracts the other stars
            bh_x_list = cur_best_entity.x_list
            in_radius_star_index_list = []
            for i in range(1, self.entity_num):
                star = self.entity_list[i]
                s_x_list = star.x_list
                for j in range(len(self.limits_list)):
                    s_x_list[j] = s_x_list[j] + random.random() * (bh_x_list[j] - s_x_list[j])
                star.update()

                # Check whether this star in the radius of the black hole
                for a in range(len(self.limits_list)):
                    r = self.radius_list[a]
                    if abs(s_x_list[a] - bh_x_list[a]) < r:
                        in_radius_star_index_list.append(i)
                        break

            # Check whether any star is absorded by the black hole, if it does, add new star in next population
            if len(in_radius_star_index_list) > 0:
                for i in reversed(in_radius_star_index_list):
                    del self.entity_list[i]
                self.entity_list = self.entity_list + [self.Entity(self.exp_data_dict, self.fitness_function) for i in range(len(in_radius_star_index_list))]

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

class BH_EIS_access:
    """
    Refer:
        Paper:
            paper0: Black hole: A new heuristic optimization approach for data clustering
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
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                limit = self.limits_list[i]
                if (x < limit[0]) or (x > limit[1]):
                    self.x_list[i] = random.uniform(limit[0], limit[1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num

        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.exp_data_dict, self.fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, self.fitness_function)

    def search(self, res_fn, start_time):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entity_list.sort(key=lambda en:en.fitness, reverse=False)
            cur_best_entity = self.entity_list[0]
            if cur_best_entity.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(cur_best_entity)
            cur_best_entity_list.append(copy.deepcopy(cur_best_entity))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Determine the radius of the Black Hole
            fitness_list = [en.fitness for en in self.entity_list]
            self.radius_list = [(limit[1] - limit[0]) * cur_best_entity.fitness / sum(fitness_list) for limit in self.limits_list]

            # Black hole attracts the other stars
            bh_x_list = cur_best_entity.x_list
            in_radius_star_index_list = []
            for i in range(1, self.entity_num):
                star = self.entity_list[i]
                s_x_list = star.x_list
                for j in range(len(self.limits_list)):
                    s_x_list[j] = s_x_list[j] + random.random() * (bh_x_list[j] - s_x_list[j])
                star.update()

                # Check whether this star in the radius of the black hole
                for a in range(len(self.limits_list)):
                    r = self.radius_list[a]
                    if abs(s_x_list[a] - bh_x_list[a]) < r:
                        in_radius_star_index_list.append(i)
                        break

            # Check whether any star is absorded by the black hole, if it does, add new star in next population
            if len(in_radius_star_index_list) > 0:
                for i in reversed(in_radius_star_index_list):
                    del self.entity_list[i]
                self.entity_list = self.entity_list + [self.Entity(self.exp_data_dict, self.fitness_function) for i in range(len(in_radius_star_index_list))]

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

def access_BH_EIS():
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
            goa = BH_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'bh_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('BH left: {0}'.format(900 - counter))
# access_BH_EIS()