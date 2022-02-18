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

class HS:
    """
    Refer:
        paper:
            paper1- A New Heuristic Optimization Algorithm: Harmony Search
            paper2- 和声搜索算法的搜索机制研究及其应用
                chapter 2 和声搜索算法及其相关算法介绍
                    2.1 和声搜索算法
                        2.1.1 和声搜索算法的基本原理
                            表2-1 基本HS算法的步骤
    Version
        First
    Adjustable parameters:
        hmcr, Harmony Memory Changing Rate, (0 ~ 1)
        par, Pitch Adjusting Rate, (0 ~ 1)
    """
    class Harmony:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    x = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
                    self.x_list[i] = x
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, harmony_num, limits_list, fitness_function, hmcr=0.5, par=0.1):
        self.iter_num = iter_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Unique parameters of HS
        # hms = Harmony memory size 记忆库大小 is equal to the number of all the harmony
        self.harmony_num = harmony_num
        # Harmony memory considering rate 记忆库选中概率
        self.hmcr = hmcr
        # Pitch adjusting rate 音调调节概率
        self.par = par

        self.harmony_list = [self.Harmony(limits_list, fitness_function) for i in range(harmony_num)]
        self.global_best_harmony = self.Harmony(limits_list, fitness_function)

    def search(self):
        cur_best_harmony_list = []
        global_best_harmony_list = []

        for iter_index in range(self.iter_num):
            sorted_harmony_list = sorted(self.harmony_list, key=lambda harmony: harmony.fitness, reverse=False)

            cur_best_harmony = sorted_harmony_list[0]
            if cur_best_harmony.fitness < self.global_best_harmony.fitness:
                self.global_best_harmony = copy.deepcopy(cur_best_harmony)
            cur_best_harmony_list.append(copy.deepcopy(cur_best_harmony))
            global_best_harmony_list.append(copy.deepcopy(self.global_best_harmony))

            for har_i in range(self.harmony_num):
                x_list_pack = [copy.deepcopy(har.x_list) for har in sorted_harmony_list]
                tmp_x_list = []
                for dim_i in range(len(self.limits_list)):
                    if random.random() < self.hmcr:
                        # random.sample returns a list
                        x = random.sample([x_list[dim_i] for x_list in x_list_pack], 1)[0]
                        # Add a disturbance to x by probability PAR, I use a uniform distribution in stead of the discrete choices
                        if random.random() < self.par:
                            dim_range = self.limits_list[dim_i][1] - self.limits_list[dim_i][0]
                            x = random.uniform(x - dim_range * self.par / 2, x + dim_range * self.par / 2)
                    else:
                        x = random.uniform(self.limits_list[dim_i][0], self.limits_list[dim_i][1])
                    tmp_x_list.append(x)
                tmp_harmony = self.Harmony(self.limits_list, self.fitness_function)
                tmp_harmony.x_list = tmp_x_list
                tmp_harmony.update()

                # Compare with the worst harmony in the harmony_list
                # tmp_sorted_harmony_list = sorted(self.harmony_list, key=lambda harmony: harmony.fitness, reverse=False)
                # if tmp_harmony.fitness < tmp_sorted_harmony_list[-1].fitness:
                #     tmp_sorted_harmony_list[-1] = tmp_harmony
                # self.harmony_list = copy.deepcopy(tmp_sorted_harmony_list)

                if tmp_harmony.fitness < sorted_harmony_list[-1].fitness:
                    sorted_harmony_list[-1] = tmp_harmony
                    sorted_harmony_list.sort(key=lambda harmony: harmony.fitness, reverse=False)
            self.harmony_list = copy.deepcopy(sorted_harmony_list)
        return cur_best_harmony_list, global_best_harmony_list

# if __name__ == '__main__':
#     iter_num = 1000
#     harmony_num = 20
#     dim = 7
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     hs = HS(iter_num, harmony_num, f1_limits_list, f1_fitness_function)
#     cur_best_harmony_list, global_best_harmony_list = hs.search()
#     print('Best entity position:', hs.global_best_harmony.x_list)
#     print('Fitness:', hs.global_best_harmony.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_harmony_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_harmony_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, harmony_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, harmony_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class HS_EIS:
    """
    Refer:
        paper:
            paper1- A New Heuristic Optimization Algorithm: Harmony Search
            paper2- 和声搜索算法的搜索机制研究及其应用
                chapter 2 和声搜索算法及其相关算法介绍
                    2.1 和声搜索算法
                        2.1.1 和声搜索算法的基本原理
                            表2-1 基本HS算法的步骤
    Version
        First
    Adjustable parameters:
        hmcr, Harmony Memory Changing Rate, (0 ~ 1)
        par, Pitch Adjusting Rate, (0 ~ 1)
    """
    class Harmony:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    x = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
                    self.x_list[i] = x
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, harmony_num, hmcr=0.5, par=0.1, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.fitness_function = fitness_function

        # Unique parameters of HS
        # hms = Harmony memory size 记忆库大小 is equal to the number of all the harmony
        self.harmony_num = harmony_num
        # Harmony memory considering rate 记忆库选中概率
        self.hmcr = hmcr
        # Pitch adjusting rate 音调调节概率
        self.par = par

        self.harmony_list = [self.Harmony(self.exp_data_dict, fitness_function) for i in range(harmony_num)]
        self.global_best_harmony = self.Harmony(self.exp_data_dict, fitness_function)

    def search(self):
        cur_best_harmony_list = []
        global_best_harmony_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            sorted_harmony_list = sorted(self.harmony_list, key=lambda harmony: harmony.fitness, reverse=False)

            cur_best_harmony = sorted_harmony_list[0]
            if cur_best_harmony.fitness < self.global_best_harmony.fitness:
                self.global_best_harmony = copy.deepcopy(cur_best_harmony)
            cur_best_harmony_list.append(copy.deepcopy(cur_best_harmony))
            global_best_harmony_list.append(copy.deepcopy(self.global_best_harmony))

            for har_i in range(self.harmony_num):
                x_list_pack = [copy.deepcopy(har.x_list) for har in sorted_harmony_list]
                tmp_x_list = []
                for dim_i in range(len(self.limits_list)):
                    if random.random() < self.hmcr:
                        # random.sample returns a list
                        x = random.sample([x_list[dim_i] for x_list in x_list_pack], 1)[0]
                        # Add a disturbance to x by probability PAR, I use a uniform distribution in stead of the discrete choices
                        if random.random() < self.par:
                            dim_range = self.limits_list[dim_i][1] - self.limits_list[dim_i][0]
                            x = random.uniform(x - dim_range * self.par / 2, x + dim_range * self.par / 2)
                    else:
                        x = random.uniform(self.limits_list[dim_i][0], self.limits_list[dim_i][1])
                    tmp_x_list.append(x)
                tmp_harmony = self.Harmony(self.exp_data_dict, self.fitness_function)
                tmp_harmony.x_list = tmp_x_list
                tmp_harmony.update()

                if tmp_harmony.fitness < sorted_harmony_list[-1].fitness:
                    sorted_harmony_list[-1] = tmp_harmony
                    sorted_harmony_list.sort(key=lambda harmony: harmony.fitness, reverse=False)
            self.harmony_list = copy.deepcopy(sorted_harmony_list)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_harmony_list[-2].x_list, global_best_harmony_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return cur_best_harmony_list, global_best_harmony_list, iter, chi_squared

class HS_EIS_access:
    """
    Refer:
        paper:
            paper1- A New Heuristic Optimization Algorithm: Harmony Search
            paper2- 和声搜索算法的搜索机制研究及其应用
                chapter 2 和声搜索算法及其相关算法介绍
                    2.1 和声搜索算法
                        2.1.1 和声搜索算法的基本原理
                            表2-1 基本HS算法的步骤
    Version
        First
    Adjustable parameters:
        hmcr, Harmony Memory Changing Rate, (0 ~ 1)
        par, Pitch Adjusting Rate, (0 ~ 1)
    """
    class Harmony:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    x = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
                    self.x_list[i] = x
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, harmony_num, hmcr=0.5, par=0.1, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.fitness_function = fitness_function

        # Unique parameters of HS
        # hms = Harmony memory size 记忆库大小 is equal to the number of all the harmony
        self.harmony_num = harmony_num
        # Harmony memory considering rate 记忆库选中概率
        self.hmcr = hmcr
        # Pitch adjusting rate 音调调节概率
        self.par = par

        self.harmony_list = [self.Harmony(self.exp_data_dict, fitness_function) for i in range(harmony_num)]
        self.global_best_harmony = self.Harmony(self.exp_data_dict, fitness_function)

    def search(self, res_fn, start_time):
        cur_best_harmony_list = []
        global_best_harmony_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            sorted_harmony_list = sorted(self.harmony_list, key=lambda harmony: harmony.fitness, reverse=False)

            cur_best_harmony = sorted_harmony_list[0]
            if cur_best_harmony.fitness < self.global_best_harmony.fitness:
                self.global_best_harmony = copy.deepcopy(cur_best_harmony)
            cur_best_harmony_list.append(copy.deepcopy(cur_best_harmony))
            global_best_harmony_list.append(copy.deepcopy(self.global_best_harmony))

            for har_i in range(self.harmony_num):
                x_list_pack = [copy.deepcopy(har.x_list) for har in sorted_harmony_list]
                tmp_x_list = []
                for dim_i in range(len(self.limits_list)):
                    if random.random() < self.hmcr:
                        # random.sample returns a list
                        x = random.sample([x_list[dim_i] for x_list in x_list_pack], 1)[0]
                        # Add a disturbance to x by probability PAR, I use a uniform distribution in stead of the discrete choices
                        if random.random() < self.par:
                            dim_range = self.limits_list[dim_i][1] - self.limits_list[dim_i][0]
                            x = random.uniform(x - dim_range * self.par / 2, x + dim_range * self.par / 2)
                    else:
                        x = random.uniform(self.limits_list[dim_i][0], self.limits_list[dim_i][1])
                    tmp_x_list.append(x)
                tmp_harmony = self.Harmony(self.exp_data_dict, self.fitness_function)
                tmp_harmony.x_list = tmp_x_list
                tmp_harmony.update()

                if tmp_harmony.fitness < sorted_harmony_list[-1].fitness:
                    sorted_harmony_list[-1] = tmp_harmony
                    sorted_harmony_list.sort(key=lambda harmony: harmony.fitness, reverse=False)
            self.harmony_list = copy.deepcopy(sorted_harmony_list)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_harmony_list[-2].x_list, global_best_harmony_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter, \
                                                                max_iter_time=self.iter_num, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_harmony_list[-1].x_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_HS_EIS():
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
            goa = HS_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, harmony_num=10*para_num)
            res_fn = 'hs_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('HS left: {0}'.format(900 - counter))
# access_HS_EIS()