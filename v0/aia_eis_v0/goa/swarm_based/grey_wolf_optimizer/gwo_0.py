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

# Grey wolf optimization
class GWO:
    """
    Refer:
        paper0: Grey Wolf Optimizer
    """
    class Wolf:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.position_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.position_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for i in range(len(self.limits_list)):
                if self.position_list[i] < self.limits_list[i][0]:
                    self.position_list[i] = self.limits_list[i][0]
                elif self.position_list[i] > self.limits_list[i][1]:
                    self.position_list[i] = self.limits_list[i][1]
            self.fitness = self.fitness_function(self.position_list)

    def __init__(self, iter_time, wolf_num, limits_list, fitness_function):
        self.iter_time = iter_time
        self.wolf_num = wolf_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.wolf_list = [self.Wolf(self.limits_list, self.fitness_function) for i in range(self.wolf_num)]

        # Initialize alpha (best minimum), beta (second minimum), and delta (third minimum)
        self.alpha_wolf = self.Wolf(self.limits_list, self.fitness_function)
        self.alpha_wolf.fitness = float('inf')
        self.beta_wolf = self.Wolf(self.limits_list, self.fitness_function)
        self.beta_wolf.fitness = float('inf')
        self.delta_wolf = self.Wolf(self.limits_list, self.fitness_function)
        self.delta_wolf.fitness = float('inf')

    def hunt(self):
        current_best_wolf_list = []
        for iter in range(self.iter_time):
            a = 2 - 2 * iter / self.iter_time

            # Assign/replace alpha (best minimum), beta (second minimum), and delta (third minimum) according to their fitness
            sorted_wolf_list = sorted(self.wolf_list, key=lambda wolf:wolf.fitness, reverse=False)[:5]
            for wolf in sorted_wolf_list:
                if wolf.fitness < self.alpha_wolf.fitness:
                    self.alpha_wolf = copy.deepcopy(wolf)
                    continue
                if (wolf.fitness > self.alpha_wolf.fitness) & (wolf.fitness < self.beta_wolf.fitness):
                    self.beta_wolf = copy.deepcopy(wolf)
                    continue
                if (wolf.fitness > self.beta_wolf.fitness) & (wolf.fitness < self.delta_wolf.fitness):
                    self.delta_wolf = copy.deepcopy(wolf)
            current_best_wolf_list.append(self.alpha_wolf)
            for wolf_index in range(self.wolf_num):
                for pos_index in range(len(self.limits_list)):
                    A_alpha = 2 * a * random.random() - a
                    C_alpha = 2 * random.random()
                    d_alpha = abs(C_alpha * self.alpha_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_alpha = self.alpha_wolf.position_list[pos_index] - A_alpha * d_alpha

                    A_beta = 2 * a * random.random() - a
                    C_beta = 2 * random.random()
                    d_beta = abs(C_beta * self.beta_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_beta = self.beta_wolf.position_list[pos_index] - A_beta * d_beta

                    A_delta = 2 * a * random.random() - a
                    C_delta = 2 * random.random()
                    d_delta = abs(C_delta * self.delta_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_delta = self.delta_wolf.position_list[pos_index] - A_delta * d_delta

                    self.wolf_list[wolf_index].position_list[pos_index] = (x_alpha + x_beta + x_delta) / 3
            for wolf_index in range(self.wolf_num):
                self.wolf_list[wolf_index].update()
        return current_best_wolf_list

# ---------------- Test GWO on f1 function (f1(x) = x**2) ----------------
# if __name__ == '__main__':
#     iter_time = 500
#     wolf_num = 30
#     dim = 2
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#     f1_fitness_function = f1
#
#     gwo = GWO(iter_time, wolf_num, limits_list=f1_limits_list, fitness_function=f1_fitness_function)
#     current_best_wolf_list = gwo.hunt()
#     global_best_wolf = gwo.alpha_wolf
#     print('Best wolf position:', global_best_wolf.position_list)
#     print('Fitness:', global_best_wolf.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_time)]
#     fitness_list = [wolf.fitness for wolf in current_best_wolf_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, fitness_list, label='Iteration {0}\nWolf number {1}\nDimension {2}'.format(iter_time, wolf_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Inflation rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()
# ---------------- Test GWO on f1 function (f1(x) = x**2) ----------------

# ---------------- Test GWO on f5 function ----------------
# if __name__ == '__main__':
#     iter_time = 5000
#     wolf_num = 30
#     dim = 30
#
#     f5_limits_list = [[-30, 30] for i in range(dim)]
#     from GA_pack.fittness_functions.f5 import f5
#     f5_fitness_function = f5
#
#     gwo = GWO(iter_time, wolf_num, limits_list = f5_limits_list, fitness_function = f5_fitness_function)
#     current_best_wolf_list = gwo.hunt()
#     global_best_wolf = gwo.alpha_wolf
#     print('Best wolf position:', global_best_wolf.position_list)
#     print('Fitness:', global_best_wolf.fitness)
#
#     # Draw the best universe in each iteration.
#     iter_list = [i for i in range(iter_time)]
#     fitness_list = [wolf.fitness for wolf in current_best_wolf_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, fitness_list, label='Iteration {0}\nWolf number {1}\nDimension {2}'.format(iter_time, wolf_num, dim))
#     line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Inflation rate')
#     plt.title('Search the minimum of f5')
#     plt.show()
# ---------------- Test GWO on f5 function ----------------

class GWO_EIS:
    class Wolf:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.fitness_function = fitness_function
            self.position_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for i in range(len(self.limits_list)):
                if self.position_list[i] < self.limits_list[i][0]:
                    self.position_list[i] = self.limits_list[i][0]
                elif self.position_list[i] > self.limits_list[i][1]:
                    self.position_list[i] = self.limits_list[i][1]
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, wolf_num, fitness_function=cal_EIS_WSE_fitness_1):
        """
        :param
            iter_time:
                人为设定的算法最大迭代次数
            exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试/模拟频率              'f',    fre_list
                    测试/模拟所得复数阻抗       'z_sim' or 'z_raw',     z_simulated/raw_complex_list
        :return:
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.wolf_num = wolf_num

        self.fitness_function = fitness_function

        self.wolf_list = [self.Wolf(self.exp_data_dict, self.fitness_function) for i in range(self.wolf_num)]

        # Initialize alpha (best minimum), beta (second minimum), and delta (third minimum)
        self.alpha_wolf = self.Wolf(self.exp_data_dict, self.fitness_function)
        self.alpha_wolf.fitness = float('inf')
        self.beta_wolf = self.Wolf(self.exp_data_dict, self.fitness_function)
        self.beta_wolf.fitness = float('inf')
        self.delta_wolf = self.Wolf(self.exp_data_dict, self.fitness_function)
        self.delta_wolf.fitness = float('inf')

    def hunt(self):
        current_best_wolf_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            a = 2 - 2 * iter / self.iter_time

            # Assign/replace alpha (best minimum), beta (second minimum), and delta (third minimum) according to their fitness
            sorted_wolf_list = sorted(self.wolf_list, key=lambda wolf:wolf.fitness, reverse=False)[:5]
            for wolf in sorted_wolf_list:
                if wolf.fitness < self.alpha_wolf.fitness:
                    self.alpha_wolf = copy.deepcopy(wolf)
                    continue
                if (wolf.fitness > self.alpha_wolf.fitness) & (wolf.fitness < self.beta_wolf.fitness):
                    self.beta_wolf = copy.deepcopy(wolf)
                    continue
                if (wolf.fitness > self.beta_wolf.fitness) & (wolf.fitness < self.delta_wolf.fitness):
                    self.delta_wolf = copy.deepcopy(wolf)
            current_best_wolf_list.append(self.alpha_wolf)
            for wolf_index in range(self.wolf_num):
                for pos_index in range(len(self.limits_list)):
                    A_alpha = 2 * a * random.random() - a
                    C_alpha = 2 * random.random()
                    d_alpha = abs(C_alpha * self.alpha_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_alpha = self.alpha_wolf.position_list[pos_index] - A_alpha * d_alpha

                    A_beta = 2 * a * random.random() - a
                    C_beta = 2 * random.random()
                    d_beta = abs(C_beta * self.beta_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_beta = self.beta_wolf.position_list[pos_index] - A_beta * d_beta

                    A_delta = 2 * a * random.random() - a
                    C_delta = 2 * random.random()
                    d_delta = abs(C_delta * self.delta_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_delta = self.delta_wolf.position_list[pos_index] - A_delta * d_delta

                    self.wolf_list[wolf_index].position_list[pos_index] = (x_alpha + x_beta + x_delta) / 3
            for wolf_index in range(self.wolf_num):
                self.wolf_list[wolf_index].update()
            # There are two entitoes only after at least two iteration
            if iter >= 1:
                x_lists_list = [copy.deepcopy(current_best_wolf_list[-2].position_list),\
                                copy.deepcopy(current_best_wolf_list[-1].position_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,\
                                                                max_iter_time=self.iter_time, data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return current_best_wolf_list, iter, chi_squared

class GWO_EIS_access:
    """
    Function:
        This class is created for accessing the accuracy, reliability and stability of GWO
    """
    class Wolf:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.fitness_function = fitness_function
            self.position_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.position_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for i in range(len(self.limits_list)):
                if self.position_list[i] < self.limits_list[i][0]:
                    self.position_list[i] = self.limits_list[i][0]
                elif self.position_list[i] > self.limits_list[i][1]:
                    self.position_list[i] = self.limits_list[i][1]
            self.fitness = self.fitness_function(self.exp_data_dict, self.position_list)

    def __init__(self, exp_data_dict, iter_time, wolf_num, fitness_function=cal_EIS_WSE_fitness_1):
        """
        :param
            iter_time:
                人为设定的算法最大迭代次数
            exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试/模拟频率              'f',    fre_list
                    测试/模拟所得复数阻抗       'z_sim' or 'z_raw',     z_simulated/raw_complex_list
        :return:
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_time = iter_time
        self.wolf_num = wolf_num

        self.fitness_function = fitness_function

        self.wolf_list = [self.Wolf(self.exp_data_dict, self.fitness_function) for i in range(self.wolf_num)]

        # Initialize alpha (best minimum), beta (second minimum), and delta (third minimum)
        self.alpha_wolf = self.Wolf(self.exp_data_dict, self.fitness_function)
        self.alpha_wolf.fitness = float('inf')
        self.beta_wolf = self.Wolf(self.exp_data_dict, self.fitness_function)
        self.beta_wolf.fitness = float('inf')
        self.delta_wolf = self.Wolf(self.exp_data_dict, self.fitness_function)
        self.delta_wolf.fitness = float('inf')

    def search(self, res_fn, start_time):
        current_best_wolf_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            a = 2 - 2 * iter / self.iter_time

            # Assign/replace alpha (best minimum), beta (second minimum), and delta (third minimum) according to their fitness
            sorted_wolf_list = sorted(self.wolf_list, key=lambda wolf:wolf.fitness, reverse=False)[:5]
            for wolf in sorted_wolf_list:
                if wolf.fitness < self.alpha_wolf.fitness:
                    self.alpha_wolf = copy.deepcopy(wolf)
                    continue
                if (wolf.fitness > self.alpha_wolf.fitness) & (wolf.fitness < self.beta_wolf.fitness):
                    self.beta_wolf = copy.deepcopy(wolf)
                    continue
                if (wolf.fitness > self.beta_wolf.fitness) & (wolf.fitness < self.delta_wolf.fitness):
                    self.delta_wolf = copy.deepcopy(wolf)
            current_best_wolf_list.append(self.alpha_wolf)
            for wolf_index in range(self.wolf_num):
                for pos_index in range(len(self.limits_list)):
                    A_alpha = 2 * a * random.random() - a
                    C_alpha = 2 * random.random()
                    d_alpha = abs(C_alpha * self.alpha_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_alpha = self.alpha_wolf.position_list[pos_index] - A_alpha * d_alpha

                    A_beta = 2 * a * random.random() - a
                    C_beta = 2 * random.random()
                    d_beta = abs(C_beta * self.beta_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_beta = self.beta_wolf.position_list[pos_index] - A_beta * d_beta

                    A_delta = 2 * a * random.random() - a
                    C_delta = 2 * random.random()
                    d_delta = abs(C_delta * self.delta_wolf.position_list[pos_index] - self.wolf_list[wolf_index].position_list[pos_index])
                    x_delta = self.delta_wolf.position_list[pos_index] - A_delta * d_delta

                    self.wolf_list[wolf_index].position_list[pos_index] = (x_alpha + x_beta + x_delta) / 3
            for wolf_index in range(self.wolf_num):
                self.wolf_list[wolf_index].update()

            # There are two entities only after at least two iteration
            if iter >= 1:
                x_lists_list = [copy.deepcopy(current_best_wolf_list[-2].position_list),\
                                copy.deepcopy(current_best_wolf_list[-1].position_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list = x_lists_list, iter=iter,\
                                                                max_iter_time = self.iter_time,\
                                                                data_dict = self.exp_data_dict,\
                                                                CS_limit = 1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',['\
                           + ','.join([str(para) for para in current_best_wolf_list[-1].position_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_GWO_EIS():
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
            goa = GWO_EIS_access(exp_data_dict=sim_ecm, iter_time=10000, wolf_num=10*para_num)
            res_fn = 'gwo_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('GWO left: {0}'.format(900 - counter))
# access_GWO_EIS()