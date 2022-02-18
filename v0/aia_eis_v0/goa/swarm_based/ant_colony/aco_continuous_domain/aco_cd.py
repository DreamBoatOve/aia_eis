import copy
import random
import math

from time import perf_counter
import os
import sys
sys.path.append('../../../../')
from utils.file_utils.filename_utils import get_ecm_num_str, get_Num_len
from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict

from goa.GOA_criterions import goa_criterion_pack
from GA_pack.fittness_functions.eis_fitness import cal_EIS_WSE_fitness_1

class aco_co_do:
    """
    Refer
        paper0: Ant colony optimization in continuous domain
    """
    class Ant:
        def __init__(self, limits_list):
            self.limits_list = limits_list
            self.s_list = self.initialize()

        def initialize(self):
            s_list = []
            for limit in self.limits_list:
                minimum = limit[0]
                maximum = limit[1]
                s_list.append(random.uniform(minimum, maximum))
            return s_list

    def __init__(self, limits_list, m, iter_time, q=0.0001, epsilon=0.85):
        """
        :param
            limits_list:
                record the minimum and maximum of each variable
            m:
                the number of ants created in each iteration
            k:
                the number of ants stored in the archive
            q:
                q*k = stdandard deviation σ, 0.0001 is taken as default value from paper 'Ant colony optimization for continuous domains'
            iter_time:
            epsilon:
                0.85 is taken as default value from paper 'Ant colony optimization for continuous domains'
        """
        self.limits_list = limits_list
        self.m = m
        self.k = len(limits_list)
        self.iter_time = iter_time
        self.q = q
        self.epsilon = epsilon

        # Initialize the ant colony (number K in the Archive) and calculate their fitness according to the given 'FITNESS FUNCTION'
        self.t_ants_list = [self.Ant(self.limits_list) for i in range(self.k)]
        self.t_fitness_list = self.cal_fitness(ants_list=self.t_ants_list, fitness_type='rosenbrock')
        self.t_rank_list, self.t_weight_list = self.cal_weight(self.t_fitness_list)

    def cal_fitness(self, ants_list, fitness_type='rosenbrock'):
        fitness_list = []
        if fitness_type == 'rosenbrock':
            from GA_pack.fittness_functions.rosenbrock_fitness import rosenbrock
            for ant in ants_list:
                x1, x2 = ant.s_list
                fitness_list.append(rosenbrock(x1, x2).get_rosenbrock_fitness())
        return fitness_list

    def cal_weight(self, fitness_list):
        # index might be 0, so add 1
        # Note: sorted in ascending order (reverse=False, default) [1,2,3,...], should be modified to descending order (reverse=True) [... , 3,2,1]
        rank_list = [index + 1 for index, value in sorted(list(enumerate(fitness_list)), key=lambda x : x[1], reverse=True)]
        weight_list = []
        for rank in rank_list:
            w = (1 / (self.q * self.k * math.sqrt(2 * math.pi))) * (math.e ** (-(pow(rank - 1, 2)) / (2 * pow(self.q*self.k, 2))))
            weight_list.append(w)
        return rank_list, weight_list

    def pdf_generator(self, s_index):
        # Use roulette wheel to select an ant in the archive
        wheel = sum(self.t_weight_list)
        wheel_pointer = random.uniform(0, wheel)
        current_wheel_pointer = 0.0
        selected_index = None
        for index, weight in enumerate(self.t_weight_list):
            current_wheel_pointer += weight
            if current_wheel_pointer > wheel_pointer:
                selected_index = index

        # Build the Gaussian function
        mu = self.t_ants_list[selected_index].s_list[s_index]
        sigma = self.epsilon * (sum([ant.s_list[s_index] for ant in self.t_ants_list]) - mu) / (self.k - 1)
        s_value = float('inf')
        while (s_value <= self.limits_list[s_index][0]) | (s_value >= self.limits_list[s_index][1]):
            s_value = random.gauss(mu, sigma)
        return s_value

    def update_archive(self, ants_list, fitness_list):
        ants_fitness_list = [(ant, fitness) for ant, fitness in sorted(zip(ants_list, fitness_list), key=lambda x: x[1], reverse=True)]
        self.t_ants_list = [ant_fitness[0] for ant_fitness in ants_fitness_list]
        self.t_fitness_list = [ant_fitness[1] for ant_fitness in ants_fitness_list]
        self.t_rank_list, self.t_weight_list = self.cal_weight(self.t_fitness_list)

    def iteration(self):
        for iter in range(self.iter_time):
            tmp_ants_list = []
            for a in range(self.m):
                ant = self.Ant(self.limits_list)
                s_list = []
                for s_index in range(len(self.limits_list)):
                    s_value = self.pdf_generator(s_index)
                    s_list.append(s_value)
                ant.s_list = s_list
                tmp_ants_list.append(ant)
            tmp_fitness_list = self.cal_fitness(tmp_ants_list)

            # Update the ants in the archive according to their rank
            self.update_archive(ants_list=self.t_ants_list + tmp_ants_list, fitness_list=self.t_fitness_list + tmp_fitness_list)

        # After the iteration, select the ant with the highest fitness from the archive
        best_ant, highest_fitness = sorted(zip(self.t_ants_list, self.t_fitness_list), key=lambda x : x[1], reverse=True)[0]
        return best_ant, highest_fitness

# if __name__ == '__main__':
    # Test with the 'ROSENBROCK FUNCTION'
    # rosenbrock_limits_list = [[-2.048, 2.048],[-2.048, 2.048]]
    # acs_cd = aco_co_do(limits_list=rosenbrock_limits_list, m=200, iter_time=50)
    # best_ant, highest_fitness = acs_cd.iteration()
    # print('Best ant:', best_ant.s_list)
    # print('Highest fitness:', highest_fitness)
    """
    R(RC)_IS_lin-kk_res.txt
        m = 50; iter_time = 50
            Best ant: [2.0072932014469274, -2.0397597012679958]
            Highest fitness: 3684.2733796560315
        m = 50; iter_time = 100
            Best ant: [2.0342360590903366, -2.0005746428561917]
            Highest fitness: 3769.422347563795
        m = 50; iter_time = 500
            Best ant: [2.0402568215296424, -2.0371961701065997]
            Highest fitness: 3844.8887808882205
        m = 100; iter_time = 50
            Best ant: [-2.0247131223978547, -2.023590690219659]
            Highest fitness: 3758.32781802765
        m = 200; iter_time = 50
            Best ant: [2.0355266672204557, -2.013497491655291]
            Highest fitness: 3791.7725847761553
    """

def ACO_CD_EIS_fitness_function(exp_data_dict, ECM_paras_list):
    """
    Function:
        The original ACO finds the MAXIMUM in search space. Now I use ACO to search MINIMUM fitness. So I have to modify
        my EIS_fitness_function (cal_EIS_WSE_fitness_1, the smaller, the better) as its inversely proportional version:
        = 1 / cal_EIS_WSE_fitness_1
        Be attention to the situation when cal_EIS_WSE_fitness_1 returns 0.0
    """
    eis_fitness = cal_EIS_WSE_fitness_1(exp_data_dict, ECM_paras_list)
    if eis_fitness < 1e-25:
        eis_fitness = 1e-25
    ant_fitness = 1.0 / eis_fitness
    return ant_fitness

class ACO_CD_EIS:
    """
    Ant colony optimization in continuous domain
    Fitting EIS
    """
    class Ant:
        def __init__(self, limits_list):
            self.limits_list = limits_list
            # s_list is the same as position_list in other GOAs
            self.s_list = self.initialize()

        def initialize(self):
            s_list = []
            for limit in self.limits_list:
                minimum = limit[0]
                maximum = limit[1]
                s_list.append(random.uniform(minimum, maximum))
            return s_list

    def __init__(self, exp_data_dict, m, iter_time, q=0.0001, epsilon=0.85, fitness_function=ACO_CD_EIS_fitness_function):
        """
        :param
            exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试/模拟频率              'f',    fre_list
                    测试/模拟所得复数阻抗       'z_sim' or 'z_raw',     z_simulated/raw_complex_list
            iter_time:
                人为设定的算法最大迭代次数
            m:
                the number of ants created in each iteration
            k:
                the number of ants stored in the archive
            q:
                q * k = standard deviation σ, 0.0001 is taken as default value from paper 'Ant colony optimization for continuous domains'
            epsilon:
                0.85 is taken as default value from paper 'Ant colony optimization for continuous domains'
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']
        self.fitness_function = fitness_function

        self.m = m
        self.k = len(self.limits_list)
        self.iter_time = iter_time
        self.q = q
        self.epsilon = epsilon

        # Initialize the ant colony (number K in the Archive) and calculate their fitness according to the given 'FITNESS FUNCTION'
        self.t_ants_list = [self.Ant(self.limits_list) for i in range(self.k)]

        self.t_fitness_list = [self.fitness_function(self.exp_data_dict, ant.s_list) for ant in self.t_ants_list]

        self.t_rank_list, self.t_weight_list = self.cal_weight(self.t_fitness_list)

    def cal_weight(self, fitness_list):
        # Note: sorted in ascending order (reverse=False, default) [1, 2, 3,...],
        # should be modified to descending order (reverse=True) [... , 3, 2, 1]
        """
        "index + 1" means what?
            In the below calculation of weight of each Ant, it involves a step "power(rank(index) - 1, 2)". To make the step
            mathematically meaningful, rank(or index) is at least as 1
        """
        rank_list = [index + 1 for index, value in sorted(list(enumerate(fitness_list)), key=lambda x : x[1], reverse=True)]
        weight_list = []
        for rank in rank_list:
            w = (1 / (self.q * self.k * math.sqrt(2 * math.pi))) * (math.e ** (-(pow(rank - 1, 2)) / (2 * pow(self.q * self.k, 2))))
            weight_list.append(w)
        return rank_list, weight_list

    def pdf_generator(self, s_index):
        # Use roulette wheel to select an ant in the archive
        wheel = sum(self.t_weight_list)
        wheel_pointer = random.uniform(0, wheel)
        current_wheel_pointer = 0.0
        selected_index = None
        for index, weight in enumerate(self.t_weight_list):
            current_wheel_pointer += weight
            if current_wheel_pointer > wheel_pointer:
                selected_index = index

        # Build the Gaussian function
        mu = self.t_ants_list[selected_index].s_list[s_index]
        sigma = self.epsilon * (sum([ant.s_list[s_index] for ant in self.t_ants_list]) - mu) / (self.k - 1)
        # s_value, like a x in position_list=[x0, x1, x2, ...]
        s_value = float('inf')
        # Constrain the range of s_value(or x)
        while (s_value <= self.limits_list[s_index][0]) | (s_value >= self.limits_list[s_index][1]):
            s_value = random.gauss(mu, sigma)
        return s_value

    def update_archive(self, ants_list, fitness_list):
        ants_fitness_list = [(ant, fitness) for ant, fitness in sorted(zip(ants_list, fitness_list), key=lambda x: x[1], reverse=True)]
        self.t_ants_list = [ant_fitness[0] for ant_fitness in ants_fitness_list]
        self.t_fitness_list = [ant_fitness[1] for ant_fitness in ants_fitness_list]
        self.t_rank_list, self.t_weight_list = self.cal_weight(self.t_fitness_list)

    # def iteration(self):
    def search(self):
        global_best_ant_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            tmp_ants_list = []
            for a in range(self.m):
                ant = self.Ant(self.limits_list)
                s_list = []
                for s_index in range(len(self.limits_list)):
                    s_value = self.pdf_generator(s_index)
                    s_list.append(s_value)
                ant.s_list = s_list
                tmp_ants_list.append(ant)
            tmp_fitness_list = [self.fitness_function(self.exp_data_dict, ant.s_list) for ant in tmp_ants_list]

            # Update the ants in the archive according to their rank
            self.update_archive(ants_list=self.t_ants_list + tmp_ants_list, fitness_list=self.t_fitness_list + tmp_fitness_list)
            # After one iteration, select the ant with the highest fitness from the archive
            tmp_best_ant, tmp_highest_fitness = sorted(zip(self.t_ants_list, self.t_fitness_list), key=lambda x: x[1], reverse=True)[0]
            global_best_ant_list.append(copy.deepcopy(tmp_best_ant))

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [copy.deepcopy(global_best_ant_list[-2].s_list),
                                copy.deepcopy(global_best_ant_list[-1].s_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_time,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        # After the iteration, select the ant with the highest fitness from the archive
        # best_ant, highest_fitness = sorted(zip(self.t_ants_list, self.t_fitness_list), key=lambda x : x[1], reverse=True)[0]
        return global_best_ant_list, iter, chi_squared

class ACO_CD_EIS_access:
    """
    Ant colony optimization in continuous domain
    Fitting EIS
    """
    class Ant:
        def __init__(self, limits_list):
            self.limits_list = limits_list
            # s_list is the same as position_list in other GOAs
            self.s_list = self.initialize()

        def initialize(self):
            s_list = []
            for limit in self.limits_list:
                minimum = limit[0]
                maximum = limit[1]
                s_list.append(random.uniform(minimum, maximum))
            return s_list

    def __init__(self, exp_data_dict, m, iter_time, q=0.0001, epsilon=0.85, fitness_function=ACO_CD_EIS_fitness_function):
        """
        :param
            exp_data_dict:
                包含EIS参数拟合的标准信息
                    ECM型号               'ecm_num',  ecm_num
                    各元件的参数取值范围    'limit',    limits_list
                    测试/模拟频率              'f',    fre_list
                    测试/模拟所得复数阻抗       'z_sim' or 'z_raw',     z_simulated/raw_complex_list
            iter_time:
                人为设定的算法最大迭代次数
            m:
                the number of ants created in each iteration
            k:
                the number of ants stored in the archive
            q:
                q * k = standard deviation σ, 0.0001 is taken as default value from paper 'Ant colony optimization for continuous domains'
            epsilon:
                0.85 is taken as default value from paper 'Ant colony optimization for continuous domains'
        """
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']
        self.fitness_function = fitness_function

        self.m = m
        self.k = len(self.limits_list)
        self.iter_time = iter_time
        self.q = q
        self.epsilon = epsilon

        # Initialize the ant colony (number K in the Archive) and calculate their fitness according to the given 'FITNESS FUNCTION'
        self.t_ants_list = [self.Ant(self.limits_list) for i in range(self.k)]

        self.t_fitness_list = [self.fitness_function(self.exp_data_dict, ant.s_list) for ant in self.t_ants_list]

        self.t_rank_list, self.t_weight_list = self.cal_weight(self.t_fitness_list)

    def cal_weight(self, fitness_list):
        # Note: sorted in ascending order (reverse=False, default) [1, 2, 3,...],
        # should be modified to descending order (reverse=True) [... , 3, 2, 1]
        """
        "index + 1" means what?
            In the below calculation of weight of each Ant, it involves a step "power(rank(index) - 1, 2)". To make the step
            mathematically meaningful, rank(or index) is at least as 1
        """
        rank_list = [index + 1 for index, value in sorted(list(enumerate(fitness_list)), key=lambda x : x[1], reverse=True)]
        weight_list = []
        for rank in rank_list:
            w = (1 / (self.q * self.k * math.sqrt(2 * math.pi))) * (math.e ** (-(pow(rank - 1, 2)) / (2 * pow(self.q * self.k, 2))))
            weight_list.append(w)
        return rank_list, weight_list

    def pdf_generator(self, s_index):
        # Use roulette wheel to select an ant in the archive
        wheel = sum(self.t_weight_list)
        wheel_pointer = random.uniform(0, wheel)
        current_wheel_pointer = 0.0
        selected_index = None
        for index, weight in enumerate(self.t_weight_list):
            current_wheel_pointer += weight
            if current_wheel_pointer > wheel_pointer:
                selected_index = index

        # Build the Gaussian function
        mu = self.t_ants_list[selected_index].s_list[s_index]
        sigma = self.epsilon * (sum([ant.s_list[s_index] for ant in self.t_ants_list]) - mu) / (self.k - 1)
        # s_value, like a x in position_list=[x0, x1, x2, ...]
        s_value = float('inf')
        # Constrain the range of s_value(or x)
        while (s_value <= self.limits_list[s_index][0]) | (s_value >= self.limits_list[s_index][1]):
            s_value = random.gauss(mu, sigma)
        return s_value

    def update_archive(self, ants_list, fitness_list):
        ants_fitness_list = [(ant, fitness) for ant, fitness in sorted(zip(ants_list, fitness_list), key=lambda x: x[1], reverse=True)]
        self.t_ants_list = [ant_fitness[0] for ant_fitness in ants_fitness_list]
        self.t_fitness_list = [ant_fitness[1] for ant_fitness in ants_fitness_list]
        self.t_rank_list, self.t_weight_list = self.cal_weight(self.t_fitness_list)

    def search(self, res_fn, start_time):
        global_best_ant_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            tmp_ants_list = []
            for a in range(self.m):
                ant = self.Ant(self.limits_list)
                s_list = []
                for s_index in range(len(self.limits_list)):
                    s_value = self.pdf_generator(s_index)
                    s_list.append(s_value)
                ant.s_list = s_list
                tmp_ants_list.append(ant)
            tmp_fitness_list = [self.fitness_function(self.exp_data_dict, ant.s_list) for ant in tmp_ants_list]

            # Update the ants in the archive according to their rank
            self.update_archive(ants_list=self.t_ants_list + tmp_ants_list, fitness_list=self.t_fitness_list + tmp_fitness_list)
            # After one iteration, select the ant with the highest fitness from the archive
            tmp_best_ant, tmp_highest_fitness = sorted(zip(self.t_ants_list, self.t_fitness_list), key=lambda x: x[1], reverse=True)[0]
            global_best_ant_list.append(copy.deepcopy(tmp_best_ant))

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [copy.deepcopy(global_best_ant_list[-2].s_list),
                                copy.deepcopy(global_best_ant_list[-1].s_list)]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list,\
                                                                iter=iter, \
                                                                max_iter_time=self.iter_time, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',['\
                           + ','.join([str(para) for para in global_best_ant_list[-1].s_list]) \
                           + '],' + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)
                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_ACO_CD_EIS(ecm_num, range_pair):
    # counter = 0
    # # Iterate on 9 ECMs
    # for i in range(1, 10):
    #     ecm_sim_folder = '../../../../datasets/goa_datasets/simulated'
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
    #         goa = ACO_CD_EIS_access(exp_data_dict=sim_ecm, m=10*para_num, iter_time=10000)
    #         res_fn = 'aco_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
    #         # ------------------------------  Change GOA name ------------------------------
    #         goa.search(res_fn, start_time=t_start)
    #
    #         counter += 1
    #         print('ACO left: {0}'.format(900 - counter))

    print('ACO starts to fit ECM{0}'.format(ecm_num))
    counter = 0
    ecm_sim_folder = '../../../../datasets/goa_datasets/simulated'
    ecm_num_str = get_ecm_num_str(ecm_num)
    file_path = os.path.join(ecm_sim_folder, 'ecm_' + ecm_num_str)
    sim_ecm = load_sim_ecm_para_config_dict(ecm_num, file_path)
    para_num = len(sim_ecm['para'])

    # Iterate for 100 times
    # for j in range(100):
    for j in range(range_pair[0], range_pair[1]):
        t_start = perf_counter()
        # ------------------------------  Change GOA name ------------------------------
        goa = ACO_CD_EIS_access(exp_data_dict=sim_ecm, m=10*para_num, iter_time=2000)
        res_fn = 'aco_ecm{0}_'.format(ecm_num) + get_Num_len(num=j, length=2) + '.txt'
        # ------------------------------  Change GOA name ------------------------------
        goa.search(res_fn, start_time=t_start)

        counter += 1
        print('ACO on ECM{0}-{1}~{2} left: {3}'.format(ecm_num, range_pair[0], range_pair[1], len(range(range_pair[0], range_pair[1])) - counter))

# ECM-4
# access_ACO_CD_EIS(ecm_num=4, range_pair=[0, 20])
# access_ACO_CD_EIS(ecm_num=4, range_pair=[20, 40])
# access_ACO_CD_EIS(ecm_num=4, range_pair=[40, 60])
# access_ACO_CD_EIS(ecm_num=4, range_pair=[60, 80])
# access_ACO_CD_EIS(ecm_num=4, range_pair=[80, 100])

# ECM-5
# access_ACO_CD_EIS(ecm_num=5, range_pair=[0, 20])
# access_ACO_CD_EIS(ecm_num=5, range_pair=[20, 40])
# access_ACO_CD_EIS(ecm_num=5, range_pair=[40, 60])
# access_ACO_CD_EIS(ecm_num=5, range_pair=[60, 80])
# access_ACO_CD_EIS(ecm_num=5, range_pair=[80, 100])

# ECM-6
# access_ACO_CD_EIS(ecm_num=6, range_pair=[0, 20])
# access_ACO_CD_EIS(ecm_num=6, range_pair=[20, 40])
# access_ACO_CD_EIS(ecm_num=6, range_pair=[40, 60])
# access_ACO_CD_EIS(ecm_num=6, range_pair=[60, 80])
# access_ACO_CD_EIS(ecm_num=6, range_pair=[80, 100])

# ECM-7
# access_ACO_CD_EIS(ecm_num=7, range_pair=[0, 20])
# access_ACO_CD_EIS(ecm_num=7, range_pair=[20, 40])
# access_ACO_CD_EIS(ecm_num=7, range_pair=[40, 60])
# access_ACO_CD_EIS(ecm_num=7, range_pair=[60, 80])
# access_ACO_CD_EIS(ecm_num=7, range_pair=[80, 100])

# ECM-8
# access_ACO_CD_EIS(ecm_num=8, range_pair=[0, 20])
# access_ACO_CD_EIS(ecm_num=8, range_pair=[20, 40])
# access_ACO_CD_EIS(ecm_num=8, range_pair=[40, 60])
# access_ACO_CD_EIS(ecm_num=8, range_pair=[60, 80])
# access_ACO_CD_EIS(ecm_num=8, range_pair=[80, 100])

# ECM-9
# access_ACO_CD_EIS(ecm_num=9, range_pair=[0, 20])
# access_ACO_CD_EIS(ecm_num=9, range_pair=[20, 40])
# access_ACO_CD_EIS(ecm_num=9, range_pair=[40, 60])
# access_ACO_CD_EIS(ecm_num=9, range_pair=[60, 80])
# access_ACO_CD_EIS(ecm_num=9, range_pair=[80, 100])

# python aco_cd.py