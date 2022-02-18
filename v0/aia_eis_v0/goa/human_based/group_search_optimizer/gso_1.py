import math
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

# 'phi' == 'angle'
def polar_2_car_1(phi_list):
    """
    Function:
        Convert polarization coordinate to euclidean
    Refer:
        三维空间中直角坐标与球坐标的相互转换
            https://www.cnblogs.com/hans_gis/archive/2012/11/21/2755126.html
            没有讲高维空间的坐标转换，也找不到其他的博客，只能模仿这个

        paper0 Eq 1
        paper1 Eq 1
            It gives an calculation example
            When phi_list = [Pi / 3, Pi / 4], direction unit vector (D) should be [1/2 = 0.5, sqrt(6)/4 = 0.6123, sqrt(2)/2 = 0.707]
        paper2 Eq 1.3
            It has GOOD explanation
    :param
        phi_list:
            list[angle(float, 0 ~ 180 degree)]
    :return:
    """
    # math.radians(x, /): Convert angle x from degrees to radians
    d_last = math.cos(math.radians(phi_list[-1]))
    d_list = []
    for i in range(len(phi_list)):
        tmp_d = 1.0
        # Calculate the cosine part
        if i > 0:
            tmp_d *= math.cos(math.radians(phi_list[i]))
        # Calculate the sine part
        for s in range(i, len(phi_list)):
            tmp_d *= math.sin(math.radians(phi_list[s]))
        d_list.append(tmp_d)
    d_list.append(d_last)
    return d_list

# ---------------------- test polar_2_car ----------------------
# if __name__ == '__main__':
#     phi_list = [60, 45]
#     R(RC)_IS_lin-kk_res.txt = polar_2_car_1(phi_list)
#     print(R(RC)_IS_lin-kk_res.txt)
# R(RC)_IS_lin-kk_res.txt: [0.6123724356957946, 0.5000000000000001, 0.7071067811865476]
# ---------------------- test polar_2_car ----------------------

class GSO_1:
    """
    Refer:
        Paper:
            paper0: A Novel Group Search Optimizer Inspired by Animal Behavioural Ecology
            paper1: Group Search Optimizer: An Optimization Algorithm Inspired by Animal Searching Behavior
            paper2: Numerical Integration over the n-Dimensional Spherical Shell
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
    Attention:
    """
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

            """
            :param 
                limits_list:
                fitness_function:
                
                Unique
                    Dimension, n
                    Initial head angle of each individual, ϕ0:
                        0 ~ π/4
                    Producer maximum stable iteration times, a:
                        round(sqrt(n + 1))
                    Maximum pursuit angle θmax:
                        Π/(a * a)
                    Maximum turning angle α:
                        Π/(2 * a * a)
                    Maximum pursuit distance lmax:
                        In paper0:
                            lmax = ||U - L|| = sqrt(sum[pow(u - l, 2) for u, l in zip(U, L)])
                        Consider I am dealing with parameters in different scale, the following eq is used:
                            lmax_i = ||Ui - Li||
            """
            # 极坐标表示的维数 （半径长1维 + 角度维数） = 笛卡尔坐标维数
            self.head_angle_list = [random.uniform(0, 45) for i in range(len(self.limits_list) - 1)]
            self.head_direction_list = polar_2_car_1(self.head_angle_list)

            # Dimension, n
            n = len(self.limits_list)
            # Producer maximum stable iteration times, a: round(sqrt(n + 1))
            self.producer_max_iter = int(math.sqrt(n+1))
            self.producer_flag = 0
            # Maximum turning angle α: Π/(2 * a * a)
            self.max_turn_angle = 180 / (2 * pow(self.producer_max_iter, 2))

        def set_producer_flag(self, flag):
            # producer_flag: 1 == producer; 0 == scrounger or ranger
            if (flag == True) and (self.producer_flag == 0):
                self.producer_flag = 1
                self.head_angle_memory_list = []
                self.producer_iter = 1
            elif (flag == True) and (self.producer_flag == 1):
                self.producer_iter += 1
                self.head_angle_memory_list.append(copy.deepcopy(self.head_angle_list))

        def update(self, t_x_list, new_angle):
            # find a better position and move to it, and do not change head angle list
            if t_x_list != None:
                # If a entity is outside of boundary, it will return back to its previous position
                is_out_bool_list = [(x >= self.limits_list[i][0]) and (x <= self.limits_list[i][1]) for i, x in enumerate(t_x_list)]
                # If the entity is in the boundary, the elements in is_out_bool_list should all be True (len(set(is_out_bool_list)) = 1), otherwise, should be True + False (len(set(is_out_bool_list)) = 2)
                if len(set(is_out_bool_list)) == 1:
                    self.x_list = t_x_list
                    self.fitness = self.fitness_function(self.x_list)
            # else producer stay in its current position and turn its head to a new angle using Eq 5
            elif new_angle == True:
                r_list = [random.random() for i in range(len(self.limits_list) - 1)]
                self.head_angle_list = [head_angle + r * self.max_turn_angle for head_angle, r in zip(self.head_angle_list, r_list)]
                # else If the producer can not find a better area after a iterations, it will turn its head back to zero degree using equation (6)
                if self.producer_iter >= self.producer_max_iter:
                    self.head_angle_list = self.head_angle_memory_list[0]
                    self.head_angle_memory_list = []

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        """
        Initialize entities:
            Only one producer (best fitness)
            80% entities are scroungers
            20% entities are rangers
        """
        self.entity_list = [self.Entity(limits_list, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

        # The max search length should be adapted along with the range on each dimension
        # I set the [mas search length] = 1/10 * dimension_range
        self.max_search_len = [0.1 * (limit[1] - limit[0]) for limit in self.limits_list]

        # Dimension, n
        n = len(self.limits_list)
        # Producer maximum stable iteration times, a: round(sqrt(n + 1))
        self.producer_max_iter = int(math.sqrt(n + 1))
        # Maximum pursuit angle θmax: Π/(a * a)
        self.theta_max = 180 / (self.producer_max_iter ** 2)
        # Maximum turning angle α: Π/(2 * a * a)
        self.max_turn_angle = 180 / (2 * pow(self.producer_max_iter, 2))

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        for iter_index in range(self.iter_num):
            self.entity_list.sort(key=lambda entity:entity.fitness, reverse=False)
            # ------------------------------ Producer ------------------------------
            producer = self.entity_list[0]
            producer.set_producer_flag(True)

            if producer.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(producer)
            cur_best_entity_list.append(copy.deepcopy(producer))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # producer will search ahead and laterally to randomly three positions (Z, L, and R)
            # r1 is one dimension, r2 is n-1 dimension BY Eq 4
            r1 = random.gauss(mu=0, sigma=1.0)
            r2_list = [random.random() for i in range(len(self.limits_list) - 1)]

            # Producer scans at zero degree
            position_z = [x + r1 * max_len * d for x, max_len, d\
                          in zip(producer.x_list, self.max_search_len, producer.head_direction_list)]
            z_fitness = self.fitness_function(position_z)

            # Producer scans the right hand side
            tmp_right_angle_list = [phi + r2 * self.theta_max / 2 for phi, r2 in\
                                    zip(producer.head_angle_list, r2_list)]
            tmp_right_head_direction_list = polar_2_car_1(tmp_right_angle_list)
            position_right = [x + r1 * max_len * tmp_d for x, max_len, tmp_d\
                              in zip(producer.x_list, self.max_search_len, tmp_right_head_direction_list)]
            right_fitness = self.fitness_function(position_right)

            # Producer scans the left hand side
            tmp_left_angle_list = [phi - r2 * self.theta_max / 2 for phi, r2 in\
                                   zip(producer.head_angle_list, r2_list)]
            tmp_left_head_direction_list = polar_2_car_1(tmp_left_angle_list)
            position_left = [x + r1 * max_len * tmp_d for x, max_len, tmp_d in\
                             zip(producer.x_list, self.max_search_len, tmp_left_head_direction_list)]
            left_fitness = self.fitness_function(position_left)

            three_side_fitness = [z_fitness, right_fitness, left_fitness]
            if producer.fitness > min(three_side_fitness):
                min_index = three_side_fitness.index(min(three_side_fitness))
                if min_index == 0:
                    t_x_list = position_z
                elif min_index == 1:
                    t_x_list = position_right
                elif min_index == 2:
                    t_x_list = position_left
                producer.update(t_x_list, False)
            else:
                producer.update(t_x_list=None, new_angle=True)
            # ------------------------------ Producer ------------------------------

            # ------------------------------ Scrounger ------------------------------
            # RANDOMLY Take 80% entity of the sorted entity list as scrounger (REMOVE THE INDEX OF PRODUCER!!!)
            scrounger_index_list = random.sample(list(range(1, self.entity_num)), int(0.8 * self.entity_num))
            for s_i in scrounger_index_list:
                scrounger = self.entity_list[s_i]
                r3_list = [random.random() for i in range(len(self.limits_list))]
                tmp_x_list = [x + r3 * (p_x - x) for x, r3, p_x in zip(scrounger.x_list, r3_list, producer.x_list)]
                scrounger.update(t_x_list=tmp_x_list, new_angle=False)
            # ------------------------------ Scrounger ------------------------------

            # ------------------------------ Ranger ------------------------------
            # The left entity of the sorted entity list are rangers
            # Take use of the set()
            ranger_index_list = list(set(list(range(1, self.entity_num))) - set(scrounger_index_list))
            for r_i in ranger_index_list:
                ranger = self.entity_list[r_i]
                r2_list = [random.random() for i in range(len(self.limits_list) - 1)]
                ranger.head_angle_list = [head_angle + r2 * self.max_turn_angle for head_angle, r2 in zip(ranger.head_angle_list, r2_list)]
                ranger.head_direction_list = polar_2_car_1(ranger.head_angle_list)
                r1 = random.random()
                tmp_x_list = [x + r1 * m_l * d_l for x, m_l, d_l in
                              zip(ranger.x_list, self.max_search_len, ranger.head_direction_list)]
                ranger.update(t_x_list=tmp_x_list, new_angle=False)
            # ------------------------------ Ranger ------------------------------
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 1000
#     entity_num = 10
#     dim = 5
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     gso = GSO_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     cur_best_entity_list, global_best_entity_list = gso.search()
#     print('Best entity position:', gso.global_best_entity.x_list)
#     print('Fitness:', gso.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_entity_list]
#     global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class GSO_EIS:
    """
    Refer:
        Paper:
            paper0: A Novel Group Search Optimizer Inspired by Animal Behavioural Ecology
            paper1: Group Search Optimizer: An Optimization Algorithm Inspired by Animal Searching Behavior
            paper2: Numerical Integration over the n-Dimensional Spherical Shell
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
    Attention:
    """

    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

            """
            :param 
                limits_list:
                fitness_function:

                Unique
                    Dimension, n
                    Initial head angle of each individual, ϕ0:
                        0 ~ π/4
                    Producer maximum stable iteration times, a:
                        round(sqrt(n + 1))
                    Maximum pursuit angle θmax:
                        Π/(a * a)
                    Maximum turning angle α:
                        Π/(2 * a * a)
                    Maximum pursuit distance lmax:
                        In paper0:
                            lmax = ||U - L|| = sqrt(sum[pow(u - l, 2) for u, l in zip(U, L)])
                        Consider I am dealing with parameters in different scale, the following eq is used:
                            lmax_i = ||Ui - Li||
            """
            # 极坐标表示的维数 （半径长1维 + 角度维数） = 笛卡尔坐标维数
            self.head_angle_list = [random.uniform(0, 45) for i in range(len(self.limits_list) - 1)]
            self.head_direction_list = polar_2_car_1(self.head_angle_list)

            # Dimension, n
            n = len(self.limits_list)
            # Producer maximum stable iteration times, a: round(sqrt(n + 1))
            self.producer_max_iter = int(math.sqrt(n + 1))
            self.producer_flag = 0
            # Maximum turning angle α: Π/(2 * a * a)
            self.max_turn_angle = 180 / (2 * pow(self.producer_max_iter, 2))

        def set_producer_flag(self, flag):
            # producer_flag: 1 == producer; 0 == scrounger or ranger
            if (flag == True) and (self.producer_flag == 0):
                self.producer_flag = 1
                self.head_angle_memory_list = []
                self.producer_iter = 1
            elif (flag == True) and (self.producer_flag == 1):
                self.producer_iter += 1
                self.head_angle_memory_list.append(copy.deepcopy(self.head_angle_list))

        def update(self, t_x_list, new_angle):
            # find a better position and move to it, and do not change head angle list
            if t_x_list != None:
                # If a entity is outside of boundary, it will return back to its previous position
                is_out_bool_list = [(x >= self.limits_list[i][0]) and (x <= self.limits_list[i][1]) for i, x in
                                    enumerate(t_x_list)]
                # If the entity is in the boundary, the elements in is_out_bool_list should all be True (len(set(is_out_bool_list)) = 1), otherwise, should be True + False (len(set(is_out_bool_list)) = 2)
                if len(set(is_out_bool_list)) == 1:
                    self.x_list = t_x_list
                    self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)
            # else producer stay in its current position and turn its head to a new angle using Eq 5
            elif new_angle == True:
                r_list = [random.random() for i in range(len(self.limits_list) - 1)]
                self.head_angle_list = [head_angle + r * self.max_turn_angle for head_angle, r in
                                        zip(self.head_angle_list, r_list)]
                # else If the producer can not find a better area after a iterations, it will turn its head back to zero degree using equation (6)
                if self.producer_iter >= self.producer_max_iter:
                    self.head_angle_list = self.head_angle_memory_list[0]
                    self.head_angle_memory_list = []

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        """
        Initialize entities:
            Only one producer (best fitness)
            80% entities are scroungers
            20% entities are rangers
        """
        self.entity_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

        # The max search length should be adapted along with the range on each dimension
        # I set the [mas search length] = 0.5 * dimension_range
        self.max_search_len = [0.5 * (limit[1] - limit[0]) for limit in self.limits_list]

        # Dimension, n
        n = len(self.limits_list)
        # Producer maximum stable iteration times, a: round(sqrt(n + 1))
        self.producer_max_iter = int(math.sqrt(n + 1))
        # Maximum pursuit angle θmax: Π/(a * a)
        self.theta_max = 180 / (self.producer_max_iter ** 2)
        # Maximum turning angle α: Π/(2 * a * a)
        self.max_turn_angle = 180 / (2 * pow(self.producer_max_iter, 2))

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            # ------------------------------ Producer ------------------------------
            producer = self.entity_list[0]
            producer.set_producer_flag(True)

            if producer.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(producer)
            cur_best_entity_list.append(copy.deepcopy(producer))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # producer will search ahead and laterally to randomly three positions (Z, L, and R)
            # r1 is one dimension, r2 is n-1 dimension BY Eq 4
            r1 = random.gauss(mu=0, sigma=1.0)
            r2_list = [random.random() for i in range(len(self.limits_list) - 1)]

            # Producer scans at zero degree
            position_z = [x + r1 * max_len * d for x, max_len, d \
                          in zip(producer.x_list, self.max_search_len, producer.head_direction_list)]
            z_fitness = self.fitness_function(self.exp_data_dict, position_z)

            # Producer scans the right hand side
            tmp_right_angle_list = [phi + r2 * self.theta_max / 2 for phi, r2 in \
                                    zip(producer.head_angle_list, r2_list)]
            tmp_right_head_direction_list = polar_2_car_1(tmp_right_angle_list)
            position_right = [x + r1 * max_len * tmp_d for x, max_len, tmp_d \
                              in zip(producer.x_list, self.max_search_len, tmp_right_head_direction_list)]
            right_fitness = self.fitness_function(self.exp_data_dict, position_right)

            # Producer scans the left hand side
            tmp_left_angle_list = [phi - r2 * self.theta_max / 2 for phi, r2 in \
                                   zip(producer.head_angle_list, r2_list)]
            tmp_left_head_direction_list = polar_2_car_1(tmp_left_angle_list)
            position_left = [x + r1 * max_len * tmp_d for x, max_len, tmp_d in \
                             zip(producer.x_list, self.max_search_len, tmp_left_head_direction_list)]
            left_fitness = self.fitness_function(self.exp_data_dict, position_left)

            three_side_fitness = [z_fitness, right_fitness, left_fitness]
            if producer.fitness > min(three_side_fitness):
                min_index = three_side_fitness.index(min(three_side_fitness))
                if min_index == 0:
                    t_x_list = position_z
                elif min_index == 1:
                    t_x_list = position_right
                elif min_index == 2:
                    t_x_list = position_left
                producer.update(t_x_list, False)
            else:
                producer.update(t_x_list=None, new_angle=True)
            # ------------------------------ Producer ------------------------------

            # ------------------------------ Scrounger ------------------------------
            # RANDOMLY Take 80% entity of the sorted entity list as scrounger (REMOVE THE INDEX OF PRODUCER!!!)
            scrounger_index_list = random.sample(list(range(1, self.entity_num)), int(0.8 * self.entity_num))
            for s_i in scrounger_index_list:
                scrounger = self.entity_list[s_i]
                r3_list = [random.random() for i in range(len(self.limits_list))]
                tmp_x_list = [x + r3 * (p_x - x) for x, r3, p_x in zip(scrounger.x_list, r3_list, producer.x_list)]
                scrounger.update(t_x_list=tmp_x_list, new_angle=False)
            # ------------------------------ Scrounger ------------------------------

            # ------------------------------ Ranger ------------------------------
            # The left entity of the sorted entity list are rangers
            # Take use of the set()
            ranger_index_list = list(set(list(range(1, self.entity_num))) - set(scrounger_index_list))
            for r_i in ranger_index_list:
                ranger = self.entity_list[r_i]
                r2_list = [random.random() for i in range(len(self.limits_list) - 1)]
                ranger.head_angle_list = [head_angle + r2 * self.max_turn_angle for head_angle, r2 in
                                          zip(ranger.head_angle_list, r2_list)]
                ranger.head_direction_list = polar_2_car_1(ranger.head_angle_list)
                r1 = random.random()
                tmp_x_list = [x + r1 * m_l * d_l for x, m_l, d_l in
                              zip(ranger.x_list, self.max_search_len, ranger.head_direction_list)]
                ranger.update(t_x_list=tmp_x_list, new_angle=False)
            # ------------------------------ Ranger ------------------------------

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

class GSO_EIS_access:
    """
    Refer:
        Paper:
            paper0: A Novel Group Search Optimizer Inspired by Animal Behavioural Ecology
            paper1: Group Search Optimizer: An Optimization Algorithm Inspired by Animal Searching Behavior
            paper2: Numerical Integration over the n-Dimensional Spherical Shell
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
    Attention:
    """

    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

            """
            :param 
                limits_list:
                fitness_function:

                Unique
                    Dimension, n
                    Initial head angle of each individual, ϕ0:
                        0 ~ π/4
                    Producer maximum stable iteration times, a:
                        round(sqrt(n + 1))
                    Maximum pursuit angle θmax:
                        Π/(a * a)
                    Maximum turning angle α:
                        Π/(2 * a * a)
                    Maximum pursuit distance lmax:
                        In paper0:
                            lmax = ||U - L|| = sqrt(sum[pow(u - l, 2) for u, l in zip(U, L)])
                        Consider I am dealing with parameters in different scale, the following eq is used:
                            lmax_i = ||Ui - Li||
            """
            # 极坐标表示的维数 （半径长1维 + 角度维数） = 笛卡尔坐标维数
            self.head_angle_list = [random.uniform(0, 45) for i in range(len(self.limits_list) - 1)]
            self.head_direction_list = polar_2_car_1(self.head_angle_list)

            # Dimension, n
            n = len(self.limits_list)
            # Producer maximum stable iteration times, a: round(sqrt(n + 1))
            self.producer_max_iter = int(math.sqrt(n + 1))
            self.producer_flag = 0
            # Maximum turning angle α: Π/(2 * a * a)
            self.max_turn_angle = 180 / (2 * pow(self.producer_max_iter, 2))

        def set_producer_flag(self, flag):
            # producer_flag: 1 == producer; 0 == scrounger or ranger
            if (flag == True) and (self.producer_flag == 0):
                self.producer_flag = 1
                self.head_angle_memory_list = []
                self.producer_iter = 1
            elif (flag == True) and (self.producer_flag == 1):
                self.producer_iter += 1
                self.head_angle_memory_list.append(copy.deepcopy(self.head_angle_list))

        def update(self, t_x_list, new_angle):
            # find a better position and move to it, and do not change head angle list
            if t_x_list != None:
                # If a entity is outside of boundary, it will return back to its previous position
                is_out_bool_list = [(x >= self.limits_list[i][0]) and (x <= self.limits_list[i][1]) for i, x in
                                    enumerate(t_x_list)]
                # If the entity is in the boundary, the elements in is_out_bool_list should all be True (len(set(is_out_bool_list)) = 1), otherwise, should be True + False (len(set(is_out_bool_list)) = 2)
                if len(set(is_out_bool_list)) == 1:
                    self.x_list = t_x_list
                    self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)
            # else producer stay in its current position and turn its head to a new angle using Eq 5
            elif new_angle == True:
                r_list = [random.random() for i in range(len(self.limits_list) - 1)]
                self.head_angle_list = [head_angle + r * self.max_turn_angle for head_angle, r in
                                        zip(self.head_angle_list, r_list)]
                # else If the producer can not find a better area after a iterations, it will turn its head back to zero degree using equation (6)
                if self.producer_iter >= self.producer_max_iter:
                    self.head_angle_list = self.head_angle_memory_list[0]
                    self.head_angle_memory_list = []

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        """
        Initialize entities:
            Only one producer (best fitness)
            80% entities are scroungers
            20% entities are rangers
        """
        self.entity_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

        # The max search length should be adapted along with the range on each dimension
        # I set the [mas search length] = 0.5 * dimension_range
        self.max_search_len = [0.5 * (limit[1] - limit[0]) for limit in self.limits_list]

        # Dimension, n
        n = len(self.limits_list)
        # Producer maximum stable iteration times, a: round(sqrt(n + 1))
        self.producer_max_iter = int(math.sqrt(n + 1))
        # Maximum pursuit angle θmax: Π/(a * a)
        self.theta_max = 180 / (self.producer_max_iter ** 2)
        # Maximum turning angle α: Π/(2 * a * a)
        self.max_turn_angle = 180 / (2 * pow(self.producer_max_iter, 2))

    def search(self, res_fn, start_time):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            # ------------------------------ Producer ------------------------------
            producer = self.entity_list[0]
            producer.set_producer_flag(True)

            if producer.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(producer)
            cur_best_entity_list.append(copy.deepcopy(producer))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # producer will search ahead and laterally to randomly three positions (Z, L, and R)
            # r1 is one dimension, r2 is n-1 dimension BY Eq 4
            r1 = random.gauss(mu=0, sigma=1.0)
            r2_list = [random.random() for i in range(len(self.limits_list) - 1)]

            # Producer scans at zero degree
            position_z = [x + r1 * max_len * d for x, max_len, d \
                          in zip(producer.x_list, self.max_search_len, producer.head_direction_list)]
            z_fitness = self.fitness_function(self.exp_data_dict, position_z)

            # Producer scans the right hand side
            tmp_right_angle_list = [phi + r2 * self.theta_max / 2 for phi, r2 in \
                                    zip(producer.head_angle_list, r2_list)]
            tmp_right_head_direction_list = polar_2_car_1(tmp_right_angle_list)
            position_right = [x + r1 * max_len * tmp_d for x, max_len, tmp_d \
                              in zip(producer.x_list, self.max_search_len, tmp_right_head_direction_list)]
            right_fitness = self.fitness_function(self.exp_data_dict, position_right)

            # Producer scans the left hand side
            tmp_left_angle_list = [phi - r2 * self.theta_max / 2 for phi, r2 in \
                                   zip(producer.head_angle_list, r2_list)]
            tmp_left_head_direction_list = polar_2_car_1(tmp_left_angle_list)
            position_left = [x + r1 * max_len * tmp_d for x, max_len, tmp_d in \
                             zip(producer.x_list, self.max_search_len, tmp_left_head_direction_list)]
            left_fitness = self.fitness_function(self.exp_data_dict, position_left)

            three_side_fitness = [z_fitness, right_fitness, left_fitness]
            if producer.fitness > min(three_side_fitness):
                min_index = three_side_fitness.index(min(three_side_fitness))
                if min_index == 0:
                    t_x_list = position_z
                elif min_index == 1:
                    t_x_list = position_right
                elif min_index == 2:
                    t_x_list = position_left
                producer.update(t_x_list, False)
            else:
                producer.update(t_x_list=None, new_angle=True)
            # ------------------------------ Producer ------------------------------

            # ------------------------------ Scrounger ------------------------------
            # RANDOMLY Take 80% entity of the sorted entity list as scrounger (REMOVE THE INDEX OF PRODUCER!!!)
            scrounger_index_list = random.sample(list(range(1, self.entity_num)), int(0.8 * self.entity_num))
            for s_i in scrounger_index_list:
                scrounger = self.entity_list[s_i]
                r3_list = [random.random() for i in range(len(self.limits_list))]
                tmp_x_list = [x + r3 * (p_x - x) for x, r3, p_x in zip(scrounger.x_list, r3_list, producer.x_list)]
                scrounger.update(t_x_list=tmp_x_list, new_angle=False)
            # ------------------------------ Scrounger ------------------------------

            # ------------------------------ Ranger ------------------------------
            # The left entity of the sorted entity list are rangers
            # Take use of the set()
            ranger_index_list = list(set(list(range(1, self.entity_num))) - set(scrounger_index_list))
            for r_i in ranger_index_list:
                ranger = self.entity_list[r_i]
                r2_list = [random.random() for i in range(len(self.limits_list) - 1)]
                ranger.head_angle_list = [head_angle + r2 * self.max_turn_angle for head_angle, r2 in
                                          zip(ranger.head_angle_list, r2_list)]
                ranger.head_direction_list = polar_2_car_1(ranger.head_angle_list)
                r1 = random.random()
                tmp_x_list = [x + r1 * m_l * d_l for x, m_l, d_l in
                              zip(ranger.x_list, self.max_search_len, ranger.head_direction_list)]
                ranger.update(t_x_list=tmp_x_list, new_angle=False)
            # ------------------------------ Ranger ------------------------------

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

def access_GSO_EIS():
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
            goa = GSO_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'gso_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('GSO left: {0}'.format(900 - counter))
# access_GSO_EIS()