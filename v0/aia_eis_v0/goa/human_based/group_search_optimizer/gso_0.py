import math
import copy
import random

# 'phi' == 'angle'
def polar_2_car(phi_list):
    d_last = math.cos(math.radians(phi_list[-1]))
    d_list = []
    for i in range(len(phi_list)):
        tmp_d = 1.0
        # Calculate the cosine part
        for c in range(i):
            tmp_d *= math.cos(math.radians(phi_list[c]))
        # Calculate the sine part
        for s in range(i, len(phi_list)):
            tmp_d *= math.sin(math.radians(phi_list[s]))
        d_list.append(tmp_d)
    d_list.append(d_last)
    return d_list
# ---------------------- test polar_2_car ----------------------
# if __name__ == '__main__':
#     phi_list = [60, 45]
#     R(RC)_IS_lin-kk_res.txt = polar_2_car(phi_list)
#     print(R(RC)_IS_lin-kk_res.txt)
# R(RC)_IS_lin-kk_res.txt: [0.6123724356957946, 0.35355339059327384, 0.7071067811865476]
# ---------------------- test polar_2_car ----------------------

class GSO_0:
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

            # 极坐标表示的维数 （半径长1维 + 角度维数） = 笛卡尔坐标维数
            self.head_angle_list = [random.uniform(0, 180) for i in range(len(self.limits_list) - 1)]
            self.head_direction_list = polar_2_car(self.head_angle_list)

            self.producer_max_iter = 10
            self.producer_flag = 0
            self.max_turn_angle = 90

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

        self.entity_list = [self.Entity(limits_list, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

        # The mas search length should be adapted along with the range on each dimension
        # I set the [mas search length] = 1/10 * dimension_range
        self.max_search_len = [0.1 * (limit[1] - limit[0]) for limit in self.limits_list]

        self.theta_max = 120
        # maximum turning angle
        self.max_turn_angle = 90

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []
        for iter_index in range(self.iter_num):
            # Find the index of the producer by comparing their fitness
            producer_index = 10000
            min_fitness = 1000000
            for i in range(self.entity_num):
                if self.entity_list[i].fitness <= min_fitness:
                    producer_index = i
                    min_fitness = self.entity_list[i].fitness

            producer = self.entity_list[producer_index]
            producer.set_producer_flag(True)

            if producer.fitness < self.global_best_entity.fitness:
                self.global_best_entity = producer
            # cur_best_entity_list.append(producer)
            # global_best_entity_list.append(self.global_best_entity)
            cur_best_entity_list.append(copy.deepcopy(producer))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # producer will search ahead and laterally to randomly three positions (Z, L, and R)
            # r1 is one dimension, r2 is n-1 dimension BY Eq 4
            r1 = random.gauss(mu=0, sigma=1.0)
            r2_list = [random.random() for i in range(len(self.limits_list) - 1)]

            position_z = [x + r1 * max_len * d for x, max_len, d in zip(producer.x_list, self.max_search_len, producer.head_direction_list)]

            tmp_right_angle_list = [phi + r2 * self.theta_max / 2 for phi, r2 in zip(producer.head_angle_list, r2_list)]
            tmp_right_head_direction_list = polar_2_car(tmp_right_angle_list)
            position_right = [x + r1 * max_len * tmp_d for x, max_len, tmp_d in zip(producer.x_list, self.max_search_len, tmp_right_head_direction_list)]

            tmp_left_angle_list = [phi - r2 * self.theta_max / 2 for phi, r2 in zip(producer.head_angle_list, r2_list)]
            tmp_left_head_direction_list = polar_2_car(tmp_left_angle_list)
            position_left = [x + r1 * max_len * tmp_d for x, max_len, tmp_d in zip(producer.x_list, self.max_search_len, tmp_left_head_direction_list)]

            producer_fitness_list = [self.fitness_function(pos_list) for pos_list in [producer.x_list, position_z, position_right, position_left]]
            # If producer find a better position (fitness_index != 0), then move to it and do not change head angle
            if producer_fitness_list.index(min(producer_fitness_list)) != 0:
                better_pos_index = producer_fitness_list.index(min(producer_fitness_list))
                better_x_list = [producer.x_list, position_z, position_left, position_right][better_pos_index]
                producer.update(better_x_list, new_angle=False)
            # else producer stay in its current position and turn its head to a new angle using Eq 5
            else:
                producer.update(t_x_list=None, new_angle=True)

            # RANDOMLY Take 80% entity of the sorted entity list as scrounger
            # scrounger_list = sorted_entity_list[1 : int(0.8 * self.entity_num)]
            # RANDOMLY Take 80% entity of the sorted entity list as scrounger (REMOVE THE INDEX OF PRODUCER!!!)
            scrounger_index_set = set(random.sample(list(range(self.entity_num)), int(0.8 * self.entity_num))) - {producer_index}
            scrounger_list = [self.entity_list[s_i] for s_i in scrounger_index_set]
            for s_index in range(len(scrounger_list)):
                r3_list = [random.random() for i in range(len(self.limits_list))]
                tmp_x_list = [x + r3 * (p_x - x) for x, r3, p_x in zip(scrounger_list[s_index].x_list, r3_list, producer.x_list)]
                scrounger_list[s_index].update(t_x_list=tmp_x_list, new_angle=False)

            # The left entity of the sorted entity list are rangers
            # Take use of the set()
            ranger_index_set = set(list(range(self.entity_num))) - scrounger_index_set - {producer_index}
            ranger_list = [self.entity_list[r_i] for r_i in ranger_index_set]
            for r_index in range(len(ranger_list)):
                ranger_r2_list = [random.random() for i in range(len(self.limits_list) - 1)]
                ranger_list[r_index].head_angle_list = [head_angle + r2 * self.max_turn_angle for head_angle, r2 in zip(ranger_list[r_index].head_angle_list, ranger_r2_list)]
                tmp_head_direction_list  = polar_2_car(ranger_list[r_index].head_angle_list)
                r1 = random.random()
                tmp_x_list = [x + r1 * m_l * d_l for x, m_l, d_l in zip(ranger_list[r_index].x_list, self.max_search_len, tmp_head_direction_list)]
                ranger_list[r_index].update(tmp_x_list, new_angle=False)
        return cur_best_entity_list, global_best_entity_list

if __name__ == '__main__':
    iter_num = 500
    entity_num = 10
    dim = 10

    f1_limits_list = [[-100, 100] for i in range(dim)]
    from GA_pack.fittness_functions.f1 import f1

    f1_fitness_function = f1
    gso = GSO_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
    cur_best_entity_list, global_best_entity_list = gso.search()
    print('Best entity position:', gso.global_best_entity.x_list)
    print('Fitness:', gso.global_best_entity.fitness)

    # Draw the best entity in each iteration.
    iter_list = [i for i in range(iter_num)]
    cur_fitness_list = [entity.fitness for entity in cur_best_entity_list]
    cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
    line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
    line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
    line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    ax.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Error rate')
    plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
    plt.show()