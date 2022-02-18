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

class TLBO_0:
    """
    Refer:
        Paper0: Teaching–learning-based optimization: A novel method for constrained mechanical design optimization problems
    """
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(limits_list, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []
        for iter_index in range(self.iter_num):
            # Calculate the mean in each dimension
            dim_mean_list = []
            for i in range(len(self.limits_list)):
                dim_sum = 0.0
                for entity in self.entity_list:
                    dim_sum += entity.x_list[i]
                dim_mean_list.append(dim_sum / len(self.limits_list))

            # Select the best entity TEACHER
            min_fitness = 100000
            teacher_index = 10000
            for i in range(self.entity_num):
                if self.entity_list[i].fitness <= min_fitness:
                    teacher_index = i
                    min_fitness = self.entity_list[i].fitness
            teacher = self.entity_list[teacher_index]

            if teacher.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(teacher)
            cur_best_entity_list.append(copy.deepcopy(teacher))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Teaching phase
            r = random.random()
            # TEACHER FACTOR by Eq 2
            t_f = random.choice([1,2])
            diff_list = [r * (t_x - t_f  * m_x) for t_x, m_x in zip(teacher.x_list, dim_mean_list)]

            # Iterate over the students
            # Get the index of student
            student_index_set = set(range(self.entity_num)) - {teacher_index}
            for student_index in student_index_set:
                student = self.entity_list[student_index]
                tmp_x_list = [x + d_x for x, d_x in zip(student.x_list, diff_list)]
                tmp_fitness = self.fitness_function(tmp_x_list)
                if tmp_fitness < student.fitness:
                    student.x_list = tmp_x_list
                    student.update()

            # Learning phase
            student_index_list = list(student_index_set)
            for i in range(self.entity_num):
                random_student_index_pair = random.sample(student_index_list, 2)
                student_a_index = random_student_index_pair[0]
                student_b_index = random_student_index_pair[1]
                student_a = self.entity_list[student_a_index]
                student_b = self.entity_list[student_b_index]

                r = random.random()
                if student_a.fitness < student_b.fitness:
                    b_tmp_x_list = [b_x + r * (a_x - b_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    b_tmp_fitness = self.fitness_function(b_tmp_x_list)
                    if b_tmp_fitness < student_b.fitness:
                        student_b.x_list = b_tmp_x_list
                        student_b.update()
                else:
                    a_tmp_x_list = [a_x + r * (b_x - a_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(a_tmp_x_list)
                    if a_tmp_fitness < student_a.fitness:
                        student_a.x_list = a_tmp_x_list
                        student_a.update()
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 50
#     entity_num = 10
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     tlbo = TLBO_0(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     cur_best_entity_list, global_best_entity_list = tlbo.search()
#     print('Best entity position:', tlbo.global_best_entity.x_list)
#     print('Fitness:', tlbo.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class TLBO_1:
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(limits_list, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(limits_list, fitness_function)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        for iter_index in range(self.iter_num):
            # Calculate the mean in each dimension
            dim_mean_list = []
            for i in range(len(self.limits_list)):
                dim_sum = 0.0
                for entity in self.entity_list:
                    dim_sum += entity.x_list[i]
                dim_mean_list.append(dim_sum / len(self.limits_list))

            # Select the best entity TEACHER
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            teacher = self.entity_list[0]

            if teacher.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(teacher)
            cur_best_entity_list.append(copy.deepcopy(teacher))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Teaching phase
            r = random.random()
            # TEACHER FACTOR by Eq 2
            t_f = random.choice([1, 2])
            diff_list = [r * (t_x - t_f  * m_x) for t_x, m_x in zip(teacher.x_list, dim_mean_list)]

            # Iterate over the students
            for student_i in range(1, self.entity_num):
                student = self.entity_list[student_i]
                tmp_x_list = [x + d_x for x, d_x in zip(student.x_list, diff_list)]
                tmp_fitness = self.fitness_function(tmp_x_list)
                if tmp_fitness < student.fitness:
                    student.x_list = tmp_x_list
                    student.update()

            # Learning phase
            for i in range(self.entity_num):
                student_a = self.entity_list[i]

                # Randomly select student_b
                left_student_index_list = list(range(self.entity_num))
                left_student_index_list.remove(i)
                student_b_index = random.choice(left_student_index_list)
                student_b = self.entity_list[student_b_index]

                r = random.random()
                if student_a.fitness < student_b.fitness:
                    a_tmp_x_list = [a_x + r * (a_x - b_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(a_tmp_x_list)
                else:
                    a_tmp_x_list = [a_x + r * (b_x - a_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(a_tmp_x_list)
                if a_tmp_fitness < student_a.fitness:
                    student_a.x_list = a_tmp_x_list
                    student_a.update()
        return cur_best_entity_list, global_best_entity_list

# if __name__ == '__main__':
#     iter_num = 50
#     entity_num = 10
#     dim = 10
#
#     f1_limits_list = [[-100, 100] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     tlbo = TLBO_1(iter_num, entity_num, f1_limits_list, f1_fitness_function)
#     cur_best_entity_list, global_best_entity_list = tlbo.search()
#     print('Best entity position:', tlbo.global_best_entity.x_list)
#     print('Fitness:', tlbo.global_best_entity.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in cur_best_entity_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line1.set_dashes([5, 5, 10, 5])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
#     line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class TLBO_EIS:
    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

    def search(self):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Calculate the mean in each dimension
            dim_mean_list = []
            for i in range(len(self.limits_list)):
                dim_sum = 0.0
                for entity in self.entity_list:
                    dim_sum += entity.x_list[i]
                dim_mean_list.append(dim_sum / len(self.limits_list))

            # Select the best entity TEACHER
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            teacher = self.entity_list[0]

            if teacher.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(teacher)
            cur_best_entity_list.append(copy.deepcopy(teacher))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Teaching phase
            r = random.random()
            # TEACHER FACTOR by Eq 2
            t_f = random.choice([1, 2])
            diff_list = [r * (t_x - t_f  * m_x) for t_x, m_x in zip(teacher.x_list, dim_mean_list)]

            # Iterate over the students
            for student_i in range(1, self.entity_num):
                student = self.entity_list[student_i]
                tmp_x_list = [x + d_x for x, d_x in zip(student.x_list, diff_list)]
                tmp_fitness = self.fitness_function(self.exp_data_dict, tmp_x_list)
                if tmp_fitness < student.fitness:
                    student.x_list = tmp_x_list
                    student.update()

            # Learning phase
            for i in range(self.entity_num):
                student_a = self.entity_list[i]

                # Randomly select student_b
                left_student_index_list = list(range(self.entity_num))
                left_student_index_list.remove(i)
                student_b_index = random.choice(left_student_index_list)
                student_b = self.entity_list[student_b_index]

                r = random.random()
                if student_a.fitness < student_b.fitness:
                    a_tmp_x_list = [a_x + r * (a_x - b_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(self.exp_data_dict, a_tmp_x_list)
                else:
                    a_tmp_x_list = [a_x + r * (b_x - a_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(self.exp_data_dict, a_tmp_x_list)
                if a_tmp_fitness < student_a.fitness:
                    student_a.x_list = a_tmp_x_list
                    student_a.update()

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

class TLBO_EIS_access:
    class Entity:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict, self.x_list)

        def update(self):
            for i, x in enumerate(self.x_list):
                if (x < self.limits_list[i][0]) or (x > self.limits_list[i][1]):
                    self.x_list[i] = random.uniform(self.limits_list[i][0], self.limits_list[i][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, entity_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.entity_num = entity_num
        self.fitness_function = fitness_function

        self.entity_list = [self.Entity(self.exp_data_dict, fitness_function) for i in range(self.entity_num)]
        self.global_best_entity = self.Entity(self.exp_data_dict, fitness_function)

    def search(self, res_fn, start_time):
        cur_best_entity_list = []
        global_best_entity_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Calculate the mean in each dimension
            dim_mean_list = []
            for i in range(len(self.limits_list)):
                dim_sum = 0.0
                for entity in self.entity_list:
                    dim_sum += entity.x_list[i]
                dim_mean_list.append(dim_sum / len(self.limits_list))

            # Select the best entity TEACHER
            self.entity_list.sort(key=lambda entity: entity.fitness, reverse=False)
            teacher = self.entity_list[0]

            if teacher.fitness < self.global_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(teacher)
            cur_best_entity_list.append(copy.deepcopy(teacher))
            global_best_entity_list.append(copy.deepcopy(self.global_best_entity))

            # Teaching phase
            r = random.random()
            # TEACHER FACTOR by Eq 2
            t_f = random.choice([1, 2])
            diff_list = [r * (t_x - t_f  * m_x) for t_x, m_x in zip(teacher.x_list, dim_mean_list)]

            # Iterate over the students
            for student_i in range(1, self.entity_num):
                student = self.entity_list[student_i]
                tmp_x_list = [x + d_x for x, d_x in zip(student.x_list, diff_list)]
                tmp_fitness = self.fitness_function(self.exp_data_dict, tmp_x_list)
                if tmp_fitness < student.fitness:
                    student.x_list = tmp_x_list
                    student.update()

            # Learning phase
            for i in range(self.entity_num):
                student_a = self.entity_list[i]

                # Randomly select student_b
                left_student_index_list = list(range(self.entity_num))
                left_student_index_list.remove(i)
                student_b_index = random.choice(left_student_index_list)
                student_b = self.entity_list[student_b_index]

                r = random.random()
                if student_a.fitness < student_b.fitness:
                    a_tmp_x_list = [a_x + r * (a_x - b_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(self.exp_data_dict, a_tmp_x_list)
                else:
                    a_tmp_x_list = [a_x + r * (b_x - a_x) for a_x, b_x in zip(student_a.x_list, student_b.x_list)]
                    a_tmp_fitness = self.fitness_function(self.exp_data_dict, a_tmp_x_list)
                if a_tmp_fitness < student_a.fitness:
                    student_a.x_list = a_tmp_x_list
                    student_a.update()

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

def access_TLBO_EIS():
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
            goa = TLBO_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, entity_num=10*para_num)
            res_fn = 'tlbo_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('TLBO left: {0}'.format(900 - counter))
# access_TLBO_EIS()