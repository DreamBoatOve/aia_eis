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

class Big_Bang_Big_Crunch:
    """
    Refer:
        paper: A new optimization method: Big Bang–Big Crunch
    """
    class Atom:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]
            self.fitness = fitness_function(self.x_list)

        def update(self):
            for index, x in enumerate(self.x_list):
                if (x < self.limits_list[index][0]) or (x > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, atom_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.atom_num = atom_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Step 1: Form an initial generation of N candidates in a random manner. Respect the limits of the search space.
        self.atom_list = [self.Atom(limits_list, fitness_function) for i in range(self.atom_num)]

        self.global_best_atom = self.Atom(limits_list, fitness_function)
        self.global_best_atom.fitness = float('inf')

    def evolve(self):
        current_best_atom_list = []
        global_best_atom_list = []

        for iter_index in range(self.iter_num):
            # Find minimum
            current_best_atom = sorted(self.atom_list, key= lambda atom : atom.fitness, reverse=False)[0]
            if current_best_atom.fitness < self.global_best_atom.fitness:
                self.global_best_atom = copy.deepcopy(current_best_atom)
            current_best_atom_list.append(copy.deepcopy(current_best_atom))
            global_best_atom_list.append(copy.deepcopy(self.global_best_atom))

            # Step 3: Find the center of mass according to Eq. (2). Best fit individual can be chosen as the
            # center of mass instead of using Eq. (2).
            # ---------------- Calculate the "mass center" ----------------
            tmp_inversed_fitness_list = [1.0 / atom.fitness for atom in self.atom_list]
            sum_inversed_fitness = sum(tmp_inversed_fitness_list)

            # The calculation of mass center refers Eq 2
            tmp_center_x_list = []
            for index in range(len(self.limits_list)):
                mul_list = []
                for atom_index in range(self.atom_num):
                    x = self.atom_list[atom_index].x_list[index]
                    mul = x * tmp_inversed_fitness_list[atom_index]
                    mul_list.append(mul)
                tmp_center_x_list.append(sum(mul_list) / sum_inversed_fitness)

            center_atom = self.Atom(self.limits_list, self.fitness_function)
            center_atom.x_list = tmp_center_x_list
            center_atom.update()
            # ---------------- Calculate the "mass center" ----------------

            # Generate fragment atoms and update fitness
            # a = 0.5
            tmp_atom_list = []
            for i in range(self.atom_num):
                """
                Step 4:
                    Calculate new candidates around the center of mass by adding or subtracting a normal random number
                    whose value decreases as the iterations elapse. This can be formalized as Eq 4:
                    Xc(c_x) stands for center of mass, 
                    l(up_bound) is the upper limit of the parameter,
                    r is a normal random number
                    k is the iteration step. 
                    Then new point x_new is upper and lower bounded.
                """
                tmp_x_list = []
                for index, c_x in enumerate(center_atom.x_list):
                    # It requires NORMAL random number, not a uniformly distributed random number
                    # r = 2.0 * random.random() - 1.0
                    r = random.normalvariate(mu=0, sigma=1)
                    # up_bound = self.limits_list[index][1]
                    # Eq 4
                    x = c_x + r * 0.5 * (self.limits_list[index][1] - self.limits_list[index][0]) / (1 + iter_index)
                    tmp_x_list.append(x)
                tmp_atom = self.Atom(self.limits_list, self.fitness_function)
                tmp_atom.x_list = tmp_x_list
                tmp_atom.update()
                tmp_atom_list.append(tmp_atom)
            self.atom_list = copy.deepcopy(tmp_atom_list)
        return current_best_atom_list, global_best_atom_list

# if __name__ == '__main__':
#     iter_num = 2000
#     atom_num = 40
#     dim = 10
#
#     f1_limits_list = [[-150, 80] for i in range(dim)]
#     from GA_pack.fittness_functions.f1 import f1
#
#     f1_fitness_function = f1
#     bb_bc = Big_Bang_Big_Crunch(iter_num, atom_num, f1_limits_list, f1_fitness_function)
#     current_best_atom_list, global_best_atom_list = bb_bc.evolve()
#     print('Best entity position:', bb_bc.global_best_atom.x_list)
#     print('Fitness:', bb_bc.global_best_atom.fitness)
#
#     # Draw the best entity in each iteration.
#     iter_list = [i for i in range(iter_num)]
#     cur_fitness_list = [entity.fitness for entity in current_best_atom_list]
#     cur_global_fitness_list = [entity.fitness for entity in global_best_atom_list]
#
#     import matplotlib.pyplot as plt
#     fig, ax = plt.subplots()
#     line1, = ax.plot(iter_list, cur_fitness_list, linestyle='-', label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, atom_num, dim))
#     # line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     line2, = ax.plot(iter_list, cur_global_fitness_list, linestyle='--', label='Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, atom_num, dim))
#     # line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
#     ax.legend()
#     plt.xlabel('Iteration times')
#     plt.ylabel('Error rate')
#     plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
#     plt.show()

class BB_BC_EIS:
    """
    Refer:
        paper: A new optimization method: Big Bang–Big Crunch
    """
    class Atom:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict ,self.x_list)

        def update(self):
            for index, x in enumerate(self.x_list):
                if (x < self.limits_list[index][0]) or (x > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, atom_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.atom_num = atom_num

        self.fitness_function = fitness_function

        # Step 1: Form an initial generation of N candidates in a random manner. Respect the limits of the search space.
        self.atom_list = [self.Atom(self.exp_data_dict, fitness_function) for i in range(self.atom_num)]

        self.global_best_atom = self.Atom(self.exp_data_dict, fitness_function)
        self.global_best_atom.fitness = float('inf')

    def search(self):
        current_best_atom_list = []
        global_best_atom_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Find minimum
            current_best_atom = sorted(self.atom_list, key= lambda atom : atom.fitness, reverse=False)[0]
            if current_best_atom.fitness < self.global_best_atom.fitness:
                self.global_best_atom = copy.deepcopy(current_best_atom)
            current_best_atom_list.append(copy.deepcopy(current_best_atom))
            global_best_atom_list.append(copy.deepcopy(self.global_best_atom))

            # Step 3: Find the center of mass according to Eq. (2). Best fit individual can be chosen as the
            # center of mass instead of using Eq. (2).
            # ---------------- Calculate the "mass center" ----------------
            tmp_inversed_fitness_list = [1.0 / atom.fitness for atom in self.atom_list]
            sum_inversed_fitness = sum(tmp_inversed_fitness_list)

            # The calculation of mass center refers Eq 2
            tmp_center_x_list = []
            for index in range(len(self.limits_list)):
                mul_list = []
                for atom_index in range(self.atom_num):
                    x = self.atom_list[atom_index].x_list[index]
                    mul = x * tmp_inversed_fitness_list[atom_index]
                    mul_list.append(mul)
                tmp_center_x_list.append(sum(mul_list) / sum_inversed_fitness)

            center_atom = self.Atom(self.exp_data_dict, self.fitness_function)
            center_atom.x_list = tmp_center_x_list
            center_atom.update()
            # ---------------- Calculate the "mass center" ----------------

            # Generate fragment atoms and update fitness
            tmp_atom_list = []
            for i in range(self.atom_num):
                """
                Step 4:
                    Calculate new candidates around the center of mass by adding or subtracting a normal random number
                    whose value decreases as the iterations elapse. This can be formalized as Eq 4:
                    Xc(c_x) stands for center of mass, 
                    l(up_bound) is the upper limit of the parameter,
                    r is a normal random number
                    k is the iteration step. 
                    Then new point x_new is upper and lower bounded.
                """
                tmp_x_list = []
                for index, c_x in enumerate(center_atom.x_list):
                    # It requires NORMAL random number, not a uniformly distributed random number
                    # r = 2.0 * random.random() - 1.0
                    r = random.normalvariate(mu=0, sigma=1)
                    # up_bound = self.limits_list[index][1]
                    # Eq 4
                    x = c_x + r * 0.5 * (self.limits_list[index][1] - self.limits_list[index][0]) / (1 + iter)
                    tmp_x_list.append(x)
                tmp_atom = self.Atom(self.exp_data_dict, self.fitness_function)
                tmp_atom.x_list = tmp_x_list
                tmp_atom.update()
                tmp_atom_list.append(tmp_atom)
            self.atom_list = copy.deepcopy(tmp_atom_list)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_atom_list[-2].x_list, global_best_atom_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, iter=iter,
                                                                max_iter_time=self.iter_num,
                                                                data_dict=self.exp_data_dict)
                if goa_criterion:
                    continue_criterion = False
            iter += 1
        return current_best_atom_list, global_best_atom_list, iter, chi_squared

class BB_BC_EIS_access:
    """
    Refer:
        paper: A new optimization method: Big Bang–Big Crunch
    """
    class Atom:
        def __init__(self, exp_data_dict, fitness_function):
            self.exp_data_dict = exp_data_dict
            self.limits_list = exp_data_dict['limit']

            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.exp_data_dict ,self.x_list)

        def update(self):
            for index, x in enumerate(self.x_list):
                if (x < self.limits_list[index][0]) or (x > self.limits_list[index][1]):
                    self.x_list[index] = random.uniform(self.limits_list[index][0], self.limits_list[index][1])
            self.fitness = self.fitness_function(self.exp_data_dict, self.x_list)

    def __init__(self, exp_data_dict, iter_num, atom_num, fitness_function=cal_EIS_WSE_fitness_1):
        self.exp_data_dict = exp_data_dict
        self.limits_list = exp_data_dict['limit']

        self.iter_num = iter_num
        self.atom_num = atom_num

        self.fitness_function = fitness_function

        # Step 1: Form an initial generation of N candidates in a random manner. Respect the limits of the search space.
        self.atom_list = [self.Atom(self.exp_data_dict, fitness_function) for i in range(self.atom_num)]

        self.global_best_atom = self.Atom(self.exp_data_dict, fitness_function)
        self.global_best_atom.fitness = float('inf')

    def search(self, res_fn, start_time):
        current_best_atom_list = []
        global_best_atom_list = []

        continue_criterion = True
        iter = 0
        while continue_criterion:
            # Find minimum
            current_best_atom = sorted(self.atom_list, key= lambda atom : atom.fitness, reverse=False)[0]
            if current_best_atom.fitness < self.global_best_atom.fitness:
                self.global_best_atom = copy.deepcopy(current_best_atom)
            current_best_atom_list.append(copy.deepcopy(current_best_atom))
            global_best_atom_list.append(copy.deepcopy(self.global_best_atom))

            # Step 3: Find the center of mass according to Eq. (2). Best fit individual can be chosen as the
            # center of mass instead of using Eq. (2).
            # ---------------- Calculate the "mass center" ----------------
            tmp_inversed_fitness_list = [1.0 / atom.fitness for atom in self.atom_list]
            sum_inversed_fitness = sum(tmp_inversed_fitness_list)

            # The calculation of mass center refers Eq 2
            tmp_center_x_list = []
            for index in range(len(self.limits_list)):
                mul_list = []
                for atom_index in range(self.atom_num):
                    x = self.atom_list[atom_index].x_list[index]
                    mul = x * tmp_inversed_fitness_list[atom_index]
                    mul_list.append(mul)
                tmp_center_x_list.append(sum(mul_list) / sum_inversed_fitness)

            center_atom = self.Atom(self.exp_data_dict, self.fitness_function)
            center_atom.x_list = tmp_center_x_list
            center_atom.update()
            # ---------------- Calculate the "mass center" ----------------

            # Generate fragment atoms and update fitness
            tmp_atom_list = []
            for i in range(self.atom_num):
                """
                Step 4:
                    Calculate new candidates around the center of mass by adding or subtracting a normal random number
                    whose value decreases as the iterations elapse. This can be formalized as Eq 4:
                    Xc(c_x) stands for center of mass, 
                    l(up_bound) is the upper limit of the parameter,
                    r is a normal random number
                    k is the iteration step. 
                    Then new point x_new is upper and lower bounded.
                """
                tmp_x_list = []
                for index, c_x in enumerate(center_atom.x_list):
                    # It requires NORMAL random number, not a uniformly distributed random number
                    # r = 2.0 * random.random() - 1.0
                    r = random.normalvariate(mu=0, sigma=1)
                    # up_bound = self.limits_list[index][1]
                    # Eq 4
                    x = c_x + r * 0.5 * (self.limits_list[index][1] - self.limits_list[index][0]) / (1 + iter)
                    tmp_x_list.append(x)
                tmp_atom = self.Atom(self.exp_data_dict, self.fitness_function)
                tmp_atom.x_list = tmp_x_list
                tmp_atom.update()
                tmp_atom_list.append(tmp_atom)
            self.atom_list = copy.deepcopy(tmp_atom_list)

            # There are two entities only after at least two iteration
            # If there is global_best_entity_list, use it,
            # If not, use current_best_entity_list to replace
            if iter >= 1:
                x_lists_list = [global_best_atom_list[-2].x_list, global_best_atom_list[-1].x_list]
                goa_criterion, chi_squared = goa_criterion_pack(x_lists_list=x_lists_list, \
                                                                iter=iter, \
                                                                max_iter_time=self.iter_num, \
                                                                data_dict=self.exp_data_dict, \
                                                                CS_limit=1e-70)
                # Write R(RC)_IS_lin-kk_res.txt into a txt file
                # R(RC)_IS_lin-kk_res.txt = iter_time + fitted_para_list + Chi-Squared + Code running time
                with open(res_fn, 'a+') as file:
                    line = str(iter) + ',[' \
                           + ','.join([str(para) for para in global_best_atom_list[-1].x_list]) + '],' \
                           + str(chi_squared) + ',' + str(perf_counter() - start_time) + '\n'
                    file.write(line)

                if goa_criterion:
                    continue_criterion = False
            iter += 1

def access_BB_BC_EIS():
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
            goa = BB_BC_EIS_access(exp_data_dict=sim_ecm, iter_num=10000, atom_num=10*para_num)
            res_fn = 'bb_bc_ecm{0}_'.format(i) + get_Num_len(num=j, length=2) + '.txt'
            # ------------------------------  Change GOA name ------------------------------
            goa.search(res_fn, start_time=t_start)

            counter += 1
            print('BB_BC left: {0}'.format(900 - counter))
access_BB_BC_EIS()