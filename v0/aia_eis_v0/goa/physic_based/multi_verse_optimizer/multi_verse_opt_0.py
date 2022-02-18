import math
import random
import copy

def normalize(nums_list, type = 2):
    num_sum = sum([pow(num, type) for num in nums_list])
    return [num / math.sqrt(num_sum) for num in nums_list]

class MVO_0:
    class Universe:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.uni_objs_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.infl_rate = fitness_function(self.uni_objs_list)

        def update(self):
            # Check the updated universe is in boundary. If not, make it in the boundary again
            for obj_index, uni_obj in enumerate(self.uni_objs_list):
                if uni_obj > self.limits_list[obj_index][1]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][1]
                elif uni_obj < self.limits_list[obj_index][0]:
                    self.uni_objs_list[obj_index] = self.limits_list[obj_index][0]
            self.infl_rate = self.fitness_function(self.uni_objs_list)

    def __init__(self, iter_time, universe_num, limits_list, fitness_function):
        self.iter_time = iter_time
        self.universe_num = universe_num
        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize the universes
        self.universes_list = [self.Universe(self.limits_list, self.fitness_function) for i in range(self.universe_num)]

        # Initialize the global best universe
        self.global_best_universe = self.Universe(limits_list = self.limits_list, fitness_function = fitness_function)
        self.global_best_universe.infl_rate = float('inf')

    def search_best_infl_rate(self):
        self.sorted_universes_list = sorted(self.universes_list, key=lambda uni: uni.infl_rate, reverse=False)
        current_best_universe = self.sorted_universes_list[0]
        if current_best_universe.infl_rate < self.global_best_universe.infl_rate:
            self.global_best_universe = copy.deepcopy(current_best_universe)
        self.normalized_uni_infl_rate = normalize([universe.infl_rate for universe in self.sorted_universes_list])

    def roulette_wheel_select_white_hole(self):
        # Stick to the original thought, white hole with higher inflation rate
        sum_normalized_uni_infl_rate = sum(self.normalized_uni_infl_rate)
        random_pointer = random.uniform(0, sum_normalized_uni_infl_rate)
        current_pointer = 0.0
        for index, uni_infl_rate in enumerate(self.normalized_uni_infl_rate):
            current_pointer += uni_infl_rate
            if current_pointer > random_pointer:
                return index

    def inflate(self):
        current_best_universes_list = []
        for iter in range(self.iter_time):
            # WEP: wormhole exist probability, eq 3.3
            WEP = 0.2 + iter * (1 - 0.2) / self.iter_time
            # TDR: Travelling distance rate, eq 3.4
            TDR = 1 - pow(iter, 1/6) / pow(self.iter_time, 1/6)
            self.search_best_infl_rate()
            current_best_universes_list.append(self.global_best_universe)
            for uni_index in range(self.universe_num):
                for obj_index in range(len(self.limits_list)):
                    # Exploration：For each dimension, we explore first. Select a bad choice (white hole with high inflation rate) to sabotage the good ones for more wider search space
                    r1 = random.random()
                    # print('r1 = ',r1)
                    if r1 < self.normalized_uni_infl_rate[uni_index]:
                        white_hole_index = self.roulette_wheel_select_white_hole()
                        self.sorted_universes_list[uni_index].uni_objs_list[obj_index] = self.sorted_universes_list[white_hole_index].uni_objs_list[obj_index]
                    # Exploitation: make each universe similar to the best one. Along with the increase of WEP, its chance of moving towards the best universe is greater
                    r2 = random.random()
                    # print('r2 = ', r2)
                    if r2 < WEP:
                        r3 = random.random()
                        r4 = random.random()
                        # print('r3 =',r3, 'r4 =', r4)
                        boundary_min = self.limits_list[obj_index][0]
                        boundary_max = self.limits_list[obj_index][1]
                        if r3 < 0.5:
                            self.sorted_universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index] + TDR * (boundary_max - boundary_min) * r4
                        else:
                            self.sorted_universes_list[uni_index].uni_objs_list[obj_index] = self.global_best_universe.uni_objs_list[obj_index] - TDR * (boundary_max - boundary_min) * r4
            # After the current iteration, update the inflation rate
            for uni_index in range(self.universe_num):
                self.sorted_universes_list[uni_index].update()
        return current_best_universes_list

if __name__ == '__main__':
    iter_time = 500
    universe_num = 600
    dim = 30

    f1_limits_list = [[-100, 100] for i in range(dim)]
    from GA_pack.fittness_functions.f1 import f1
    f1_fitness_function = f1

    mvo = MVO_0(iter_time, universe_num, limits_list = f1_limits_list, fitness_function = f1_fitness_function)
    current_best_universes_list = mvo.inflate()
    global_best_univese = mvo.global_best_universe
    print('Best position:', global_best_univese.uni_objs_list)
    print('Found minimum of the f1 function', global_best_univese.infl_rate)

    # Draw the best universe in each iteration.
    iter_list = [i for i in range(iter_time)]
    infl_rate_list = [universe.infl_rate for universe in current_best_universes_list]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line1, = ax.plot(iter_list, infl_rate_list, label='Iteration {0}\nUniverse number {1}\nDimension {2}'.format(iter_time, universe_num, dim))
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    ax.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Inflation rate')
    plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
    plt.show()
