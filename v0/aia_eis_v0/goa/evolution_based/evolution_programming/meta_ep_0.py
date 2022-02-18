import copy
import math
import random

class EP:
    class Entity:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in limits_list]

            # Set the value boundary for sigma, Sigma is variances = (standard deviations) ** 2
            self.sigma_abs_ceil = 3
            # random.gauss(mu, sigma)  # 随机生成符合高斯分布的随机数，mu,sigma为高斯分布的两个参数
            self.sigma_list = [random.uniform(0, self.sigma_abs_ceil) for i in range(len(limits_list))]

            self.fitness = fitness_function(self.x_list)
            self.q_score = 0

        def update(self):
            # Restrain the x in its boundary
            for i in range(len(self.limits_list)):
                if self.x_list[i] > self.limits_list[i][1]:
                    self.x_list[i] = self.limits_list[i][1]
                if self.x_list[i] < self.limits_list[i][0]:
                    self.x_list[i] = self.limits_list[i][0]
            # Restrain sigma in its boundary
            for s_index, sigma in enumerate(self.sigma_list):
                if abs(sigma) > self.sigma_abs_ceil:
                    # self.sigma_list[s_index] = (sigma / abs(sigma)) * self.sigma_abs_ceil
                    self.sigma_list[s_index] = random.uniform(0, self.sigma_abs_ceil)
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, entity_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.entity_num = entity_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        # Initialize entities
        self.entities_list = [self.Entity(limits_list, fitness_function) for i in range(entity_num)]
        # Initialize global best entity
        self.global_best_entity = self.Entity(limits_list, fitness_function)
        self.global_best_entity.fitness = float('inf')

        # Q-selection setting
        self.q_len = int(self.entity_num / 10)

    def evolve(self):
        current_best_entity_list = []
        global_best_entity_list = []
        for iter in range(self.iter_num):
            # Get the current&global best entity
            current_best_entity = sorted(self.entities_list, key=lambda entity: entity.fitness, reverse=False)[0]
            # print('Current best x:', current_best_entity.x_list)
            # print('Current best fitness:', current_best_entity.fitness)
            if self.global_best_entity.fitness > current_best_entity.fitness:
                self.global_best_entity = copy.deepcopy(current_best_entity)
            current_best_entity_list.append(current_best_entity)
            global_best_entity_list.append(self.global_best_entity)

            father_child_entities_list = copy.deepcopy(self.entities_list)
            for father_entity in self.entities_list:
                child_entity = self.Entity(self.limits_list, self.fitness_function)
                child_x_list = []
                child_sigma_list = []
                for f_x, f_sigma in zip(father_entity.x_list, father_entity.sigma_list):
                    # Mutation
                    # c_x = f_x + f_sigma * random.gauss(mu=0, sigma=1)
                    # c_sigma = f_sigma * (1 + 0.2 * random.gauss(mu=0, sigma=1))
                    c_sigma = f_sigma + math.sqrt(0.2 * f_sigma) * random.gauss(mu=0, sigma=1)
                    if c_sigma <= 0:
                        c_sigma = 0.001
                    c_x = f_x + math.sqrt(f_sigma) * random.gauss(mu=0, sigma=1)
                    child_x_list.append(c_x)
                    child_sigma_list.append(c_sigma)
                child_entity.x_list = copy.deepcopy(child_x_list)
                child_entity.sigma_list = copy.deepcopy(child_sigma_list)
                child_entity.update()
                father_child_entities_list.append(child_entity)
            # Select Q entities randomly
            q_entities_list = [father_child_entities_list[i] for i in random.sample(range(len(father_child_entities_list)), self.q_len)]

            # Score every entity
            for en_index in range(len(father_child_entities_list)):
                for q_entity in q_entities_list:
                    if father_child_entities_list[en_index].fitness < q_entity.fitness:
                        father_child_entities_list[en_index].q_score += 1

            # 1-Rank every entity by their score
            # 2-Select the first N entities
            # 3-Replace entities_list with the selected N entities
            # sorted(reverse=False)--Ascending order
            # sorted(reverse=True)--Descending order
            self.entities_list = sorted(father_child_entities_list, key=lambda entity: entity.q_score, reverse=True)[:len(self.entities_list)]

            # Set the score as 0 in each entity
            for en_index in range(len(self.entities_list)):
                self.entities_list[en_index].q_score = 0
        return current_best_entity_list, global_best_entity_list

if __name__ == '__main__':
    iter_num = 500
    entity_num = 200
    dim = 16

    f1_limits_list = [[-100, 100] for i in range(dim)]
    from GA_pack.fittness_functions.f1 import f1

    f1_fitness_function = f1
    ep = EP(iter_num, entity_num, f1_limits_list, f1_fitness_function)
    current_best_entity_list, global_best_entity_list = ep.evolve()
    print('Best entity position:', ep.global_best_entity.x_list)
    print('Best Fitness:', ep.global_best_entity.fitness)

    # Draw the best entity in each iteration.
    iter_list = [i for i in range(iter_num)]
    cur_fitness_list = [entity.fitness for entity in current_best_entity_list]
    cur_global_fitness_list = [entity.fitness for entity in global_best_entity_list]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, entity_num, dim))
    line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    ax.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Error rate')
    plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
    plt.show()