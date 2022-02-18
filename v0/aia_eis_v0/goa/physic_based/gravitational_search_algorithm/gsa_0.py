import copy
import math
import random
"""
这是第0版代码，忘记考虑Kbest了，而且算法需要迭代很多次或很多个体才能有合理的效果
"""
class GSA:
    class Mass:
        def __init__(self, limits_list, fitness_function):
            self.limits_list = limits_list
            self.fitness_function = fitness_function

            self.x_list = [random.uniform(limit[0], limit[1]) for limit in self.limits_list]
            self.fitness = fitness_function(self.x_list)

            self.v_list = [0.0 for i in range(len(self.limits_list))]

        def update(self):
            self.x_list = [x + v for x, v in zip(self.x_list, self.v_list)]
            for x_i, x in enumerate(self.x_list):
                if (x < self.limits_list[x_i][0]) or (x > self.limits_list[x_i][1]):
                    self.x_list[x_i] = random.uniform(self.limits_list[x_i][0], self.limits_list[x_i][1])
            self.fitness = self.fitness_function(self.x_list)

    def __init__(self, iter_num, mass_num, limits_list, fitness_function):
        self.iter_num = iter_num
        self.mass_num = mass_num

        self.limits_list = limits_list
        self.fitness_function = fitness_function

        self.mass_list = [self.Mass(self.limits_list, self.fitness_function) for i in range(self.mass_num)]

        self.global_best_mass = self.Mass(self.limits_list, self.fitness_function)
        self.global_best_mass.fitness = float('inf')

    def evolve(self):
        cur_best_mass_list = []
        global_best_mass_list = []
        for iter_index in range(self.iter_num):
            # Select the best mass
            sorted_mass_list = sorted(self.mass_list, key= lambda mass : mass.fitness, reverse=False)
            cur_best_mass = sorted_mass_list[0]
            cur_worst_mass = sorted_mass_list[-1]
            if cur_best_mass.fitness < self.global_best_mass.fitness:
                print('{} time: current best fitness {}'.format(iter_index, cur_best_mass.fitness))
                self.global_best_mass = cur_best_mass
            cur_best_mass_list.append(cur_best_mass)
            print('{} time: current global best fitness {}'.format(iter_index, self.global_best_mass.fitness))
            global_best_mass_list.append(self.global_best_mass)

            # relative_fitness对应公式15中的mi
            try:
                relative_fitness_list = [(mass.fitness - cur_worst_mass.fitness)/(cur_best_mass.fitness - cur_worst_mass.fitness) for mass in self.mass_list]
            except ZeroDivisionError as e:
                print(e)
                print('Best x = {} fitness = {}'.format(cur_best_mass.x_list, cur_best_mass.fitness))
                print('Worst x = {} fitness = {}'.format(cur_worst_mass.x_list, cur_worst_mass.fitness))
            # relative_Fitness对应公式16中的Mi
            relative_Fitness_list = [r_f / sum(relative_fitness_list) for r_f in relative_fitness_list]

            """
            健康值最小的个体，相对健康值为0，如果真的处于很边缘的边界，可能无论迭代多少次都不会更新
            在此处将其健康值 = 相对健康值倒数第二小/2
            """
            # Get the index of Mass with second small relative fitness
            sec_small_r_F = sorted(relative_Fitness_list, reverse=False)[1]
            relative_Fitness_list = [r_F if r_F != 0.0 else sec_small_r_F / 2 for r_F in relative_Fitness_list]

            # Calculate the Force and Acceleration
            # 重力常数的初始值，以及迭代公式和相关计算系数参考公式28
            G0 = 100
            alpha = 20
            G = G0 * math.exp(- alpha * iter_index / self.iter_num)
            c = 0.0001
            tmp_mass_list = []
            for m_i, mass_i in enumerate(self.mass_list):
                # Calculate the F in each dimension
                force_list = []
                for m_j, mass_j in enumerate(self.mass_list):
                    if m_i != m_j:
                        # Calculate the distance between mass_i and mass_j
                        d = math.sqrt(sum([(x1 - x2) ** 2 for x1, x2 in zip(mass_i.x_list, mass_j.x_list)]))
                        random_force_list = []
                        for dim_i in range(len(self.limits_list)):
                            # Eq 7 in GSA paper
                            force = G * relative_Fitness_list[m_j] * relative_Fitness_list[m_i] * (mass_j.x_list[dim_i] - mass_i.x_list[dim_i]) / (d + c)
                            # force乘一个随机数，引入随机扰动
                            random_force_list.append(force * random.random())
                        force_list.append(random_force_list)
                # Calculate the a in each dimension
                a_list = []
                for i in range(len(self.limits_list)):
                    f_sum = 0.0
                    for f_l in force_list:
                        f_sum += f_l[i]
                    try:
                        # 每一轮最差的结果，其【相对】健康值必定为0，就会出现【除0错误】，此时用一个很小的值代替
                        a = f_sum / relative_Fitness_list[m_i]
                        a_list.append(a)
                    except ZeroDivisionError as e:
                        a = 0.0
                        a_list.append(a)
                # Update Velocity and Position
                tmp_v_list = [random.random() * v + a for v, a in zip(mass_i.v_list, a_list)]
                tmp_mass = copy.deepcopy(mass_i)
                tmp_mass.v_list = tmp_v_list
                tmp_mass.update()
                tmp_mass_list.append(tmp_mass)
            self.mass_list = tmp_mass_list
        return cur_best_mass_list, global_best_mass_list

if __name__ == '__main__':
    iter_num = 500
    mass_num = 5
    dim = 5

    f1_limits_list = [[-100, 100] for i in range(dim)]
    from GA_pack.fittness_functions.f1 import f1

    f1_fitness_function = f1
    gsa = GSA(iter_num, mass_num, f1_limits_list, f1_fitness_function)
    cur_best_mass_list, global_best_mass_list = gsa.evolve()
    print('Best entity position:', gsa.global_best_mass.x_list)
    print('Fitness:', gsa.global_best_mass.fitness)

    # Draw the best entity in each iteration.
    iter_list = [i for i in range(iter_num)]
    cur_fitness_list = [entity.fitness for entity in cur_best_mass_list]
    cur_global_fitness_list = [entity.fitness for entity in global_best_mass_list]

    import matplotlib.pyplot as plt
    fig, ax = plt.subplots()
    line1, = ax.plot(iter_list, cur_fitness_list, label='Current Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, mass_num, dim))
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    line2, = ax.plot(iter_list, cur_global_fitness_list, label='Current Global Iteration {0}\nentity number {1}\nDimension {2}'.format(iter_num, mass_num, dim))
    line2.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    ax.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Error rate')
    plt.title('Search the minimum of f1 = sum(Xi ^ 2)')
    plt.show()