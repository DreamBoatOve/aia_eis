import math
import random
from matplotlib import pyplot as plt

class asp_tsp:
    class city_edge:
        def __init__(self, distance, init_pheromone):
            self.distance = distance
            self.pheromone = init_pheromone
    class Ant:
        """
        Ant
            1-在城市中随机选取一个城市City0作为起点，并添加到已访问的城市列表中
            2-计算City0与剩下所有未访问城市之间的转移概率，利用轮盘赌选择出下一站的城市
            3-重复第二步，直到蚂蚁走完所有城市
        """
        def __init__(self, alpha, beta, city_edge_list):
            self.alpha = alpha
            self.beta = beta
            # self.city_locations_list = city_locations_list
            self.visted_city_index_list = []
            self.unvisted_city_index_list = [i for i in range(len(city_edge_list[0]))]
            self.city_edge_list = city_edge_list

        def select_city(self, visted_city_index_list, unvisted_city_index_list):
            wheel = 0.0
            current_city_index = visted_city_index_list[-1]
            for u_c_i in unvisted_city_index_list:
                wheel += pow(self.city_edge_list[current_city_index][u_c_i].pheromone, self.alpha) * pow(1 / self.city_edge_list[current_city_index][u_c_i].distance, self.beta)
            random_pointer = random.uniform(0, wheel)
            # Start rolling the wheel
            current_pointer = 0.0
            for u_c_i in unvisted_city_index_list:
                current_pointer += pow(self.city_edge_list[current_city_index][u_c_i].pheromone, self.alpha) * pow(1 / self.city_edge_list[current_city_index][u_c_i].distance, self.beta)
                if current_pointer > random_pointer:
                    return u_c_i

        def travel(self):
            self.visted_city_index_list.append(self.unvisted_city_index_list.pop(random.randint(0, len(self.unvisted_city_index_list)-1)))
            while len(self.unvisted_city_index_list) > 0:
                selected_city_index = self.select_city(self.visted_city_index_list, self.unvisted_city_index_list)
                self.visted_city_index_list.append(selected_city_index)
                self.unvisted_city_index_list.remove(selected_city_index)
            # Calculate the travel distance
            distance = 0.0
            for index, v_c_i in enumerate(self.visted_city_index_list[:-1]):
                distance += self.city_edge_list[v_c_i][self.visted_city_index_list[index+1]].distance
            # Add the distance between the Beginning and the end
            distance += self.city_edge_list[self.visted_city_index_list[0]][self.visted_city_index_list[-1]].distance
            return self.visted_city_index_list, distance

    def __init__(self, m, mode, alpha, beta, Q, iteration_time, city_locations_list, init_pheromone, evapor_rate=0.8, elitist_extra=0.5):
        """
        :param m: The number of ants
        :param mode: 'acs':basic mode, 'elitist': elitist mode, 'MaxMin': Max-min mode
        :param alpha: factor/weight of pheromone
        :param beta: factor/weight of distance
        :param Q: the amount of pheromone
        :param iteration_time:
        :param city_locations_list: the number of cities
        """
        self.m = m
        self.mode = mode
        self.alpha = alpha
        self.beta = beta
        self.Q = Q
        self.iteration_time = iteration_time
        self.city_locations_list = city_locations_list
        self.init_pheromone = init_pheromone
        self.evapor_rate = evapor_rate
        self.elitist_extra = elitist_extra

        self.city_edge_list = [[None] * len(self.city_locations_list) for i in range(len(self.city_locations_list))]
        # 计算不同城市间的路程距离
        for i in range(len(self.city_locations_list)):
            for j in range(i + 1, len(self.city_locations_list)):
                self.city_edge_list[i][j] = self.city_edge_list[j][i] = self.city_edge(math.sqrt(pow(self.city_locations_list[i][0] - self.city_locations_list[j][0],2) + pow(self.city_locations_list[i][1] - self.city_locations_list[j][1],2)), self.init_pheromone)
        self.global_best_route = None
        self.global_best_distance = float('inf')

    def acs(self):
        for t in range(self.iteration_time):
            ant_res_list = []
            for i in range(self.m):
                route, distance = self.Ant(self.alpha, self.beta, self.city_edge_list).travel()
                ant_res_list.append((route, distance))
                if distance < self.global_best_distance:
                    self.global_best_distance = distance
                    self.global_best_route = route
            # Update the pheromone on the edges
            for ant_res in ant_res_list:
                route = ant_res[0]
                distance = ant_res[1]
                added_pheromone = self.Q/distance
                for index, a in enumerate(route[:-1]):
                    b = route[index + 1]
                    self.city_edge_list[a][b].pheromone += added_pheromone
                    self.city_edge_list[b][a].pheromone += added_pheromone
            # Evaporate the pheromone on all the path
            for i in range(len(self.city_edge_list[0])):
                for j in range(i+1, len(self.city_edge_list[0])):
                    self.city_edge_list[i][j].pheromone *= (1 - self.evapor_rate)
                    self.city_edge_list[j][i].pheromone *= (1 - self.evapor_rate)

    def elitist(self):
        # Add extra pheromone on the path taken by the current best ant
        for t in range(self.iteration_time):
            ant_res_list = []
            current_best_distance = float('inf')
            current_best_route = None
            for i in range(self.m):
                route, distance = self.Ant(self.alpha, self.beta, self.city_edge_list).travel()
                ant_res_list.append((route, distance))
                # Find the best current ant
                if distance < current_best_distance:
                    current_best_distance = distance
                    current_best_route = route
                if distance < self.global_best_distance:
                    self.global_best_distance = distance
                    self.global_best_route = route
            #  Update the pheromone and add extra pheromone from the best current ant
            for ant_res in ant_res_list:
                route = ant_res[0]
                distance = ant_res[1]
                added_pheromone = self.Q / distance

                # elitist_mark : the mark of elitist: 1 == Yes, 0 == No
                elitist_mark = 0
                if distance == current_best_distance:
                    elitist_mark = 1
                for index, a in enumerate(route[:-1]):
                    b = route[index + 1]
                    if elitist_mark == 1:
                        self.city_edge_list[a][b].pheromone += (1 + self.elitist_extra) * added_pheromone
                        self.city_edge_list[b][a].pheromone += (1 + self.elitist_extra) * added_pheromone
                    else:
                        self.city_edge_list[a][b].pheromone += added_pheromone
                        self.city_edge_list[b][a].pheromone += added_pheromone

            # Evaporate the pheromone on the edges
            for i in range(len(self.city_edge_list[0])):
                for j in range(i+1, len(self.city_edge_list[0])):
                    self.city_edge_list[i][j].pheromone *= (1 - self.evapor_rate)
                    self.city_edge_list[j][i].pheromone *= (1 - self.evapor_rate)

    def max_min(self, best_chance = 0.05):
        # According to the paper "MAX-MIN Ant system", Pbest = best_chance = 0.05
        for t in range(self.iteration_time):
            ant_res_list = []
            current_best_distance = float('inf')
            current_best_route = None

            pheromone_max = None
            pheromone_min = None

            for i in range(self.m):
                route, distance = self.Ant(self.alpha, self.beta, self.city_edge_list).travel()
                ant_res_list.append((route, distance))
                # Find the best current ant
                if distance < current_best_distance:
                    current_best_distance = distance
                    current_best_route = route
                if distance < self.global_best_distance:
                    self.global_best_distance = distance
                    self.global_best_route = route

            # Update the pheromone on the edges taken by the current best ant or the global best ant (a time / 3 iteration)
            if (t % 3) == 0:
                # Use the global best ant
                for index, city_index in enumerate(self.global_best_route[:-1]):
                    next_city_index = self.global_best_route[index + 1]
                    self.city_edge_list[city_index][next_city_index].pheromone += self.Q / self.global_best_distance
                    self.city_edge_list[next_city_index][city_index].pheromone += self.Q / self.global_best_distance
            else:
                # Use the current best ant
                for index, city_index in enumerate(current_best_route[:-1]):
                    next_city_index = current_best_route[index + 1]
                    self.city_edge_list[city_index][next_city_index].pheromone += self.Q / current_best_distance
                    self.city_edge_list[next_city_index][city_index].pheromone += self.Q / current_best_distance

            # Evaporate the pheromone on all the edges
            for i in range(len(self.city_edge_list[0])):
                for j in range(i+1, len(self.city_edge_list[0])):
                    self.city_edge_list[i][j].pheromone *= (1 - self.evapor_rate)
                    self.city_edge_list[j][i].pheromone *= (1 - self.evapor_rate)
                    if t >= 1:
                        pheromone_max = (1 / (1 - self.evapor_rate)) * (1 / current_best_distance)
                        pheromone_min = (pheromone_max * (1 - pow(best_chance, 1 / self.m))) / (((self.m / 2) - 1) * pow(best_chance, 1 / self.m))
                        if self.city_edge_list[i][j].pheromone > pheromone_max:
                            self.city_edge_list[i][j].pheromone = pheromone_max
                        elif self.city_edge_list[i][j].pheromone < pheromone_min:
                            self.city_edge_list[i][j].pheromone = pheromone_min

                        if self.city_edge_list[j][i].pheromone > pheromone_max:
                            self.city_edge_list[j][i].pheromone = pheromone_max
                        elif self.city_edge_list[j][i].pheromone < pheromone_min:
                            self.city_edge_list[j][i].pheromone = pheromone_min

    def run(self):
        if self.mode == 'acs':
            self.acs()
        elif self.mode == 'elitist':
            self.elitist()
        elif self.mode == 'maxmin':
            self.max_min()
        return self.global_best_route, self.global_best_distance
    def plot(self, line_width=1, point_radius=math.sqrt(2.0), annotation_size=8, dpi=120, save=True, name=None):
        x = [self.city_locations_list[i][0] for i in self.global_best_route]
        x.append(x[0])
        y = [self.city_locations_list[i][1] for i in self.global_best_route]
        y.append(y[0])
        plt.plot(x, y, linewidth=line_width)
        plt.scatter(x, y, s=math.pi * (point_radius ** 2.0))
        plt.title(self.mode)
        # for i in self.global_best_route:
        #     plt.annotate(self.labels[i], self.nodes[i], size=annotation_size)
        #     plt.annotate(str(i), i, size=annotation_size)
        if save:
            if name is None:
                name = '{0}_tour.png'.format(self.mode)
            plt.savefig(name, dpi=dpi)
        plt.show()
        plt.gcf().clear()
if __name__ == '__main__':
    random.seed(0)
    city_locations_list = [(random.uniform(-400, 400), random.uniform(-400, 400)) for i in range(15)]
    # print(city_locations_list)
    m = 5
    iteration_time = 50

    # acs = asp_tsp(m, mode='acs', alpha=1.0, beta=3.0, Q=5, iteration_time=iteration_time, city_locations_list=city_locations_list, init_pheromone=1.0)
    # r, d = acs.run()
    # print(r, round(d,2))
    # acs.plot()

    # acs_elitist = asp_tsp(m, mode='elitist', alpha=1.0, beta=3.0, Q=5, iteration_time=iteration_time, city_locations_list=city_locations_list, init_pheromone=1.0)
    # r, d = acs_elitist.run()
    # print(r, round(d,2))
    # acs_elitist.plot()

    acs_max_min = asp_tsp(m, mode='maxmin', alpha=1.0, beta=3.0, Q=5, iteration_time=iteration_time, city_locations_list=city_locations_list, init_pheromone=1.0)
    r, d = acs_max_min.run()
    print(r, round(d,2))
    acs_max_min.plot()