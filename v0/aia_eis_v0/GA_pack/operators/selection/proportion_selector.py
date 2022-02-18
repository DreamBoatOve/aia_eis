import random
from itertools import accumulate
from bisect import bisect_left

class proportion_selector():
    """
    使用比例选择（轮盘赌选择），适应度所占总适应度的比例越大，被选中的概率越大
    """
    def __init__(self, DNAs_list, fitness_list):
        self.DNAs_list = DNAs_list
        self.fitness_list = fitness_list
    def select(self):
        min_fitness = min(self.fitness_list)
        fitness_list = [fitness - min_fitness for fitness in self.fitness_list]
        sum_fitness = sum(fitness_list)
        wheel = list(accumulate([fitness/sum_fitness for fitness in fitness_list]))
        selected_DNAs_list = []
        for i in range(len(self.fitness_list)):
            # 一个有趣的python排序模块：bisect:https://www.cnblogs.com/skydesign/archive/2011/09/02/2163592.html
            selected_DNA_index = bisect_left(wheel, random.random())
            selected_DNA = self.DNAs_list[selected_DNA_index]
            selected_DNAs_list.append(selected_DNA)
        return selected_DNAs_list