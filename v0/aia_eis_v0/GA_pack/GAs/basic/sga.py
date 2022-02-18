from GA_pack.fittness_functions import fitness_pack
from GA_pack.genetic_material import DNA
from GA_pack.operators.selection import proportion_selector
from GA_pack.operators.crossover import crossover_operator
from GA_pack.operators.mutation import mutation_operator
from parameters.paras_wrapper import para_serializer
import numpy as np
import random
import time
import shelve

class SGA:
    """
    Refer:
        Book:
            Book0: 遗传算法原理及应用
        Paper:
    Adjustable parameters:
        casual:
            number of search agents
            number of iteration
        unique:
    Attention:
    Version:
        0
            This version use binary to code float variable. It is not convenient for fitting EIS ECM's paras
    """
    def __init__(self, paras_path, fitness_type, dna_num, max_epoch, p_crossover, p_mutation):
        """
        :param
            dna_num: 种群的数量,偶数，即DNA的数量
            max_epoch: 当没有明确的遗传终止标准的时候，达到最大进化代数（默认1000）的时候终止程序
            p_crossover: 基因的交叉概率
            p_mutation: 基因发生变异的概率
        """
        self.dna_num = dna_num
        self.max_epoch = max_epoch

        self.p_crossover = p_crossover
        self.p_mutation = p_mutation

        self.fitness_type = fitness_type
        self.gene_range_pair_tuple_list, self.precision_list = para_serializer(paras_path).read_raw_para()

    def DNAs_random_initialize(self):
        DNAs_list = []
        for i in range(self.dna_num):
            gene_num_list = []
            for gene_range_pair_tuple in self.gene_range_pair_tuple_list:
                gene_num_list.append(random.uniform(gene_range_pair_tuple[0], gene_range_pair_tuple[1]))
            DNAs_list.append(DNA.DNA(gene_num_list=gene_num_list, gene_range_pair_tuple_list=self.gene_range_pair_tuple_list, precision_list=self.precision_list).join_binary_code())
        return DNAs_list

    def get_fittness(self, DNAs_list):
        """
        把适应度统一转化为求最大值
        :param fitness_type:
            'rosenbrock'(rosenbrock函数的适应度);
            'eis'(将来要设置EIS专门的适应度计算/测试函数);
        :return: DNAs_fitness_list = [DNA0_fitness(float), DNA1_fitness, ...]
        """
        DNAs_fitness_list = fitness_pack.fitness_pack(DNAs_list, self.fitness_type, self.gene_range_pair_tuple_list,self.precision_list).get_fitness()
        return DNAs_fitness_list

    def select(self, DNAs_list, fitness_list, select_type):
        selected_DNAs_list = []
        if select_type == 'proportion':
            selected_DNAs_list = proportion_selector.proportion_selector(DNAs_list, fitness_list).select()
        elif select_type == '':
            pass
        return selected_DNAs_list

    def crossover(self, selected_DNAs_list, crossover_type):
        crossed_DNAs_list = crossover_operator.crossover(selected_DNAs_list, crossover_type, self.p_crossover).cross()
        return crossed_DNAs_list

    def mutation(self, crossed_DNAs_list, mutation_type):
        mutated_DNAs_list = mutation_operator.mutation_operators(crossed_DNAs_list, mutation_type, self.p_mutation).mutate()
        return mutated_DNAs_list

    def evolution(self, relate_res_folder_path):
        terminated_condition = True
        DNAs_list = self.DNAs_random_initialize()
        DNAs_fitness_list = self.get_fittness(DNAs_list)
        epoch = 0
        # 保存每次进化得到的个体的信息（进化代数，最优个体（变量值，适应度），变量的值-二进制转浮点数，适应度）
        # 获取当前时间的字符串--'19_01_27_22_30'
        time_str = time.strftime('%y_%m_%d_%H_%M', time.localtime(time.time()))
        db_name = relate_res_folder_path + '/sga_' + time_str
        while terminated_condition:
            self.serialize(db_name, epoch, DNAs_list, DNAs_fitness_list)
            selected_DNAs_list = self.select(DNAs_list, DNAs_fitness_list, select_type='proportion')
            crossed_DNAs_list = self.crossover(selected_DNAs_list, crossover_type='single')
            DNAs_list = self.mutation(crossed_DNAs_list, mutation_type='simple')
            DNAs_fitness_list = self.get_fittness(DNAs_list)
            epoch += 1
            terminated_condition = self.termination_judge(DNAs_list, DNAs_fitness_list, epoch, min_MSE_threshold=1)

    def termination_judge(self, DNAs_fitness_list, epoch, min_MSE_threshold):
        terminated_condition = True
        # 群体中出现一个满足要求的个体（适应度）,只有已知目标适应度的前提下才能衡量个体适应度和目标的差距
        # 群体中所有个体适应度的方差小于一个极小的阈值
        DNAs_fitness_arr = np.array(DNAs_fitness_list)
        DNAs_MSE = np.sqrt(np.sum((DNAs_fitness_arr - np.mean(DNAs_fitness_arr))**2) / len(DNAs_fitness_arr))
        if DNAs_MSE < min_MSE_threshold:
            terminated_condition = False
            print('The mean square of fitness of all DNAs is under {0}, stop the evolution'.format(min_MSE_threshold))
        # 进化代数不得超过认为规定的最大次数
        if epoch >= self.max_epoch:
            terminated_condition = False
            print('The epoch reachs the max epoch {0}, stop the evolution'.format(self.max_epoch))
        return terminated_condition

    def serialize(self, db_name, epoch, DNAs_list, fitness_list):
        with shelve.open(db_name, flag='c') as db_file:
            # shelve中的key规定是字符串
            db_file[str(epoch)] = {'dna':DNAs_list, 'fitness':fitness_list}

if __name__ == '__main__':
    # 用双变量的rosenbrock函数测试简单遗传算法的效果，已知fmax(2.048,-2.048)=3897.7342 fmax(2.048,2.048)=3905.9262
    sga = SGA(paras_path='../../../parameters/rosenbrock_paras.txt', fitness_type='rosenbrock',\
              dna_num=80, max_epoch=200, p_crossover=0.6, p_mutation=0.001)
    sga.evolution(relate_res_folder_path = '../../../results/sga_results/rosenbrock')