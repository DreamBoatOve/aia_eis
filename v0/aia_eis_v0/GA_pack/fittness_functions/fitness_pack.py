from GA_pack.fittness_functions import rosenbrock_fitness
from GA_pack.encoders import binary_en_de_coder

class fitness_pack():
    def __init__(self, DNAs_list, fitness_type, gene_range_pair_tuple_list, precision_list):
        """
        :param DNAs_list: [
                            DNA[
                                gene[0,1,0,1,0,1],
                                gene
                                ],
                            DNA,
                            ]
        :param fitness_type:.
            'rosenbrock'(rosenbrock函数的适应度),主要用于GA多目标拟合的测试;
            'eis'(将来要设置EIS专门的适应度计算/测试函数);
        """
        self.DNAs_list = DNAs_list
        self.fitness_type = fitness_type
        self.gene_range_pair_tuple_list = gene_range_pair_tuple_list
        self.precision_list = precision_list
    def get_fitness(self):
        fitness_list = []
        if self.fitness_type == 'rosenbrock':
            DNAs_num_list = []
            for DNA in self.DNAs_list:
                gene_num_list = []
                for gene_binary_list, range_pair, precision in zip(DNA, self.gene_range_pair_tuple_list, self.precision_list):
                    gene_num_list.append(binary_en_de_coder.binary_decoder(gene_binary_list, range_pair, precision).decode())
                DNAs_num_list.append(gene_num_list)
                fitness_list.append(rosenbrock_fitness.rosenbrock(gene_num_list[0], gene_num_list[1]).get_rosenbrock_fitness())
        elif self.fitness_type == 'eis':
            pass
        return fitness_list