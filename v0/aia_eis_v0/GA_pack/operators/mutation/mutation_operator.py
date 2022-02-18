import random

class mutation_operators():
    def __init__(self, crossed_DNAs_list, mutation_type, p_mutation):
        self.crossed_DNAs_list = crossed_DNAs_list
        self.mutation_type = mutation_type
        self.p_mutation = p_mutation
    def simple_mutation(self):
        for DNA_index in range(len(self.crossed_DNAs_list)):
            for gene_index in range(len(self.crossed_DNAs_list[0])):
                if random.random() < self.p_mutation:
                    gene_mutation_index = random.randint(0, len(self.crossed_DNAs_list[0][gene_index])-1)
                    # Python “&”、“|”、“^”按位逻辑运算到底是咋回事 https://blog.csdn.net/wxy_csdn_world/article/details/80759915
                    self.crossed_DNAs_list[DNA_index][gene_index][gene_mutation_index] = self.crossed_DNAs_list[DNA_index][gene_index][gene_mutation_index]^1
        return self.crossed_DNAs_list
    def uniform_mutation(self):
        pass
    def mutate(self):
        mutated_DNAs_list = []
        if self.mutation_type == 'simple':
            mutated_DNAs_list = self.simple_mutation()
        elif self.mutation_type == 'uniform':
            mutated_DNAs_list = self.uniform_mutation()
        return mutated_DNAs_list