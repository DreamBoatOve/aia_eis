import copy
import random

class crossover():
    def __init__(self, selected_DNAs_list, crossover_type, p_crossover):
        self.selected_DNAs_list = selected_DNAs_list
        self.crossover_type = crossover_type
        self.p_crossover = p_crossover
        self.couple_list = self.select_parents()
    def select_parents(self):
        couple_list = []
        selected_DNAs_list = copy.deepcopy(self.selected_DNAs_list)
        while len(selected_DNAs_list) > 0:
            # random.randint(a,b) a <= n <= b
            father_index = random.randint(0, len(selected_DNAs_list)-1)
            mother_index = random.randint(0, len(selected_DNAs_list)-1)
            if father_index == mother_index:
                continue
            else:
                father_DNA_list = selected_DNAs_list[father_index]
                mother_DNA_list = selected_DNAs_list[mother_index]
                # list根据索引删除元素用pop()
                # selected_DNAs_list.pop(father_index)
                # selected_DNAs_list.pop(mother_index)
                selected_DNAs_list.remove(father_DNA_list)
                selected_DNAs_list.remove(mother_DNA_list)
                couple_list.append([father_DNA_list, mother_DNA_list])
        return couple_list
    def single_point_crossover(self):
        new_DNA_list = []
        for father, mother in self.couple_list:
            child0_DNA_list = []
            child1_DNA_list = []
            for father_gene, mother_gene in zip(father, mother):
                if random.random() > self.p_crossover:
                    child0_DNA_list.append(father_gene)
                    child1_DNA_list.append(mother_gene)
                else:
                    random_single_point_crossover_index = random.randint(1, len(father_gene)-1)
                    child0_gene_list = father_gene[:random_single_point_crossover_index] + mother_gene[random_single_point_crossover_index:]
                    child1_gene_list = mother_gene[:random_single_point_crossover_index] + father_gene[random_single_point_crossover_index:]
                    child0_DNA_list.append(child0_gene_list)
                    child1_DNA_list.append(child1_gene_list)
            new_DNA_list.extend([child0_DNA_list, child1_DNA_list])
        return new_DNA_list
    def double_points_crossover(self):
        new_DNA_list = []
        for father, mother in self.couple_list:
            child0_DNA_list = []
            child1_DNA_list = []
            for father_gene, mother_gene in zip(father, mother):
                if random.random() > self.p_crossover:
                    child0_DNA_list.append(father_gene)
                    child1_DNA_list.append(mother_gene)
                else:
                    first_index = random.randint(1, len(father_gene) - 1)
                    second_index = random.randint(first_index + 1, len(father_gene) - 1)
                    child0_DNA_list = father_gene[:first_index] + mother_gene[first_index:second_index] + father_gene[second_index:]
                    child1_DNA_list = mother_gene[:first_index] + father_gene[first_index:second_index] + mother_gene[second_index:]
                    new_DNA_list.extend([child0_DNA_list, child1_DNA_list])
        return new_DNA_list
    def uniform_crossover(self):
        new_DNA_list = []
        for father, mother in self.couple_list:
            child0_DNA_list = []
            child1_DNA_list = []
            for father_gene, mother_gene in zip(father, mother):
                child0_gene_list = []
                child1_gene_list = []
                for f_ele, m_ele in zip(father_gene, mother_gene):
                    if random.random() > self.p_crossover:
                        child0_gene_list.append(f_ele)
                        child1_gene_list.append(m_ele)
                    else:
                        child0_gene_list.append(m_ele)
                        child1_gene_list.append(f_ele)
                child0_DNA_list.append(child0_gene_list)
                child1_DNA_list.append(child1_gene_list)
            new_DNA_list.extend([child0_DNA_list, child1_DNA_list])
        return new_DNA_list
    def cross(self):
        new_DNA_list = []
        if self.crossover_type == 'single':
            new_DNA_list = self.single_point_crossover()
        elif self.crossover_type == 'double':
            new_DNA_list = self.double_points_crossover()
        elif self.crossover_type == 'uniform':
            new_DNA_list = self.uniform_crossover()
        return new_DNA_list