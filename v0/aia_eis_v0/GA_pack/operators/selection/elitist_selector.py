class elitist_selector():
    """
    按照最优保存策略设计选择算子
    """
    def __init__(self, DNAs_List, fitness_list):
        self.DNAs_List = DNAs_List
        self.fitness_list = fitness_list
    def select(self):
        pass