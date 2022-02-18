import matplotlib.pyplot as plt

"""
This module contains plot utils for GOA
"""
def fitness_v_iter(fitness_list, entity_num, alg_name):
    iter_list = [i for i in range(len(fitness_list))]
    fig, ax = plt.subplots()
    line1, = ax.plot(iter_list, fitness_list,\
                     label='Iteration {0}\nEntity number {1}'.format(len(fitness_list), entity_num))
    line1.set_dashes([2, 2, 10, 2])  # 2pt line, 2pt break, 10pt line, 2pt break
    ax.legend()
    plt.xlabel('Iteration times')
    plt.ylabel('Fitness')
    plt.title('Search the minimum of EIS Weighted Square Error')
    plt.show()