from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict
from global_optimizations.swarm_based.grey_wolf_optimizer.gwo_0 import GWO_EIS
from global_optimizations.GOA_criterions import goa_rel_std_err

from utils.visualize_utils.goa_plots import fitness_v_iter
"""
Steps
    1-Import each ECM_Simulation results
    2-Test each simulation EIS data on GWO_EIS
"""
# 1-Import GOA-ECM_Simulation-Dataset
# Import ECM-2 Simulation
ecm2_sim_data_dict = load_sim_ecm_para_config_dict(ecm_num=2, file_path='../../../datasets/goa_datasets/simulated')

# 2-Test each simulation EIS data on GWO_EIS
# Initialize GWO_EIS
iter_time = 10000
wolf_num = 100
gwo_eis = GWO_EIS(exp_data_dict = ecm2_sim_data_dict, iter_time=iter_time, wolf_num=wolf_num)
current_best_wolf_list, iter, chi_squared = gwo_eis.hunt()
print('Real Elements', ecm2_sim_data_dict['para'])
print('Iteration:{0}\nEntity:{1}\nFitting:'.format(iter, wolf_num), current_best_wolf_list[-1].position_list)

x_lists_list = [w.position_list for w in current_best_wolf_list]
rel_std_err_list = goa_rel_std_err(x_lists_list)
print('Rel. Std. Err:', rel_std_err_list)
print('Chi-Squared', chi_squared)

print('Best Fitness:', current_best_wolf_list[-1].fitness)
fitness_list = [wolf.fitness for wolf in current_best_wolf_list]
fitness_v_iter(fitness_list, entity_num=wolf_num, alg_name='GWO')