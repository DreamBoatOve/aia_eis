from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict
from global_optimizations.swarm_based.particle_swarm_optimization.pso_0 import PSO_EIS
from global_optimizations.GOA_criterions import goa_rel_std_err

"""
Steps
    1-Import each ECM_Simulation results
    2-Test each simulation EIS data on PSO_EIS
"""
# 1-Import GOA-ECM_Simulation-Dataset
# Import ECM-2 Simulation
ecm2_sim_data_dict = load_sim_ecm_para_config_dict(ecm_num=2, file_path='../../../datasets/goa_datasets/simulated')

# 2-Test each simulation EIS data on GWO_EIS
# Initialize GWO_EIS
iter_time = 10000
particle_num = 50
pso_eis = PSO_EIS(exp_data_dict = ecm2_sim_data_dict, iter_time=iter_time, particle_num=particle_num)
cur_best_particle_list, global_best_particle_list, iter, chi_squared = pso_eis.search()
print('Real Elements', ecm2_sim_data_dict['para'])
print('Iteration:{0}\nEntity:{1}\nFitting:'.format(iter, particle_num), global_best_particle_list[-1].position_list)

x_lists_list = [pso.position_list for pso in global_best_particle_list]
rel_std_err_list = goa_rel_std_err(x_lists_list)
print('Rel. Std. Err:', rel_std_err_list)
print('Chi-Squared', chi_squared)