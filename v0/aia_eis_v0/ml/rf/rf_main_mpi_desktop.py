import sys
sys.path.append('../../')

from ml_sl.rf.rf_0 import RF, save_random_forest
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from utils.file_utils.filename_utils import get_date_prefix
from ml_sl.ml_critrions import cal_accuracy, cal_kappa

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    all_para_list = []
    for i in range(10):
        for tree_num in [200, 300, 350, 400, 450]:
            all_para_list.append([i, tree_num])

    para_num = len(all_para_list)
    comm.send(all_para_list[int(para_num / 7) * 0 : int(para_num / 7) * 1], dest=1)
    comm.send(all_para_list[int(para_num / 7) * 1 : int(para_num / 7) * 2], dest=2)
    comm.send(all_para_list[int(para_num / 7) * 2 : int(para_num / 7) * 3], dest=3)
    comm.send(all_para_list[int(para_num / 7) * 3 : int(para_num / 7) * 4], dest=4)
    comm.send(all_para_list[int(para_num / 7) * 4 : int(para_num / 7) * 5], dest=5)
    comm.send(all_para_list[int(para_num / 7) * 5 : int(para_num / 7) * 6], dest=6)
    comm.send(all_para_list[int(para_num / 7) * 6 : para_num], dest=7)

else:
    # Accept parameter from Process_0
    print('Process-{0} got parameters and starts working'.format(rank))
    rec_para_list = comm.recv(source=0)

    label_list = [2, 4, 5, 6, 7, 8, 9]
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    counter = 0
    rf_final_res_fn = get_date_prefix() + 'rf_final_res.txt'
    for para in rec_para_list:
        i, tree_num = para
        rf = RF(unlabeled_dataset_list = test_data_list,
                labeled_dataset_list = training_dataset + validation_dataset,
                label_list = label_list, tree_num=tree_num)
        rf.create_forest()
        sample_label_prob_dict_list = rf.classify()

        # 1.2- 保存树及其性能
        # create a name for rf, like date_rf_tree=num_pickle_i.file
        rf_model_name = get_date_prefix() + 'rf_tree=' + str(tree_num) + '_pickle_' + str(i) + '.file'

        # calculate accuracy and kappa
        acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
        kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

        # record rf and its name, tree_num, accuracy, and kappa
        save_random_forest(random_forest=rf, filename=rf_model_name, filepath='./')

        with open(rf_final_res_fn, 'a+') as file:
            line_str = ','.join([rf_model_name, str(tree_num), str(i), str(acc), str(kappa), str(acc + kappa)]) + '\n'
            file.write(line_str)
        counter += 1
        print('Process-{0}: Finished {1}, {2} left'.format(rank, counter, len(rec_para_list) - counter))

"""
Plan
    1-Choose the number of used processes
        One master to assign parameter to 7 slaves, there is 8 processes
    2-Run the file
        Use 'cmd' in the folder containing this file, then type the following command:
            mpiexec -n 8 python rf_main_mpi_desktop.py
"""