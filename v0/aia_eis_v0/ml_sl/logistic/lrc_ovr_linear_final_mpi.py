import os
import sys
"""
python解释器会把当前文件（模块）当作项目的根目录，以此来寻找其他的模块
使用sys.path调整解释器的寻找路径，以此文件为起点，调整路径到项目的真实根目录
"""
sys.path.append('../../')

from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa
from ml_sl.logistic.lrc_0 import LRC

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    # Process_0 assigns parameter to other processes
    para_list = []
    selected_para_list = [[3000, 1], [13000, 1], [9000, 1], [13000, 0.1], [13000, 100]]
    for selected_para in selected_para_list: # len = 5
        max_iter, alpha_init = selected_para
        for i in range(10): # len = 10
            para_list.append([max_iter, alpha_init, i])

    comm.send(para_list[int(len(para_list) / 15) * 0:int(len(para_list) / 15) * 1], dest=1)
    comm.send(para_list[int(len(para_list) / 15) * 1:int(len(para_list) / 15) * 2], dest=2)
    comm.send(para_list[int(len(para_list) / 15) * 2:int(len(para_list) / 15) * 3], dest=3)
    comm.send(para_list[int(len(para_list) / 15) * 3:int(len(para_list) / 15) * 4], dest=4)
    comm.send(para_list[int(len(para_list) / 15) * 4:int(len(para_list) / 15) * 5], dest=5)
    comm.send(para_list[int(len(para_list) / 15) * 5:int(len(para_list) / 15) * 6], dest=6)
    comm.send(para_list[int(len(para_list) / 15) * 6:int(len(para_list) / 15) * 7], dest=7)
    comm.send(para_list[int(len(para_list) / 15) * 7:int(len(para_list) / 15) * 8], dest=8)
    comm.send(para_list[int(len(para_list) / 15) * 8:int(len(para_list) / 15) * 9], dest=9)
    comm.send(para_list[int(len(para_list) / 15) * 9:int(len(para_list) / 15) * 10], dest=10)
    comm.send(para_list[int(len(para_list) / 15) * 10:int(len(para_list) / 15) * 11], dest=11)
    comm.send(para_list[int(len(para_list) / 15) * 11:int(len(para_list) / 15) * 12], dest=12)
    comm.send(para_list[int(len(para_list) / 15) * 12:int(len(para_list) / 15) * 13], dest=13)
    comm.send(para_list[int(len(para_list) / 15) * 13:int(len(para_list) / 15) * 14], dest=14)
    comm.send(para_list[int(len(para_list) / 15) * 14:len(para_list)], dest=15)

else:
    # Accept parameter from Process_0
    print('Process-{0} got parameters and starts working'.format(rank))
    rec_para_list = comm.recv(source=0)
    for para in rec_para_list:
        max_iter, alpha_init, i = para

        ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
        training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)

        tr_va_dataset = training_dataset + validation_dataset
        te_label_list, te_data_list = split_labeled_dataset_list(test_dataset)
        label_list = [2, 4, 5, 6, 7, 8, 9]

        lrc = LRC(alpha=alpha_init, max_iter=max_iter, \
                  unlabeled_dataset_list=te_data_list, labeled_dataset_list=tr_va_dataset, \
                  label_list=label_list)
        lrc_classifer_dict_pickle_filename = get_date_prefix() + 'lrc_ovr_linear_final_iter='+ str(max_iter)\
                                             + '_alpha_init=' + str(alpha_init) + '_classifer_dict_pickle_'\
                                             + str(i) + '.file'
        lrc.create_lrc_classifer_dict_ovr(lrc_classifer_dict_pickle_filename)
        lrc_sample_label_prob_dict_list = lrc.classify_ovr(lrc_classifer_dict_pickle_filename)
        lrc_acc = cal_accuracy(lrc_sample_label_prob_dict_list, te_label_list)
        lrc_kappa = cal_kappa(lrc_sample_label_prob_dict_list, te_label_list)

        res_file_name = get_date_prefix() + 'ovr_linear_final.txt'
        with open(res_file_name, 'a+') as file:
            line_str = lrc_classifer_dict_pickle_filename + ',' + str(max_iter) + ',' + str(alpha_init) + ',' + str(i)\
                       + ',' + str(lrc_acc) + ',' + str(lrc_kappa) + '\n'
            print(line_str)
            file.write(line_str)
    print('Process-{0} finishes work'.format(rank))
"""
Plan-1
    1-Choose the number of used processes
        One master to assign parameter to 15 slaves, there is 16 processes
    2-Run the file
        Use 'cmd' in the folder containing this file, then type the following command:
            mpiexec -n 16 python lrc_ovr_linear_final_mpi.py
"""