import sys
"""
python解释器会把当前文件（模块）当作项目的根目录，以此来寻找其他的模块
使用sys.path调整解释器的寻找路径，以此文件为起点，调整路径到项目的真实根目录
"""
sys.path.append('../../')

from ml_sl.svm.multiclass_svm_0 import Multiclass_SVM
from ml_sl.ml_critrions import cal_accuracy, cal_kappa
from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from utils.post_process.ml_sort_mpi_res import svm_res_checker

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    # Process_0 assigns parameter to other processes
    linear_svm_para_list, poly_svm_para_list, rbf_svm_para_list = svm_res_checker(folder='ovo_txt_res')
    # G505 solves Linear+RBF
    para_list = linear_svm_para_list + rbf_svm_para_list
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
    comm.send(para_list[int(len(para_list) / 15) * 14: len(para_list) ], dest=15)

else:
    # Accept parameter from Process_0
    print('Process-{0} got parameters and starts working'.format(rank))
    rec_para_list = comm.recv(source=0)

    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])

    for para in rec_para_list:
        i, svm_para_dict, kernel_para_dict = para
        multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict, \
                                   unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset, \
                                   label_list=[2, 4, 5, 6, 7, 8, 9])
        svm_model_pickle_name = get_date_prefix()
        kernel_type = kernel_para_dict['type']
        if kernel_type == 'linear':
            svm_model_pickle_name += 'svm_{0}_C={1}_iter={2}_pickle_{3}.file'.format(kernel_type, \
                                                                svm_para_dict['C'], svm_para_dict['max_iter'], i)
        elif kernel_type == 'poly':
            p,w,q = kernel_para_dict['paras']
            svm_model_pickle_name += 'svm_{0}_C={1}_iter={2}_P={3}_q={4}_pickle_{5}.file'.format(kernel_type,\
                                                        svm_para_dict['C'], svm_para_dict['max_iter'], p, q, i)
        elif kernel_type == 'rbf':
            svm_model_pickle_name +=  'svm_{0}_C={1}_iter={2}_sigma={3}_pickle_{4}.file'.format(kernel_type, \
                                    svm_para_dict['C'], svm_para_dict['max_iter'], kernel_para_dict['paras'], i)

        multi_svm.create_svm_ovo_classifer(svm_model_pickle_name)
        sample_label_prob_dict_list = multi_svm.classify_ovo(svm_model_pickle_name)

        acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
        kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)

        res_name = get_date_prefix()+'svm_ovo_mpi.txt'
        with open(res_name, 'a+') as file:
            line = ','.join([svm_model_pickle_name, str(i), str(acc), str(kappa)]) + '\n'
            file.write(line)
    print('Process-{0} ends working'.format(rank))
"""
Plan
    1-Choose the number of used processes
        Use 15 slaves to handle
        One master to assign parameter to 15 slaves, there is 16 processes
    2-Run the file
        Use 'cmd' in the folder containing this file, then type the following command:
            mpiexec -n 16 python svm_ovo_main_mpi_G505_16p.py
"""