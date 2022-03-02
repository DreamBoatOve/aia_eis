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
    # NH55 focus on Poly training
    # para_list = poly_svm_para_list
    para_list = linear_svm_para_list + poly_svm_para_list + rbf_svm_para_list

    # 24 slaves + 1 master
    comm.send(para_list[int(len(para_list) / 24) * 0:int(len(para_list) / 24) * 1], dest=1)
    comm.send(para_list[int(len(para_list) / 24) * 1:int(len(para_list) / 24) * 2], dest=2)
    comm.send(para_list[int(len(para_list) / 24) * 2:int(len(para_list) / 24) * 3], dest=3)
    comm.send(para_list[int(len(para_list) / 24) * 3:int(len(para_list) / 24) * 4], dest=4)
    comm.send(para_list[int(len(para_list) / 24) * 4:int(len(para_list) / 24) * 5], dest=5)
    comm.send(para_list[int(len(para_list) / 24) * 5:int(len(para_list) / 24) * 6], dest=6)
    comm.send(para_list[int(len(para_list) / 24) * 6:int(len(para_list) / 24) * 7], dest=7)
    comm.send(para_list[int(len(para_list) / 24) * 7:int(len(para_list) / 24) * 8], dest=8)
    comm.send(para_list[int(len(para_list) / 24) * 8:int(len(para_list) / 24) * 9], dest=9)
    comm.send(para_list[int(len(para_list) / 24) * 9:int(len(para_list) / 24) * 10], dest=10)
    comm.send(para_list[int(len(para_list) / 24) * 10:int(len(para_list) / 24) * 11], dest=11)
    comm.send(para_list[int(len(para_list) / 24) * 12:int(len(para_list) / 24) * 13], dest=13)
    comm.send(para_list[int(len(para_list) / 24) * 11:int(len(para_list) / 24) * 12], dest=12)
    comm.send(para_list[int(len(para_list) / 24) * 13:int(len(para_list) / 24) * 14], dest=14)
    comm.send(para_list[int(len(para_list) / 24) * 14:int(len(para_list) / 24) * 15], dest=15)
    comm.send(para_list[int(len(para_list) / 24) * 15:int(len(para_list) / 24) * 16], dest=16)
    comm.send(para_list[int(len(para_list) / 24) * 16:int(len(para_list) / 24) * 17], dest=17)
    comm.send(para_list[int(len(para_list) / 24) * 17:int(len(para_list) / 24) * 18], dest=18)
    comm.send(para_list[int(len(para_list) / 24) * 18:int(len(para_list) / 24) * 19], dest=19)
    comm.send(para_list[int(len(para_list) / 24) * 19:int(len(para_list) / 24) * 20], dest=20)
    comm.send(para_list[int(len(para_list) / 24) * 20:int(len(para_list) / 24) * 21], dest=21)
    comm.send(para_list[int(len(para_list) / 24) * 21:int(len(para_list) / 24) * 22], dest=22)
    comm.send(para_list[int(len(para_list) / 24) * 22:int(len(para_list) / 24) * 23], dest=23)
    comm.send(para_list[int(len(para_list) / 24) * 23:len(para_list)], dest=24)

    # 30 slaves + 1 master
    # comm.send(para_list[int(len(para_list) / 30) * 0:int(len(para_list) / 30) * 1], dest=1)
    # comm.send(para_list[int(len(para_list) / 30) * 1:int(len(para_list) / 30) * 2], dest=2)
    # comm.send(para_list[int(len(para_list) / 30) * 2:int(len(para_list) / 30) * 3], dest=3)
    # comm.send(para_list[int(len(para_list) / 30) * 3:int(len(para_list) / 30) * 4], dest=4)
    # comm.send(para_list[int(len(para_list) / 30) * 4:int(len(para_list) / 30) * 5], dest=5)
    # comm.send(para_list[int(len(para_list) / 30) * 5:int(len(para_list) / 30) * 6], dest=6)
    # comm.send(para_list[int(len(para_list) / 30) * 6:int(len(para_list) / 30) * 7], dest=7)
    # comm.send(para_list[int(len(para_list) / 30) * 7:int(len(para_list) / 30) * 8], dest=8)
    # comm.send(para_list[int(len(para_list) / 30) * 8:int(len(para_list) / 30) * 9], dest=9)
    # comm.send(para_list[int(len(para_list) / 30) * 9:int(len(para_list) / 30) * 10], dest=10)
    # comm.send(para_list[int(len(para_list) / 30) * 10:int(len(para_list) / 30) * 11], dest=11)
    # comm.send(para_list[int(len(para_list) / 30) * 12:int(len(para_list) / 30) * 13], dest=13)
    # comm.send(para_list[int(len(para_list) / 30) * 11:int(len(para_list) / 30) * 12], dest=12)
    # comm.send(para_list[int(len(para_list) / 30) * 13:int(len(para_list) / 30) * 14], dest=14)
    # comm.send(para_list[int(len(para_list) / 30) * 14:int(len(para_list) / 30) * 15], dest=15)
    # comm.send(para_list[int(len(para_list) / 30) * 15:int(len(para_list) / 30) * 16], dest=16)
    # comm.send(para_list[int(len(para_list) / 30) * 16:int(len(para_list) / 30) * 17], dest=17)
    # comm.send(para_list[int(len(para_list) / 30) * 17:int(len(para_list) / 30) * 18], dest=18)
    # comm.send(para_list[int(len(para_list) / 30) * 18:int(len(para_list) / 30) * 19], dest=19)
    # comm.send(para_list[int(len(para_list) / 30) * 19:int(len(para_list) / 30) * 20], dest=20)
    # comm.send(para_list[int(len(para_list) / 30) * 20:int(len(para_list) / 30) * 21], dest=21)
    # comm.send(para_list[int(len(para_list) / 30) * 21:int(len(para_list) / 30) * 22], dest=22)
    # comm.send(para_list[int(len(para_list) / 30) * 22:int(len(para_list) / 30) * 23], dest=23)
    # comm.send(para_list[int(len(para_list) / 30) * 23:int(len(para_list) / 30) * 24], dest=24)
    # comm.send(para_list[int(len(para_list) / 30) * 24:int(len(para_list) / 30) * 25], dest=25)
    # comm.send(para_list[int(len(para_list) / 30) * 25:int(len(para_list) / 30) * 26], dest=26)
    # comm.send(para_list[int(len(para_list) / 30) * 26:int(len(para_list) / 30) * 27], dest=27)
    # comm.send(para_list[int(len(para_list) / 30) * 27:int(len(para_list) / 30) * 28], dest=28)
    # comm.send(para_list[int(len(para_list) / 30) * 28:int(len(para_list) / 30) * 29], dest=29)
    # comm.send(para_list[int(len(para_list) / 30) * 29:len(para_list)], dest=30)

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
        Use 24 slaves to handle
        One master to assign parameter to 24 slaves, there is 25 processes
    2-Run the file
        Use 'cmd' in the folder containing this file, then type the following command:
            mpiexec -n 25 python svm_ovo_main_mpi_25p.py
"""