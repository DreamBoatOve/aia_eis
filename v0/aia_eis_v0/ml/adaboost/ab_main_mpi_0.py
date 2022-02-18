import sys
"""
python解释器会把当前文件（模块）当作项目的根目录，以此来寻找其他的模块
使用sys.path调整解释器的寻找路径，以此文件为起点，调整路径到项目的真实根目录
"""
sys.path.append('../../')

from ml.adaboost.ab_0 import AB
from ml.ml_critrions import cal_accuracy, cal_kappa
from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset

from mpi4py import MPI
comm = MPI.COMM_WORLD
rank = comm.rank

if rank == 0:
    # Process_0 assigns parameter to other processes
    para_list = []
    for i in range(10):
        for boost_num in [50 * i for i in range(1, 10)]:
            para_list.append([i, boost_num])
    comm.send(para_list[int(len(para_list) / 10) * 0:int(len(para_list) / 10) * 1], dest=1)
    comm.send(para_list[int(len(para_list) / 10) * 1:int(len(para_list) / 10) * 2], dest=2)
    comm.send(para_list[int(len(para_list) / 10) * 2:int(len(para_list) / 10) * 3], dest=3)
    comm.send(para_list[int(len(para_list) / 10) * 3:int(len(para_list) / 10) * 4], dest=4)
    comm.send(para_list[int(len(para_list) / 10) * 4:int(len(para_list) / 10) * 5], dest=5)
    comm.send(para_list[int(len(para_list) / 10) * 5:int(len(para_list) / 10) * 6], dest=6)
    comm.send(para_list[int(len(para_list) / 10) * 6:int(len(para_list) / 10) * 7], dest=7)
    comm.send(para_list[int(len(para_list) / 10) * 7:int(len(para_list) / 10) * 8], dest=8)
    comm.send(para_list[int(len(para_list) / 10) * 8:int(len(para_list) / 10) * 9], dest=9)
    comm.send(para_list[int(len(para_list) / 10) * 9:len(para_list)], dest=10)

else:
    # Accept parameter from Process_0
    print('Process-{0} got parameters and starts working'.format(rank))
    rec_para_list = comm.recv(source=0)

    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    va_label_list = [d[0] for d in validation_dataset]
    va_data_list = [d[1] for d in validation_dataset]
    label_list = [2,4,5,6,7,8,9]

    for para in rec_para_list:
        i, boost_num = para
        # alpha_init and max_iter are decided from the result of Logistic-OVO-Linear-Alpha training results
        ab = AB(boost_num=boost_num, resample_num=10, alpha_init = 1, max_iter = 9,\
                unlabeled_dataset_list=va_data_list, labeled_dataset_list=training_dataset, label_list = label_list)
        ab_model_name = get_date_prefix() + 'ab_boost_num=' + str(boost_num) + '_' + str(i) + '_pickle.file'

        # 1.2 - Create and Store each model
        ab.create_ab_classifer(ab_model_name)
        ab_sample_label_prob_dict_list = ab.classify(ab_model_name)

        # 1.3- calculate accuracy and kappa
        acc = cal_accuracy(ab_sample_label_prob_dict_list, va_label_list)
        kappa = cal_kappa(ab_sample_label_prob_dict_list, va_label_list)

        # 1.4-Record performance(accuracy + kappa)
        res_filename = get_date_prefix() + 'ab_mpi_res.txt'
        with open(res_filename, 'a+') as file:
            line_str = ','.join([ab_model_name, str(i), str(acc), str(kappa)]) + '\n'
            print('Process-{0} write: {1}'.format(rank, line_str))
            file.write(line_str)
"""
Plan-1
    1-Choose the number of used processes
        One master to assign parameter to 10 slaves, there is 11 processes
    2-Run the file
        Use 'cmd' in the folder containing this file, then type the following command:
            mpiexec -n 11 python ab_main_mpi.py
"""