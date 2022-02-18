import sys
sys.path.append('../../../')
import os
from multiprocessing import Process, Queue

from data_processor.GOA_preprocessor.goa_data_wrapper import load_Lai_EIS_data
from goa.integration.goa_intergration import goa_fitter_0, goa_fitter_multiProcess_0

"""
Module function
    The best five GOAs for each ECM have been picked out
    Procedure:
        1- Read raw EIS experiment
        2- Assign matched the best five GOAs to EIS according to its ECM
        3- Using python multiprocessing module to run tasks parallel, self-defined process number according to hardware's 
            condition
Version 1:
    我统计Lai每个参数的拟合结果，得到每个参数拟合的最大和最小值作为GOA拟合的上下限范围，这个范围很大，拟合结果估计很不理想
"""
# 1- Load Lai's normed(* multiply experimental area 1.01 * 1e-6 cm^2) EIS data
lai_normed_eis_dict_list = load_Lai_EIS_data(file_path='../../../datasets/goa_datasets/normed',
                                             file_name='2020_08_22_goa_lai_normed_dataset_pickle.file')
# 2- Load each EIS's ECM type
def load_eis_ECM_dict(file_path, file_name):
    eis_ecm_dict = {}
    with open(os.path.join(file_path, file_name), 'r') as file:
        for line in file.readlines():
            exp_file_name, ecm_num_str = line.strip().split(',')
            eis_ecm_dict[exp_file_name] = int(ecm_num_str)
    return eis_ecm_dict
eis_ecm_dict = load_eis_ECM_dict(file_path='../../../datasets/goa_datasets/Lai_manual_fitting_res',
                                 file_name='2020_07_22_lai_ecm_Num_record.txt')

# 3- Load ECM paras' limits
def load_ecm_para_limits(ecm_num):
    """
    para limits:
        ECM-2
            2020-01-04 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                3.949E-26	    0.00001363	    0.4003	    0.06686	        0.00002237	    0.6	        0.06092
                Maximum:
                0.04596	        0.5935          0.9969      120500          0.6094          0.9445      31270
                Range
                [1e-27, 1e-1],  [1e-5, 1.0],    [0.3, 1.0], [1e-3, 1e7],    [1e-5, 1.0],    [0.5, 1.0], [1e-3, 1e6]
            2020-07-22 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                6.357E-17	    0.00001363	    0.4003	    0.06686	        0.00002237	    0.6	        0.06092
                Maximum:
                0.04596	        0.5935          0.9969      120500          0.6094          0.9445      3704
                Range
                [1e-18, 1e-1],  [1e-6, 1.0],    [0.3, 1.0], [1e-3, 1e7],    [1e-5, 1.0],    [0.5, 1.0], [1e-3, 1e5]
        ECM-9
            2020-01-04 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                2.319E-28	    3.819E-11	    0.6666	    2.595E-08	    7.379E-11	    0.5157	    82.8
                Maximum:
                0.02633	        0.946           0.9986	    26460000	    0.0123	        0.968	    41850000000
                Range
                [1e-29, 1e-1],  [1e-12, 10],    [0.6, 1.0], [1e-9, 1e9],    [1e-12, 1e-1],  [0.5, 1.0], [10, 1e12]
            2020-07-22 version:
                R0              Q0_q            Q0_n        R1              Q1_q            Q1_n        R2
                Minimum:
                5.426E-30	    3.819E-11	    0.6666	    2.595E-08	    7.379E-11	    0.5157	    82.8
                Maximum:
                0.02633	        0.002235        0.9986	    26460000	    0.0123	        0.968	    41850000000
                Range
                [1e-31, 1e-1],  [1e-12, 10],    [0.6, 1.0], [1e-9, 1e9],    [1e-12, 1e-1],  [0.5, 1.0], [10, 1e12]
    """
    if ecm_num == 2:
        limit_list = [[1e-18, 1e-1], [1e-6, 1.0], [0.3, 1.0], [1e-3, 1e7], [1e-5, 1.0], [0.5, 1.0], [1e-3, 1e5]]
    elif ecm_num == 9:
        limit_list = [[1e-31, 1e-1], [1e-12, 10], [0.6, 1.0], [1e-9, 1e9], [1e-12, 1e-1], [0.5, 1.0], [10, 1e12] ]
    return limit_list

# 4- Gather all needed info for GOA fitting
def pack_ecm_para_config():
    """
    function
        Gather all the needed information for GOA's fitting, specifically:
            ecm_para_config_dict['exp_fn']: raw experimental file name, like: '1-23', '4-34'
            ecm_para_config_dict['ecm_num'] = ecm_num, int
            ecm_para_config_dict['limit']: [[para_0_low_limit, para_0_upper_limit], [para_1_low_limit, para_1_upper_limit], ...]
            ecm_para_config_dict['f']: list[float, frequency ]
            ecm_para_config_dict['z_raw']: list[complex, Impedance], but using NORMED EIS data here
    :param

    :return:
        ecm_para_config_dict_list = [ecm_para_config_dict_0, ecm_para_config_dict_1, dict, ...]
    """
    global lai_normed_eis_dict_list
    global eis_ecm_dict

    ecm_para_config_dict_list = []
    for normed_eis_dict in lai_normed_eis_dict_list:
        ecm_para_config_dict = {}
        ecm_para_config_dict['exp_fn'] = normed_eis_dict['file_name']
        ecm_para_config_dict['ecm_num'] = normed_eis_dict['ecm_num']
        ecm_para_config_dict['limit'] = load_ecm_para_limits(normed_eis_dict['ecm_num'])
        ecm_para_config_dict['f'] = normed_eis_dict['f']
        ecm_para_config_dict['z_raw'] = normed_eis_dict['z_raw']
        ecm_para_config_dict_list.append(ecm_para_config_dict)
    return ecm_para_config_dict_list
ecm_para_config_dict_list = pack_ecm_para_config()

# 测试goa_fitter
# goa_fitter(ecm_para_config_dict=ecm_para_config_dict_list[41], repeat_time=5)

if __name__ == '__main__':
    # --------- 已知 multiprocess 的进程可以包含一个简单的函数，但是一个包含类的函数不知能否带动？ 下方一行代码可运行，能带动 ---------
    # p = Process(target=goa_fitter, args=(ecm_para_config_dict_list[41],5))
    # p.start()
    # p.join()
    # --------- 已知 multiprocess 的进程可以包含一个简单的函数，但是一个包含类的函数不知能否带动？ 下方一行代码可运行，能带动 ---------

    """
    一个电脑最多开N个进程，N个进程同时执行相同的程序（不同的内容），有些进程处理的较快，提前结束了，此时能不能再增开一个进程来充分利用计算资源
    Task Assignment:
        127 files in total
        2020.08.21 程序因ICA中断
            Traceback (most recent call last):
              File "G:\Python\Python\Pythov3.6.1\lib\multiprocessing\process.py", line 297, in _bootstrap
                self.run()
              File "G:\Python\Python\Pythov3.6.1\lib\multiprocessing\process.py", line 99, in run
                self._target(*self._args, **self._kwargs)
              File "../../..\global_optimizations\integration\goa_intergration.py", line 263, in goa_fitter_multiProcess_0
                goa_fitter(ecm_para_config_dict, repeat_time)
              File "../../..\global_optimizations\integration\goa_intergration.py", line 244, in goa_fitter
                cur_best_entity_list, global_best_entity_list, iter, chi_squared = ica.search()
            ValueError: not enough values to unpack (expected 4, got 2)
            导致如下文件训练中断：
                文件名     任务索引
                1-18        13
                1-29        22
                1-31        24
                2-5         31
                2-7         33
                2-11        37
    """
    # task_start_index, task_end_index = 0, 30
    # task_list = ecm_para_config_dict_list[task_start_index : task_end_index]

    # 2020.08.21 程序因ICA中断,重跑程序
    # task_list = []
    # for i in [13, 22, 24, 31, 33, 37] + list(range(38, len(ecm_para_config_dict_list))):
    #     task_list.append(ecm_para_config_dict_list[i])

    # 2020.08.22 发现阻抗在乘以实验面积的时候，错误的除以面积了，之前的结果都是错的，重跑
    task_list = ecm_para_config_dict_list[56:]

    repeat_time = 10

    que = Queue()
    task_counter = 0
    running_flag = True

    process_num = 16 # user defined according to hardware's condition
    process_list = []
    while running_flag:
        # if (que.qsize() < que_len) and (task_counter < len(task_list)):
        # if (len(process_list) < process_num) and (task_counter < len(task_list)):
        while (len(process_list) < process_num) and (task_counter < len(task_list)):
            p_name = 'p{0}'.format(task_counter)
            p = Process(name=p_name, target=goa_fitter_multiProcess_0,
                        args=(que, p_name, task_list[task_counter], repeat_time))
            """
            multiprocessing process start join 的顺序:
                python 多进程 —— multiprocessing.Process https://blog.csdn.net/HeatDeath/article/details/72842899
                这个教程的下方有个示意图，非常好
            """
            p.start()
            process_list.append([p, False]) # p: Process, False: join_flag, join = True, not join = False
            task_counter += 1

        for i, pro in enumerate(process_list):
            if not pro[1]:
                pro[0].join()
                pro[1] = True

        if que.qsize() > 0:
            for i in range(que.qsize()):
                finished_process_name = que.get()
                for i, pro in enumerate(process_list):
                    if pro[0].name == finished_process_name:
                        """
                        python multiprocessing terminate子进程后需要close吗？
                            close是multiprocessing.Pool专有的方法，Process根本没有这个方法
                        """
                        pro[0].terminate()
                        # process_list.remove(p)
                        del process_list[i]

        if (task_counter == len(task_list)) and (len(process_list) == 0):
            running_flag = False
# python GOAs_fit_EIS_1.py