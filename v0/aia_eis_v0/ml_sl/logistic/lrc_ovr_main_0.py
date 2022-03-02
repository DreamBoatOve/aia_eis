import os
import pickle

from ml_sl.logistic.lrc_0 import LRC
from ml_sl.ml_critrions import cal_accuracy, cal_kappa
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from utils.post_process.ml_post_process import binary_para_sorter
from utils.visualize_utils.ml_heatmap_utils import lrc_heatmap

# 1-Import dataset (Training, validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)
val_label_list, val_dataset_list = split_labeled_dataset_list(validation_dataset)
label_list = [2, 4, 5, 6, 7, 8, 9]

# 1-加载之前训练好的模型，直接进行预测
# 1.1-加载之前训练好的stable_alpha模型，直接进行预测
def load_and_run_previous_stable_models(folder):
    """
    function
        之前在OvR-OvO模式下已经训练过一批模型，存储位置（folder）：
        直接加载，使用OvR模式再次运行，查看模型效果
    :param
        folder:
    :return:
    """
    global val_label_list, val_dataset_list, label_list
    filenames = os.listdir(folder)
    for fn in filenames:
        # fn, like: 2020_04_18_lrc_ovr_stable_alpha_iter_1000_classifer_dict_pickle_0.file
        lrc_model_file_path = os.path.join(folder, fn)
        # create an basically empty LRC, just need its classify_ovr function
        lrc = LRC(alpha=0, max_iter=0, unlabeled_dataset_list=val_dataset_list, labeled_dataset_list=None, \
                  label_list=label_list)
        sample_label_prob_dict_list = lrc.classify_ovr(lrc_ovr_classifer_dict_pickle_filename=lrc_model_file_path)
        lrc_acc = cal_accuracy(sample_label_prob_dict_list, val_label_list)
        lrc_kappa = cal_kappa(sample_label_prob_dict_list, val_label_list)

        res_file_name = get_date_prefix() + 'ovr_stable.txt'
        with open(res_file_name, 'a+') as file:
            para_str_list = fn.split('iter')[1].split('_')
            max_iter = int(para_str_list[1])
            repeated_time = int(para_str_list[-1].split('.')[0])
            line_str = ','.join([fn,str(max_iter),str(repeated_time),str(lrc_acc),str(lrc_kappa)])+'\n'
            file.write(line_str)
        print('Add a line in the result file, check it')
# load_and_run_previous_stable_models(folder='ovr_res/stable/models')

# 1.2-加载之前训练好的linear_alpha模型，直接进行预测
def load_and_run_previous_linear_models(folder):
    """
    function
        之前在OvR-OvO模式下已经训练过一批模型，存储位置（folder）：
        直接加载，使用OvR模式再次运行，查看模型效果
    :param
        folder:
    :return:
    """
    global val_label_list, val_dataset_list, label_list
    filenames = os.listdir(folder)
    for fn in filenames:
        # fn, like: 2020_04_18_lrc_ovr_linear_alpha_iter_1000_alpha_init=0.1_classifer_dict_pickle_4.file
        lrc_model_file_path = os.path.join(folder, fn)
        # create an basically empty LRC, just need its classify_ovr function
        lrc = LRC(alpha=0, max_iter=0, unlabeled_dataset_list=val_dataset_list, labeled_dataset_list=None, \
                  label_list=label_list)
        sample_label_prob_dict_list = lrc.classify_ovr(lrc_ovr_classifer_dict_pickle_filename=lrc_model_file_path)
        lrc_acc = cal_accuracy(sample_label_prob_dict_list, val_label_list)
        lrc_kappa = cal_kappa(sample_label_prob_dict_list, val_label_list)

        res_file_name = get_date_prefix() + 'ovr_linear.txt'
        with open(res_file_name, 'a+') as file:
            para_str_list = fn.split('iter')[1].split('_')
            max_iter = int(para_str_list[1])
            alpha_init = float(para_str_list[3].split('=')[1])
            repeated_time = int(para_str_list[-1].split('.')[0])
            line_str = ','.join([fn,str(max_iter),str(alpha_init),str(repeated_time),str(lrc_acc),str(lrc_kappa)])+'\n'
            file.write(line_str)
        print('Add a line in the result file, check it')
# load_and_run_previous_linear_models(folder='ovr_res/linear/ovo_models')

# 2-画heatmap看看效果如何
# txt_filename = get_date_prefix()+'lrc_ovr_avg_res_0.txt'
# 求10次运行结果的平均性能
# binary_para_sorter(table_start_row=3, table_start_col=21, sheet_name='LRC',\
#                    excel_abs_path='../ml_training_records.xlsx', txt_filename=txt_filename,\
#                    para1_margin=1, para2_margin=2)
# 画heatmap
# axis_title_list = [y_axis_title, x_axis_title, z_axis_title]
# axis_title_list = ['Iteration', 'Alpha init', "Acc+Kappa"]
# lrc_heatmap(txt_file_path='ovr_res/linear/ovo_txt_res/2020_05_04_lrc_ovr_avg_res_0.txt', axis_title_list=axis_title_list)

# 3-增加训练参数，以保持与OVO的训练参数一致
def added_paras():
    para_list = []
    # add: iter = [1000 ~ 9000], alpha = [10 ~ 1000]
    for i in range(5):
        iter = 1000 + 2000 * i
        for j in range(1, 4):
            alpha = 10 ** j
            for a in range(10):
                # max_iter, alpha_init, i
                para_list.append([iter, alpha, a])
    # add: iter = [11000 ~ 15000], alpha = [1e-5 ~ 1e3]
    for i in range(3):
        iter = 11000 + 2000 * i
        for j in range(-5, 4):
            alpha = 10 ** j
            for a in range(10):
                # max_iter, alpha_init, i
                para_list.append([iter, alpha, a])
    return para_list
# added_paras()

# 4-画完整的Heatmap查看训练效果
# axis_title_list = [y_axis_title, x_axis_title, z_axis_title]
# axis_title_list = ['Iteration', 'Alpha init', "Acc+Kappa"]
# lrc_heatmap(txt_file_path='ovr_res/linear/plots/2020_05_07_lrc_ovr_linear_avg_res.txt', axis_title_list=axis_title_list)
# img: prj\ml_sl\logistic\ovr_res\linear\plots\2020_05_07_lrc_ovr_linear_avg_res.png
"""
由Heatmap及文件Excel结果可得出一下五组参数的10次平均性能排名前5
Rank        Iteration       Alpha_init      Accuracy+Kappa
1           3000            1               0.600494301
2           13000           1               0.566532511
3           9000            1               0.563128839
4           13000           0.1             0.561197776
5           13000           100             0.560529215
"""