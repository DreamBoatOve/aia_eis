import math

from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset
from ml_sl.ml_data_wrapper import split_labeled_dataset_list
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from ml_sl.logistic.lrc_0 import LRC
from utils.post_process.ml_post_process import binary_para_sorter
from utils.visualize_utils.ml_heatmap_utils import lrc_heatmap

# ------------------------------------ First time: LRC on EIS classification (One VS One)------------------------------------
# 1-Import dataset (Training, validation, Test)
ml_dataset_pickle_file_path = '../../datasets/ml_datasets/normed'
training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path=ml_dataset_pickle_file_path)

# 2-Training model with training-dataset and assess their performance(Accuracy, Kappa) on validation-dataset
val_label_list, val_dataset_list = split_labeled_dataset_list(validation_dataset)
label_list = [2, 4, 5, 6, 7, 8, 9]

# 2.1-Learning rate in gradient descent, alpha, is a constant and equal to 1 / max_iter
def lrc_ovo_stable_alpha():
    # max_iter starts from 1000~3000~5000~7000~9000, and its corresponding alphas are 1/1000, 1/3000, ..., 1/9000
    max_iter_list = [1000 + 2000 * i for i in range(5)]
    # Set small value for max_iter for the convenience of debug
    # max_iter_list = [200 + 20 * i for i in range(5)]

    for max_iter in max_iter_list:
        for i in range(10):
            lrc = LRC(alpha = None, max_iter = max_iter, \
                      unlabeled_dataset_list=val_dataset_list, labeled_dataset_list=training_dataset, label_list=label_list)
            date_str = get_date_prefix()
            lrc_classifer_dict_pickle_filename = date_str + 'lrc_ovo_stable_alpha_iter_'+str(max_iter)+'_classifer_dict_pickle_'+str(i)+'.file'
            lrc.create_lrc_classifer_dict_ovo(lrc_classifer_dict_pickle_filename)
            lrc_sample_label_prob_dict_list = lrc.classify_ovo(lrc_classifer_dict_pickle_filename)
            lrc_acc = cal_accuracy(lrc_sample_label_prob_dict_list, val_label_list)
            lrc_kappa = cal_kappa(lrc_sample_label_prob_dict_list, val_label_list)

            res_file_name = date_str + 'ovo_stable_alpha.txt'
            with open(res_file_name, 'a+') as file:
                line_str = lrc_classifer_dict_pickle_filename+','+str(max_iter)+','+str(i)+','+str(lrc_acc)+','+str(lrc_kappa)+'\n'
                file.write(line_str)
            print('Add a line in the result file, check it')

# 2.2- Learning rate in gradient descent, alpha, is linearly decreased and equal to alpha_init * (max_iter - iter_index) / max_iter
def lrc_ovo_linear_alpha():
    max_iter_list = [1000 + 2000 * i for i in range(8)] # len = 8
    # alpha_init starts from 1 to 1e-5
    alpha_init_list = [10 ** (-i) + 0.0 for i in range(6)] + [10, 100, 1000] # len = 9
    alpha_init_list = [alpha for alpha in alpha_init_list if alpha > 1]
    for max_iter in max_iter_list:
        for alpha_init in alpha_init_list:
            for i in range(10):
                lrc = LRC(alpha = alpha_init, max_iter = max_iter,\
                          unlabeled_dataset_list = val_dataset_list, labeled_dataset_list = training_dataset,\
                          label_list = label_list)
                date_str = get_date_prefix()
                lrc_classifer_dict_pickle_filename = date_str + 'lrc_ovo_linear_alpha_iter_'\
                                                     + str(max_iter) +'_alpha_init_' + str(alpha_init) + '_classifer_dict_pickle_' + str(i) + '.file'
                lrc.create_lrc_classifer_dict_ovo(lrc_classifer_dict_pickle_filename)
                lrc_sample_label_prob_dict_list = lrc.classify_ovo(lrc_classifer_dict_pickle_filename)
                lrc_acc = cal_accuracy(lrc_sample_label_prob_dict_list, val_label_list)
                lrc_kappa = cal_kappa(lrc_sample_label_prob_dict_list, val_label_list)

                res_file_name = date_str + 'linear_alpha.txt'
                with open(res_file_name, 'a+') as file:
                    line_str = lrc_classifer_dict_pickle_filename + ',' + str(max_iter) + ',' +str(alpha_init)+','\
                               + str(i) + ',' + str(lrc_acc) + ',' + str(lrc_kappa) + '\n'
                    file.write(line_str)
                print('Add a line in the result file, check it')

# lrc_ovo_stable_alpha()
# lrc_ovo_linear_alpha()

# 3-Manually select the best parameters(max_iter, stable or linear alpha)
# 3.1.1- OVO-Stable
#     不管迭代多少次，结果都是一样的Accuracy (validation)=0.170212765957447，Kappa (validation)=0，AK(validation)=0.170212765957447,没必要画热度图

# 3.1.2- OVO-Linear
# txt_filename = get_date_prefix()+'lrc_ovo_linear_avg_res_0.txt'
# binary_para_sorter(table_start_row=3, table_start_col=12, sheet_name='LRC', excel_abs_path='../ml_training_records.xlsx', txt_filename=txt_filename)
# 画Lrc-OVO_linear热度图
# txt_filename = 'ovo_res/linear/plots/2020_04_20_lrc_ovo_linear_avg_res_0.txt'
# lrc_heatmap(txt_file_path=txt_filename, axis_title_list=['Iteration', 'Learing rate', 'Acc+Kappa'])

# 3.2.1-OVR-stable
#   不管迭代多少次，结果都是一样的Accuracy (validation)=0.436170213，Kappa (validation)=0.313868613，OVO linear useage rate = 1没必要画热度图
#   OVO linear useage rate = 1: 意味着对于每一个无标签数据，在用N个分类器分类时，均出现有2个及以上的标签获得投票，此时放弃OVR策略，转而使用之前训练好的OVO-Linear的一个模型进行分类预测

# 3.2.2-OVR-Linear
# txt_filename = 'ovr_res/linear/plots/2020_05_02_lrc_ovr_ovo_linear_avg_res.txt'
# 画Lrc-OVR-Linear热度图
# lrc_heatmap(txt_file_path=txt_filename, axis_title_list=['Iteration', 'Learing rate', 'Acc+Kappa'])

# 3.3- OvR(OvO)-Linear
#
# 4-Assess the performance of LRC on [training+validation] and [Test] datasets, respectively
def lrc_ovo_linear_final_alpha():
    global training_dataset, validation_dataset, test_dataset
    tr_va_dataset = training_dataset + validation_dataset
    te_label_list, te_data_list = split_labeled_dataset_list(test_dataset)

    # iter_alpha_list records the first best 5 iteration+Alpha_init
    iter_alpha_list = [ [5000, 10], [5000, 100], [5000, 1000], [7000, 100], [7000, 1000] ]
    for iter_alpha in iter_alpha_list:
        max_iter, alpha_init = iter_alpha
        for i in range(10):
            lrc = LRC(alpha = alpha_init, max_iter = max_iter,\
                      unlabeled_dataset_list = te_data_list, labeled_dataset_list = tr_va_dataset,\
                      label_list = label_list)
            date_str = get_date_prefix()
            lrc_classifer_dict_pickle_filename = date_str + 'lrc_ovo_linear_final_iter='\
                                                 + str(max_iter) +'_alpha_init=' + str(alpha_init) + '_classifer_dict_pickle_' + str(i) + '.file'
            lrc.create_lrc_classifer_dict_ovo(lrc_classifer_dict_pickle_filename)
            lrc_sample_label_prob_dict_list = lrc.classify_ovo(lrc_classifer_dict_pickle_filename)
            lrc_acc = cal_accuracy(lrc_sample_label_prob_dict_list, te_label_list)
            lrc_kappa = cal_kappa(lrc_sample_label_prob_dict_list, te_label_list)

            res_file_name = date_str + 'linear_final.txt'
            with open(res_file_name, 'a+') as file:
                line_str = lrc_classifer_dict_pickle_filename + ',' + str(max_iter) + ',' +str(alpha_init)+','\
                           + str(i) + ',' + str(lrc_acc) + ',' + str(lrc_kappa) + '\n'
                print(line_str)
                file.write(line_str)

"""
由 Heatmap 及文件Excel结果可得出一下五组参数的10次平均性能排名前5
Rank        Iteration       Alpha_init      Accuracy+Kappa
1           5000            1000            0.890252707
2           5000            100             0.890200138
3           5000            10              0.888490665
4           7000            100             0.838317247
5           7000            1000            0.822569381
"""

# 5- Access the accuracy of the first one, two, and three predictions
# Load trained models, lrc
# lrc classify test-dataset to get lable_prob_dict
# ------------------------------------ First time: LRC on EIS classification ------------------------------------