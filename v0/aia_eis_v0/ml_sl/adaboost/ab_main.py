import os
import pickle
import numpy as np
from matplotlib import pyplot as plt

from ml_sl.adaboost.ab_0 import AB
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from utils.visualize_utils.shareX_2Y_plots import my_shareX_2Y_plot_4_AB_1

"""
Adaboost 可调参数
    弱分类器的数量 = [1(no meaning, just pure logistic), 50, 100, 150, 200, 250, 300, 350, 400, 450(no more than the number training-dataset)]
    弱分类器的可调参数
        Logistic
            OVO + Linearly decreasing alpha
        iteration and alpha_init are decided from the training results of Logistic 
"""
# 1-使用训练集训练AB，在验证集上测试性能
def AB_tr_va(training_dataset, validation_dataset, alpha_init=1, max_iter=9000, label_list=[2,4,5,6,7,8,9]):
    """
    :param
        training_dataset:
        validation_dataset:
        alpha_init:
        max_iter:
            为何选择9000？
        label_list:
    :return:
    """
    boost_num_list = [50 * i for i in range(1, 10)]
    va_label_list = [d[0] for d in validation_dataset]
    va_data_list = [d[1] for d in validation_dataset]
    # 1.1- AB with a boost_num is repeated for 10 times
    for boost_num in boost_num_list:
        for i in range(10):
            ab = AB(boost_num=boost_num, resample_num=10, alpha_init=alpha_init, max_iter=max_iter,\
                    unlabeled_dataset_list=va_data_list, labeled_dataset_list=training_dataset, label_list=label_list)
            ab_model_name = get_date_prefix()+'ab_boost_num='+str(boost_num)+'_'+str(i)+'_pickle.file'

            # 1.2 - Create and Store each model
            ab.create_ab_classifer(ab_model_name)
            ab_sample_label_prob_dict_list = ab.classify(ab_model_name)

            # 1.3- calculate accuracy and kappa
            acc = cal_accuracy(ab_sample_label_prob_dict_list, va_label_list)
            kappa = cal_kappa(ab_sample_label_prob_dict_list, va_label_list)

            # 1.4-Record performance(accuracy + kappa)
            res_filename = get_date_prefix()+'ab_res.txt'
            with open(res_filename, 'a+') as file:
                line_str = ','.join([ab_model_name, str(i), str(acc), str(kappa)]) + '\n'
                print(line_str)
                file.write(line_str)

# 2- Decide the boosting number from above results
# 2.1- Use boxplot to visualize the averaged performance of AB with different boost_num
# if __name__ == '__main__':
#     training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    # Use small iteration to test the function of the code is ok
    # AB_tr_va(training_dataset, validation_dataset, alpha_init=100, max_iter=5000)

# --------------------- Trained on tr+va and Tested on te ---------------------
def AB_TV_Te():
    """
    Function:
        After a through grid search of AdaBoost's hyperparameter, Boost number or the number of weak learners,
        we get the following best five candidates:
            Boost number: 150, 200, 250, 300, 350
    :param
        training_dataset:
        validation_dataset:
        alpha_init:
        max_iter:
        label_list:
    """
    alpha_init = 1
    max_iter = 9000
    label_list = [2, 4, 5, 6, 7, 8, 9]

    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    ab_final_res_fn = get_date_prefix() + 'ab_final_res.txt'
    counter = 0
    for b in range(5):
        boost_num = 150 + 50 * b
        for i in range(10):
            ab = AB(boost_num=boost_num, resample_num=10, alpha_init=alpha_init, max_iter=max_iter,
                    unlabeled_dataset_list=test_data_list,
                    labeled_dataset_list=training_dataset+validation_dataset,
                    label_list=label_list)
            ab_model_name = get_date_prefix()+'ab_boost_num='+str(boost_num)+'_'+str(i)+'_pickle.file'

            # 1.2 - Create and Store each model
            ab.create_ab_classifer(ab_model_name)
            ab_sample_label_prob_dict_list = ab.classify(ab_model_name)

            # 1.3- calculate accuracy and kappa
            acc = cal_accuracy(ab_sample_label_prob_dict_list, test_label_list)
            kappa = cal_kappa(ab_sample_label_prob_dict_list, test_label_list)

            # 1.4- Record performance(accuracy + kappa)
            with open(ab_final_res_fn, 'a+') as file:
                line_str = ','.join([ab_model_name, str(boost_num), str(i), str(acc), str(kappa), str(acc+kappa)]) + '\n'
                file.write(line_str)
            counter += 1
            print('Finished {0}, {1} left'.format(counter, 50 - counter))
# AB_TV_Te()
# --------------------- Trained on tr+va and Tested on te ---------------------

# 3- Access top 5 Adaboost para settings' accuracy on the first one, two, and three predictions
def get_first3_acc():
    # Load trained model
    ab_model_path = os.path.join('models/trained_on_TV_tested_on_test',\
                                 '2020_07_06_ab_final_boost_num=150_3_pickle.file')

    label_list = [2, 4, 5, 6, 7, 8, 9]
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    alpha_init = 1
    max_iter = 9000
    boost_num = 150
    ab = AB(boost_num=boost_num, resample_num=10, alpha_init=alpha_init, max_iter=max_iter,
            unlabeled_dataset_list=test_data_list,
            labeled_dataset_list=training_dataset + validation_dataset,
            label_list=label_list)
    ab_sample_label_prob_dict_list = ab.classify(ab_model_name = ab_model_path)
    ab_acc = cal_accuracy(ab_sample_label_prob_dict_list, test_label_list)
    ab_acc_on_2 = cal_accuracy_on_2(ab_sample_label_prob_dict_list, test_label_list)
    ab_acc_on_3 = cal_accuracy_on_3(ab_sample_label_prob_dict_list, test_label_list)
    ab_kappa = cal_kappa(ab_sample_label_prob_dict_list, test_label_list)
    print('Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(ab_acc, ab_acc_on_2, ab_acc_on_3, ab_kappa))
# get_first3_acc()
# Results:
#   Accuracy on 1 = 0.5714285714285714, Accuracy on 2 = 0.7692307692307693,
#   Accuracy on 3 = 0.8791208791208791, Kappa=0.4740663900414937

# 4- Look into the final AB model
# Final AB model has the highest AK = 1.045
def parse_final_AB_model(fp, fn, ecm_tuple, xy_type, output_flag=True):
    # 1- load Final AB model
    """
    ecm_tuple:
        tuple(int, int), like (2,4) means (ECM2 and ECM4)
    xy_type:
        str, 'XYXY' or 'XYY'
        根据Origin-WaterFall的数据格式要求 XYXY or XYY
    Final AB model
        150 LRC
            Possible ECM types: ECM- 2, 4, 5, 6, 7, 8, 9 (7 kinds)
            OvO classification strategy:
                6 + 5 + 4 + ... + 2 + 1 = (6 + 1) / (6 / 2) = 21 binary LRC classifier
        data structure
            [multi-class-LRC_000, multi-class-LRC_001, multi-class-LRC_002, ..., multi-class-LRC_148, multi-class-LRC_149]
            multi-class-LRC_000:{
                'model':{
                    (2,4) means (ECM2 and ECM4):{
                        'w': [float]
                        'b': float
                        'r': ratio, to balance an unbalanced binary dataset, like class-1 has 1000 pieces of data,
                            class-2 has only 100 pieces of data
                    }
                }
                'c': float, weight factor of multi-class-LRC_000
            }
    """
    with open(os.path.join(fp, fn), 'rb') as file:
        ab_model = pickle.load(file)

    w_list = []
    # c_list = []
    for ab in ab_model:
        w_list.append(ab['model'][ecm_tuple]['w'])
        # c_list.append(ab['c'])
    print('ECM {0} / {1}'.format(*ecm_tuple))
    print('W:Max={0},Min={1}'.format(max(max(w_list)), min(min(w_list))))
    # print('C:Max={0};Min={1}'.format(max(c_list), min(c_list)))

    if output_flag:
        # output w_list, c_list to txt files, then use origin to plot
        w_fn = get_date_prefix()
        if xy_type == 'XYXY':
            w_fn += 'final_ab_w_ecm{0}n{1}_XYXY.txt'.format(*ecm_tuple)
            with open(w_fn, 'a+') as file:
                """
                因为Origin中画Waterfall需要XYXYXY...格式的数据，所以此处不能把每个sub_w_list打印到一行里
                for w in w_list:
                    line = ','.join(list(map(str, w))) + '\n'
                    file.write(line)
                按照Origin-WaterFall-XYXYXYXY的数据格式要求，一列X=list(range(1,161)),一列y(sub_w_list,150个)，一共300列
                之后还要手动把很多列设置为X，很麻烦
                """
                line_list = ['' for i in range(len(w_list[0]))]
                for c in range(len(w_list)):
                    # for r, line in enumerate(line_list):
                    #     line += ','.join(list(map(str, [r+1, w_list[c][r]])))
                    for r in range(len(w_list[0])):
                        if c > 0:
                            line_list[r] += ','.join(list(map(str, [' ', r + 1, w_list[c][r]])))
                        else:
                            line_list[r] += ','.join(list(map(str, [r + 1, w_list[c][r]])))

                for line in line_list:
                    file.write(line + '\n')

        elif xy_type == 'XYY':
            w_fn += 'final_ab_w_ecm{0}n{1}_XYY.txt'.format(*ecm_tuple)
            with open(w_fn, 'a+') as file:
                """
                因为Origin中画Waterfall需要XYY...格式的数据，所以此处不能把每个sub_w_list打印到一行里
                for w in w_list:
                    line = ','.join(list(map(str, w))) + '\n'
                    file.write(line)
                按照Origin-WaterFall-XYY的数据格式要求，第一列 X = list(range(1,161)),其余每列一个 y(sub_w_list,150个)，一共1 + 150 = 151 列
                file row = 160， col = 151
                """
                line_list = ['' for i in range(len(w_list[0]))]
                for c in range(len(w_list)):
                    # for r, line in enumerate(line_list):
                    #     line += ','.join(list(map(str, [r+1, w_list[c][r]])))
                    for r in range(len(w_list[0])):
                        if c > 0:
                            # line_list[r] += ','.join(list(map(str, [' ', r + 1, w_list[c][r]])))
                            line_list[r] += ',' + str(w_list[c][r])
                        else:
                            line_list[r] += ','.join(list(map(str, [r + 1, w_list[c][r]])))

                for line in line_list:
                    file.write(line + '\n')
        # c_fn = get_date_prefix() + 'final_ab_c_ecm{0}n{1}.txt'.format(*ecm_tuple)
        # with open(c_fn, 'a+') as file:
        #     file.write(','.join(list(map(str, c_list))))
    else:
        # return w_list, c_list
        return w_list

# ECM- 2 / 4 / 5 / 6 / 7 / 8 / 9 (7 kinds)
def generate_separated_weight_2_txt():
    # 每条EIS曲线有80个点，但是输入模型的时候要把一个点的横纵坐标分别输入，ML的输入就变成80*2=160个
    # separated_weight是指这160个输入的各自的权重
    ecm_kinds_list = [2,4,5,6,7,8,9]
    for i, ni in enumerate(ecm_kinds_list):
        for j, nj in enumerate(ecm_kinds_list):
            if j > i:
                # ecm_tuple = (2, 4)
                ecm_tuple = (ni, nj)
                xy_type = 'XYY'
                parse_final_AB_model(fp = './models/trained_on_TV_tested_on_test',\
                                     fn = '2020_07_06_ab_final_boost_num=150_3_pickle.file',
                                     ecm_tuple = ecm_tuple,
                                     xy_type = xy_type,
                                     output_flag=True)
"""
Generated R(RC)_IS_lin-kk_res.txt files are stored at post_process_res/separated_weights
W-list      
                Maximum                 Minimum
    ECM 2 / 4   1.2286873955285673      -0.8538582897930005 (-0.691441324880619)
    ECM 2 / 5   1.231620491665156       -0.7097916371153025
    ECM 2 / 6   1.6695083981402656      -2.069994930802966
    ECM 2 / 7   1.3061688566264549      -1.1275552218362166
    ECM 2 / 8   1.2846676127965773      -0.9572276746329311
    ECM 2 / 9   6.6767520592138405      -10.423721871453507
    ECM 4 / 5   7.150513508365039       -7.8360579374884365
    ECM 4 / 6   6.2520502526010375      -1.7292873646905333
    ECM 4 / 7   5.199055104497198       -6.068369102866692
    ECM 4 / 8   5.4833257861301465      -1.8803343903144132
    ECM 4 / 9   1.4141858043023863      -1.3500355596728713
    ECM 5 / 6   10.030247035481148      -5.6209205326558465
    ECM 5 / 7   8.984031751312173       -2.65402456545625
    ECM 5 / 8   1.7593309975728453      -1.7914499589046688
    ECM 5 / 9   1.3266734301935803      -1.1287072308372883
    ECM 6 / 7   8.045020121404317       -4.197971369925129
    ECM 6 / 8   1.3402402334309376      -5.452679905432866
    ECM 6 / 9   0.973580466650771       -1.1406943672653918
    ECM 7 / 8   1.8245532314564694      -4.966007674354187
    ECM 7 / 9   1.5332432282757698      -1.7220298428022391
    ECM 8 / 9   1.506420395424859       -2.6105418189443332
    
    在设置Y轴范围时，均向上或向下近似如（小数点后一位）：
        ECM 2 / 4
            Maximum = 1.2286873955285673    =》  1.3
            Minimum = -0.8538582897930005   =》  -0.9
C-list      
    Maximum                 Minimum
    8.590257762273243       1.3701090366988735
"""
# generate_separated_weight_2_txt()

def generate_merged_weight_2_txt(fp, fn):
    # 每条EIS曲线有80个点，但是输入模型的时候要把一个点的横纵坐标分别输入，ML的输入就变成 80 * 2 = 160 个
    # separated_weight 是指这 160 个输入的各自的权重
    # merged_weight 是指每个坐标的实部和虚部的权重的和
    # 函数的流程基本 和 函数 parse_final_AB_model 一致
    with open(os.path.join(fp, fn), 'rb') as file:
        ab_model = pickle.load(file)

    ecm_kinds_list = [2,4,5,6,7,8,9]
    for i, ni in enumerate(ecm_kinds_list):
        for j, nj in enumerate(ecm_kinds_list):
            if j > i:
                ecm_tuple = (ni, nj)

                w_list = []
                for ab in ab_model:
                    w_list.append(ab['model'][ecm_tuple]['w'])

                merged_w_list = []
                for w_l in w_list:
                    merged_w_l = [w_l[i * 2] + w_l[i * 2 + 1] for i in range(int(len(w_l) / 2))]
                    merged_w_list.append(merged_w_l)

                print('ECM {0} / {1} : Merged Max W = {2}, Min W = {3}'.format(ni, nj, max(max(merged_w_list)), min(min(merged_w_list))))
                merged_w_fn = get_date_prefix() + 'final_ab_merged_w_ecm{0}n{1}_XYY.txt'.format(*ecm_tuple)
                with open(merged_w_fn, 'a+') as file:
                    line_list = ['' for i in range(len(merged_w_list[0]))]
                    for c in range(len(merged_w_list)):
                        for r in range(len(merged_w_list[0])):
                            if c > 0:
                                line_list[r] += ',' + str(merged_w_list[c][r])
                            else:
                                line_list[r] += ','.join(list(map(str, [r + 1, merged_w_list[c][r]])))

                    for line in line_list:
                        file.write(line + '\n')
"""
Generated R(RC)_IS_lin-kk_res.txt files are stored at post_process_res/merged_weights
    ECM 2 / 4 : Merged Max W = 1.4897039845265616, Min W = -0.6138024244886248
    ECM 2 / 5 : Merged Max W = 1.3232109892748483, Min W = -0.7406524859123116
    ECM 2 / 6 : Merged Max W = 1.8605100408433375, Min W = -0.6314735930560225
    ECM 2 / 7 : Merged Max W = 2.1324019203391495, Min W = -1.4697585866536205
    ECM 2 / 8 : Merged Max W = 2.4819007503859956, Min W = -0.7617532766868721
    ECM 2 / 9 : Merged Max W = 11.448519163765162, Min W = -13.864866933113312
    ECM 4 / 5 : Merged Max W = 2.1396823955335025, Min W = -8.965356269781685
    ECM 4 / 6 : Merged Max W = 7.9548916671385, Min W = -9.81725411395834
    ECM 4 / 7 : Merged Max W = 8.55367005359613, Min W = -6.660601911478098
    ECM 4 / 8 : Merged Max W = 5.562215792010497, Min W = -2.0271880739978814
    ECM 4 / 9 : Merged Max W = 1.218018376488303, Min W = -1.244132157159707
    ECM 5 / 6 : Merged Max W = 11.923136566920665, Min W = -1.9902675670196697
    ECM 5 / 7 : Merged Max W = 10.281292038559874, Min W = -6.8752101012606985
    ECM 5 / 8 : Merged Max W = 5.485153271158135, Min W = -1.1367404962991654
    ECM 5 / 9 : Merged Max W = 1.584608555854077, Min W = -0.7756276447376509
    ECM 6 / 7 : Merged Max W = 7.261564363092191, Min W = -4.30766175795962
    ECM 6 / 8 : Merged Max W = 4.868369591115451, Min W = -4.934668878736391
    ECM 6 / 9 : Merged Max W = 1.3884785026692876, Min W = -1.2625595634835216
    ECM 7 / 8 : Merged Max W = 2.8746950247170724, Min W = -1.0939633597827692
    ECM 7 / 9 : Merged Max W = 2.0751889323351564, Min W = -2.1950109877199147
    ECM 8 / 9 : Merged Max W = 1.1312734849467994, Min W = -2.7638820286047103
"""
# generate_merged_weight_2_txt(fp = './models/trained_on_TV_tested_on_test',\
#                              fn = '2020_07_06_ab_final_boost_num=150_3_pickle.file')

def divide_Zimg_Zreal_Ws():
    """
    Function
        One LRC in AdaBoost has 160 weights (= 80 points * 2), divide Zreal's weight and Zimg's weight
    :return:
    """
    # 1- Get all the weights
    ecm_kinds_list = [2,4,5,6,7,8,9]
    for i, ni in enumerate(ecm_kinds_list):
        for j, nj in enumerate(ecm_kinds_list):
            # if j > i:
            if (j > i) and (i >= 4):
                # ecm_tuple = (2, 4)
                ecm_tuple = (ni, nj)
                xy_type = 'XYY'
                w_list = parse_final_AB_model(fp='./models/trained_on_TV_tested_on_test', \
                                     fn='2020_07_06_ab_final_boost_num=150_3_pickle.file',
                                     ecm_tuple=ecm_tuple,
                                     xy_type=xy_type,
                                     output_flag=False)
                """
                2- Draw
                    2.1- For one kind binary (take ecm2/4 for example) LRC in AdaBoost, there 150 binary LRC, each is represented 
                    by a plot, so there is going to be 21 (combination of ECMi/j) * 150 (number of binary LRC in ECMi/j) = 3150
                    2.2- Plot details
                        1- Find minimum and maximum in a W to set the limit of y-axis
                        2- Use Red to represent Zreal, Blue to represent Zimg
                        3- Use the same img size on each img
                        4- Set naming rules
                """

                wide_pix = 150  # 150 * 100 pix
                high_pix = 100  # 100 * 100 pix
                # plt.figure(figsize=(wide_pix, high_pix))
                # 10 rows, 15 cols == 150
                nrow = 10
                ncol = 15
                fig, ax = plt.subplots(nrow, ncol, figsize=(wide_pix, high_pix))

                x_list = [a for a in range(80)] # 0 ~ 79
                y_list = [0 for d in range(80)] # line y = 0
                for b, w in enumerate(w_list):
                    # 0-based index, odd_index_w = Zimg; even_index_w = Zreal
                    odd_index_w_list = []
                    even_index_w_list = []

                    min_w, max_w = min(w) - 0.1 , max(w) + 0.1

                    row = int(b / ncol)
                    col = b % ncol

                    for c, a in enumerate(w):
                        if c % 2 == 0:
                            even_index_w_list.append(a)
                        elif c % 2 == 1:
                            odd_index_w_list.append(a)

                    ax[row][col].plot(x_list, even_index_w_list, color='Red', linestyle='-', label='Zreal')
                    ax[row][col].plot(x_list, odd_index_w_list, color='Blue', linestyle='-', label='Zimg')
                    ax[row][col].plot(x_list, y_list, color='Black', linestyle='-.')
                    ax[row][col].set_ylim([min_w, max_w])
                    # fontsize : int or float or {‘xx-small’, ‘x-small’, ‘small’, ‘medium’, ‘large’, ‘x-large’, ‘xx-large’}
                    ax[row][col].legend(loc='upper left', shadow=True, fontsize='x-large')
                plt.savefig(get_date_prefix()+'final_AB_Zreal_Zimg_Ws_on_ECM{0}n{1}.tiff'.format(ecm_kinds_list[i], ecm_kinds_list[j]))
                print('Plot for Multiple-Class LRC on ECM{0}/{1} is done'.format(i, j))
# imgs are stored at dpfc_large_files\ml_sl\adaboost
# divide_Zimg_Zreal_Ws()

def abs_merged_weights():
    """
    Function
        取merged_weights所有权重的绝对值
    :return:
    """
    fp = 'post_process_res/merged_weights'
    fn_list = os.listdir(fp)
    for fn in fn_list:
        if fn.endswith('txt'):
            # mid_fn: like 'final_ab_merged_w_ecm2n4'
            # mid_fn = fn.split('.')[0].split('2020_09_14_')[1].split('_XYY')[0]
            # mid_fn: like 'ecm2n4'
            mid_fn = fn.split('.')[0].split('_')[-2]
            new_file_line = ''
            with open(os.path.join(fp, fn), 'r') as file:
                for line in file.readlines():
                    line_str_list = line.strip().split(',')
                    point_seq = int(line_str_list[0])
                    abs_data_list = [str(abs(float(num_str))) for num_str in line_str_list[1:]]
                    abs_data_list.insert(0, str(point_seq))
                    new_file_line += ','.join(abs_data_list) + '\n'

            new_fn = get_date_prefix()+'final_ab_abs(merged_w)_'+mid_fn+'.txt'
            with open(new_fn, 'a+') as new_file:
                new_file.write(new_file_line)
# abs_merged_weights()

def abs_separated_weights():
    """
    Function
        取 separated_weights 所有权重的绝对值
    :return:
    """
    fp = 'post_process_res/separated_weights'
    fn_list = os.listdir(fp)
    for fn in fn_list:
        if fn.endswith('txt'):
            # mid_fn: like 'final_ab_merged_w_ecm2n4'
            # mid_fn = fn.split('.')[0].split('2020_09_14_')[1].split('_XYY')[0]
            # mid_fn: like 'ecm2n4'
            mid_fn = fn.split('.')[0].split('_')[-2]
            new_file_line = ''
            with open(os.path.join(fp, fn), 'r') as file:
                for line in file.readlines():
                    line_str_list = line.strip().split(',')
                    point_seq = int(line_str_list[0])
                    abs_data_list = [str(abs(float(num_str))) for num_str in line_str_list[1:]]
                    abs_data_list.insert(0, str(point_seq))
                    new_file_line += ','.join(abs_data_list) + '\n'

            new_fn = get_date_prefix()+'final_ab_abs(separated_w)_'+mid_fn+'.txt'
            with open(new_fn, 'a+') as new_file:
                new_file.write(new_file_line)
# abs_separated_weights()

def cal_merged_weights_Avg_Variance():
    """
    Function
        计算合并的权重的平均值和方差
    :return:
        txt
            Header:
                EIS point sequence, Average of 150 merged weights for that point, Variance of 150 merged weights for that point
            line:   1,                  Avg,                                            Var
                    2,                  Avg,                                            Var
                    ...
            fn
                date+_final_ab_merged_w_ecm2n4_Avg_Var.txt
    """
    # fp = 'post_process_res/merged_weights/'
    fp = 'post_process_res/merged_weights/abs(merged_weights)'
    fn_list = os.listdir(fp)
    for fn in fn_list:
        if fn.endswith('txt'):
            # mid_fn: like 'final_ab_merged_w_ecm2n4'
            # mid_fn = fn.split('.')[0].split('2020_09_14_')[1].split('_XYY')[0]
            # 2021_01_08_final_ab_abs(merged_w)_ecm2n4.txt --> final_ab_abs(merged_w)_ecm2n4
            mid_fn = fn.split('.')[0].split('2021_01_08_')[1]
            new_file_line = 'EIS_Point_Seq,Average,Variance\n'
            with open(os.path.join(fp, fn), 'r') as file:
                for line in file.readlines():
                    line_str_list = line.strip().split(',')
                    point_seq = int(line_str_list[0])
                    data_list = [float(num_str) for num_str in line_str_list[1:]]
                    # print(mid_fn,'Maximum:',max(data_list),'Minimum:',min(data_list))
                    data_avg = sum(data_list) / len(data_list)

                    data_var = sum([(d - data_avg) ** 2 for d in data_list])/len(data_list)

                    new_file_line += ','.join([str(point_seq), str(data_avg), str(data_var)]) + '\n'

            new_fn = get_date_prefix()+mid_fn+'_Avg_Var.txt'
            with open(new_fn, 'a+') as new_file:
                new_file.write(new_file_line)
# cal_merged_weights_Avg_Variance()

# separated_weights_avg_variance
def cal_separated_weights_Avg_Variance():
    """
    Function
        计算合并的权重的平均值和方差
    :return:
        txt
            Header:
                EIS point sequence, Average of 150 separated weights for that point, Variance of 150 separated weights for that point
            line:   1,                  Avg,                                            Var
                    2,                  Avg,                                            Var
                    ...
            fn
                date+_final_ab_separated_w_ecm2n4_Avg_Var.txt
    """
    # fp = 'post_process_res/separated_weights'
    fp = 'post_process_res/separated_weights/abs(seperated_weights)'
    fn_list = os.listdir(fp)
    for fn in fn_list:
        if fn.endswith('txt'):
            # mid_fn: like 'final_ab_merged_w_ecm2n4'
            # mid_fn = fn.split('.')[0].split('2020_09_13_')[1].split('_XYY')[0]

            # 2021_01_09_final_ab_abs(separated_w)_ecm2n4.txt --》 final_ab_abs(separated_w)_ecm2n4
            mid_fn = fn.split('.')[0].split('2021_01_09_')[1]
            new_file_line = 'EIS_Point_Seq,Average,Variance\n'
            data_avg_list = []
            with open(os.path.join(fp, fn), 'r') as file:
                for line in file.readlines():
                    line_str_list = line.strip().split(',')
                    point_seq = int(line_str_list[0])
                    data_list = [float(num_str) for num_str in line_str_list[1:]]
                    data_avg = sum(data_list) / len(data_list)
                    data_avg_list.append(data_avg)

                    data_var = sum([(d - data_avg) ** 2 for d in data_list])/len(data_list)

                    new_file_line += ','.join([str(point_seq), str(data_avg), str(data_var)]) + '\n'

            new_fn = get_date_prefix()+mid_fn+'_Avg_Var.txt'
            # print(mid_fn, 'Maximum:', max(data_avg_list), 'Minimum:', min(data_avg_list))
            print(mid_fn+'_Avg_Var', 'Maximum:', max(data_avg_list), 'Minimum:', min(data_avg_list))
            with open(new_fn, 'a+') as new_file:
                new_file.write(new_file_line)
# cal_separated_weights_Avg_Variance()
"""
final_ab_w_ecm2n4 Maximum: 0.8967434988914359 Minimum: -0.5847481653381142
final_ab_w_ecm2n5 Maximum: 0.7194534630396248 Minimum: -0.3070149969560082
final_ab_w_ecm2n6 Maximum: 0.7816862196969737 Minimum: -0.4759964770315161
final_ab_w_ecm2n7 Maximum: 0.991871496756411 Minimum: -0.8707391988276946
final_ab_w_ecm2n8 Maximum: 0.8448547975772746 Minimum: -0.6266170085977683
final_ab_w_ecm2n9 Maximum: 2.088947012470658 Minimum: -1.6900646639750185
final_ab_w_ecm4n5 Maximum: 1.7588937684592092 Minimum: -2.4035580564297923
final_ab_w_ecm4n6 Maximum: 2.460111701959032 Minimum: -3.6710791619248084
final_ab_w_ecm4n7 Maximum: 1.1589589298084972 Minimum: -1.4166585609081643
final_ab_w_ecm4n8 Maximum: 1.648558660792289 Minimum: -1.2412418827951943
final_ab_w_ecm4n9 Maximum: 0.7177467218652457 Minimum: -0.9384968788217058
final_ab_w_ecm5n6 Maximum: 1.5894409836637209 Minimum: -1.6042965276370489
final_ab_w_ecm5n7 Maximum: 0.8421032129766365 Minimum: -1.006472982394856
final_ab_w_ecm5n8 Maximum: 0.7118728957527239 Minimum: -1.0057592617050168
final_ab_w_ecm5n9 Maximum: 0.4111336255661058 Minimum: -0.600645256288394
final_ab_w_ecm6n7 Maximum: 1.3536520469618523 Minimum: -1.4228944220324635
final_ab_w_ecm6n8 Maximum: 0.9870224769969578 Minimum: -1.0003997935575855
final_ab_w_ecm6n9 Maximum: 0.6037144036676885 Minimum: -0.7905402440346834
final_ab_w_ecm7n8 Maximum: 1.177952776176398 Minimum: -0.8287766910596268
final_ab_w_ecm7n9 Maximum: 0.9912845280032593 Minimum: -0.7635886896367426
final_ab_w_ecm8n9 Maximum: 0.7796687549772284 Minimum: -0.7212858446343935
"""
"""
final_ab_abs(separated_w)_ecm2n4_Avg_Var Maximum: 0.8967434988914359 Minimum: 0.2310171396473721
final_ab_abs(separated_w)_ecm2n5_Avg_Var Maximum: 0.7194534630396248 Minimum: 0.24022040188690866
final_ab_abs(separated_w)_ecm2n6_Avg_Var Maximum: 0.7816862196969737 Minimum: 0.2417360242064718
final_ab_abs(separated_w)_ecm2n7_Avg_Var Maximum: 0.991871496756411 Minimum: 0.2537568154566094
final_ab_abs(separated_w)_ecm2n8_Avg_Var Maximum: 0.8455478576388838 Minimum: 0.27410807425911177
final_ab_abs(separated_w)_ecm2n9_Avg_Var Maximum: 2.8469697792218045 Minimum: 0.2696729926471715
final_ab_abs(separated_w)_ecm4n5_Avg_Var Maximum: 2.466461716149416 Minimum: 0.43281344458159926
final_ab_abs(separated_w)_ecm4n6_Avg_Var Maximum: 4.289717804548505 Minimum: 0.4741096315271348
final_ab_abs(separated_w)_ecm4n7_Avg_Var Maximum: 1.5560957222501939 Minimum: 0.4910644290325314
final_ab_abs(separated_w)_ecm4n8_Avg_Var Maximum: 1.6735269200939338 Minimum: 0.2651294952341402
final_ab_abs(separated_w)_ecm4n9_Avg_Var Maximum: 0.9384968788217058 Minimum: 0.25690565001925997
final_ab_abs(separated_w)_ecm5n6_Avg_Var Maximum: 2.371134554388526 Minimum: 0.4430234552847925
final_ab_abs(separated_w)_ecm5n7_Avg_Var Maximum: 1.0724990522834879 Minimum: 0.37187436490411857
final_ab_abs(separated_w)_ecm5n8_Avg_Var Maximum: 1.021782191009095 Minimum: 0.296189475916756
final_ab_abs(separated_w)_ecm5n9_Avg_Var Maximum: 0.6095462877529259 Minimum: 0.24420154468675553
final_ab_abs(separated_w)_ecm6n7_Avg_Var Maximum: 1.6562627687190636 Minimum: 0.3434101724478762
final_ab_abs(separated_w)_ecm6n8_Avg_Var Maximum: 1.0532766571800427 Minimum: 0.3034056328469064
final_ab_abs(separated_w)_ecm6n9_Avg_Var Maximum: 0.7915518975073113 Minimum: 0.25134048312897483
final_ab_abs(separated_w)_ecm7n8_Avg_Var Maximum: 1.27692290125542 Minimum: 0.36471452586894315
final_ab_abs(separated_w)_ecm7n9_Avg_Var Maximum: 1.0081027782154508 Minimum: 0.2465157512513725
final_ab_abs(separated_w)_ecm8n9_Avg_Var Maximum: 0.8463835295639245 Minimum: 0.29440508844685603
"""

def plot_ABS_Weight_Avg_Var(res_fp):
    # 1- read files:
    fn_list = os.listdir(res_fp)
    for fn in fn_list:
        if fn.endswith('.txt'):
            print('Reading:', fn)
            x_list = []
            avg_list = []
            var_list = []
            with open(os.path.join(res_fp, fn), 'r') as file:
                for line in file.readlines()[1:]:
                    line_str_list = line.strip().split(',')
                    x_list.append(int(line_str_list[0]))
                    avg_list.append(float(line_str_list[1]))
                    var_list.append(float(line_str_list[2]))

            # 2- plot R(RC)_IS_lin-kk_res.txt in each file
            x_arr = np.array(x_list)
            avg_arr = np.array(avg_list)
            var_arr = np.array(var_list)

            my_shareX_2Y_plot_4_AB_1(x_arr=x_arr, y1_arr=avg_arr, y2_arr=var_arr)
plot_ABS_Weight_Avg_Var(res_fp='post_process_res/separated_weights/abs(seperated_weights)/abs(seperated_weights)_avg_var')