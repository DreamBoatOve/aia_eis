import sys
sys.path.append('../../')

import numpy as np
import matplotlib.pyplot as plt
import lime
import lime.lime_tabular

from ml_sl.rf.rf_0 import RF, save_random_forest, load_random_forest
from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from ml_sl.ml_data_wrapper import pack_list_2_list

from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset, get_TV_T_dataset
from utils.file_utils.filename_utils import get_date_prefix
from utils.post_process.ml_post_process import single_para_sorter
from utils.visualize_utils.ml_boxplot_utils import rf_boxplot
from utils.file_utils.pickle_utils import load_pickle_file
from utils.visualize_utils.shareX_2Y_plots import my_shareX_2Y_plot_4_RF_0,my_shareX_2Y_plot_4_RF_1
from utils.post_process.lime_utils import lime_explain_1Kind_RF_instances

label_list = [2,4,5,6,7,8,9]
"""
随机森林的可调变量只有树的数量
    树的数量：1，50，100，150，200，250，300，350，400，450
    当树的数量=1：
        和决策树比较，估计决策树的性能会好一些
    当树的数量=450
        避免超过训练集的样本数量
"""
# 1-使用训练集训练RF，在验证集上测试性能
def rf_tr_va(training_dataset, validation_dataset, label_list=[2,4,5,6,7,8,9]):
    tree_num_list = [1] + [50 * i for i in range(1, 10)]
    # tree_num_list = [t for t in tree_num_list if t >= 200]
    va_label_list = [d[0] for d in validation_dataset]
    va_data_list = [d[1] for d in validation_dataset]

    # 1.1-每种RF（树的数量不同）要重复10遍，求平均性能
    for t_n in tree_num_list:
        for i in range(10):
            rf = RF(unlabeled_dataset_list = va_data_list, labeled_dataset_list = training_dataset,\
                    label_list = label_list, tree_num=t_n)
            rf.create_forest()
            sample_label_prob_dict_list = rf.classify()
            # 1.2- 保存树及其性能
            # create a name for rf, like date_rf_tree=num_pickle_i.file
            rf_name = get_date_prefix()+'rf_tree='+str(t_n)+'_pickle_'+str(i)+'.file'

            # calculate accuracy and kappa
            acc = cal_accuracy(sample_label_prob_dict_list, va_label_list)
            kappa = cal_kappa(sample_label_prob_dict_list, va_label_list)

            # record rf and its name, tree_num, accuracy, and kappa
            save_random_forest(random_forest = rf, filename = rf_name, filepath = './')

            res_file = get_date_prefix()+'rf_res.txt'
            with open(res_file, 'a+') as file:
                line_str = ','.join([rf_name, str(t_n), str(acc), str(kappa)]) + '\n'
                file.write(line_str)

# 2-手动确定树的数量
# single_para_sorter(table_start_row=3, table_start_col=0, sheet_name='RF', excel_abs_path='../ml_training_records.xlsx', txt_filename='rf_avg_res.txt')
# 使用箱型图可视化使用不同参数的模型的平均性能
# plot_para_dict = {'title':'Randon Forest',
#                     'x_label': 'Tree Number',
#                     'y_label': 'Accuracy + Kappa',
#                     'figsize': (16, 9)
#                   }
# rf_boxplot(table_start_row=3, table_start_col=0, sheet_name='RF',\
#            excel_abs_path='../ml_training_records.xlsx', plot_para_dict=plot_para_dict)
# if __name__ == '__main__':
#     training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
#     rf_tr_va(training_dataset, validation_dataset)

# 3-使用训练+验证集训练RF，在测试集上测试性能
def duplication_dector(data_list):
    """
    Function:
        Detect the duplication data:
            [8, [0.0, 0.9813483522855719, 0.009644051584596552, 0.9664039611606744, 0.024045316683273095, 0.9564410337003102, 0.043203795427326774, 0.9514595700357766, 0.057605060526003296, 0.9414966425754125, 0.07676353927005697, 0.9365151789108791, 0.11506753431565037, 0.9290429832171331, 0.1342130506171901, 0.9265522514505149, 0.16293780629075666, 0.9215707876546843, 0.1916625619643232, 0.9165893239901508, 0.201254763778864, 0.9116078603256174, 0.2300054442061613, 0.9016449328652532, 0.2491898478352429, 0.8916820054048891, 0.28751976776586413, 0.8792283460466096, 0.32109247405110825, 0.8617932231550939, 0.3355455889198406, 0.8418673682343656, 0.33561040113241036, 0.8294137088760862, 0.3691831074176545, 0.8119785858532733, 0.3979856376150076, 0.7920527310638421, 0.4123869027136841, 0.7820898036034779, 0.43638036962690147, 0.7671454124785804, 0.46039976142514655, 0.7472195575578522, 0.479636014824284, 0.727293702768421, 0.5037331812776128, 0.6924234567227951, 0.5517460399890752, 0.6575532106771694, 0.5903352085073812, 0.5952849141483664, 0.6240893889878205, 0.5429795452112248, 0.6817722240376103, 0.4881834442448707, 0.744290047387861, 0.4234244159494495, 0.8021154693053042, 0.34123026463121553, 0.8600186658778309, 0.24409172205678675, 0.9275529514611432, 0.13449952038667282, 1.0, 0.0, 0.9662328570770469, 0.05479610083505697, 0.951701967684528, 0.08966634688068287, 0.893824695997029, 0.18182342565928095, 0.8793586186857827, 0.20424001234662748, 0.8262385606436609, 0.3013785549210562, 0.8117724833324147, 0.3237951416084026, 0.7780053405407584, 0.3785912424434596, 0.7635522256720262, 0.39851709736418783, 0.7202058433770461, 0.45580393009716014, 0.6624841209997144, 0.518072226625963, 0.6095844243490325, 0.5728683274610199, 0.5661991548578076, 0.637627355756441, 0.5133253830921536, 0.6874419929269645, 0.4892800664088805, 0.7123493115122264, 0.4892800664088805, 0.7123493115122264, 0.4652347497256074, 0.737256630097488, 0.45078163485687506, 0.7571824850182163, 0.4268011303861717, 0.7696361443764957, 0.3884063982429808, 0.7945434629617575, 0.35480776707270867, 0.8169600496491038, 0.3355715138048685, 0.8368859044385349, 0.3115650844491372, 0.854321027461348, 0.2875716175359199, 0.8692654187175426, 0.2539729863656479, 0.8916820054048891, 0.23478858273656633, 0.9016449328652532, 0.19170144929186506, 0.9091171284277021, 0.16298965606081242, 0.9116078603256174, 0.1533844918037577, 0.9190800558880661, 0.10073108142954357, 0.9265522514505149, 0.07200632575597699, 0.9315337151150485, 0.04803878359649042, 0.9414966425754125, 0.028854380098706, 0.9514595700357766, 0.014440152426218378, 0.963913229262759, 0.009644051584596552, 0.9664039611606744, 0.0, 0.9813483522855719, 0.024058279125787046, 0.9539503018023948, 0.04803878359649042, 0.9414966425754125, 0.09593498045662457, 0.9290429832171331, 0.1342130506171901, 0.9265522514505149, 0.21560417910748478, 0.9116078603256174, 0.301843258209457, 0.8842098098424404, 0.3259404246627859, 0.8493395637968144, 0.3643740440022216, 0.8169600496491038, 0.39320249921589967, 0.7920527310638421, 0.4459984963264701, 0.7571824850182163, 0.4844191533546889, 0.727293702768421, 0.5228786975791525, 0.6899327248248799]]
            [4, [0.0, 0.9813483522855719, 0.009644051584596552, 0.9664039611606744, 0.024045316683273095, 0.9564410337003102, 0.043203795427326774, 0.9514595700357766, 0.057605060526003296, 0.9414966425754125, 0.07676353927005697, 0.9365151789108791, 0.11506753431565037, 0.9290429832171331, 0.1342130506171901, 0.9265522514505149, 0.16293780629075666, 0.9215707876546843, 0.1916625619643232, 0.9165893239901508, 0.201254763778864, 0.9116078603256174, 0.2300054442061613, 0.9016449328652532, 0.2491898478352429, 0.8916820054048891, 0.28751976776586413, 0.8792283460466096, 0.32109247405110825, 0.8617932231550939, 0.3355455889198406, 0.8418673682343656, 0.33561040113241036, 0.8294137088760862, 0.3691831074176545, 0.8119785858532733, 0.3979856376150076, 0.7920527310638421, 0.4123869027136841, 0.7820898036034779, 0.43638036962690147, 0.7671454124785804, 0.46039976142514655, 0.7472195575578522, 0.479636014824284, 0.727293702768421, 0.5037331812776128, 0.6924234567227951, 0.5517460399890752, 0.6575532106771694, 0.5903352085073812, 0.5952849141483664, 0.6240893889878205, 0.5429795452112248, 0.6817722240376103, 0.4881834442448707, 0.744290047387861, 0.4234244159494495, 0.8021154693053042, 0.34123026463121553, 0.8600186658778309, 0.24409172205678675, 0.9275529514611432, 0.13449952038667282, 1.0, 0.0, 0.9662328570770469, 0.05479610083505697, 0.951701967684528, 0.08966634688068287, 0.893824695997029, 0.18182342565928095, 0.8793586186857827, 0.20424001234662748, 0.8262385606436609, 0.3013785549210562, 0.8117724833324147, 0.3237951416084026, 0.7780053405407584, 0.3785912424434596, 0.7635522256720262, 0.39851709736418783, 0.7202058433770461, 0.45580393009716014, 0.6624841209997144, 0.518072226625963, 0.6095844243490325, 0.5728683274610199, 0.5661991548578076, 0.637627355756441, 0.5133253830921536, 0.6874419929269645, 0.4892800664088805, 0.7123493115122264, 0.4892800664088805, 0.7123493115122264, 0.4652347497256074, 0.737256630097488, 0.45078163485687506, 0.7571824850182163, 0.4268011303861717, 0.7696361443764957, 0.3884063982429808, 0.7945434629617575, 0.35480776707270867, 0.8169600496491038, 0.3355715138048685, 0.8368859044385349, 0.3115650844491372, 0.854321027461348, 0.2875716175359199, 0.8692654187175426, 0.2539729863656479, 0.8916820054048891, 0.23478858273656633, 0.9016449328652532, 0.19170144929186506, 0.9091171284277021, 0.16298965606081242, 0.9116078603256174, 0.1533844918037577, 0.9190800558880661, 0.10073108142954357, 0.9265522514505149, 0.07200632575597699, 0.9315337151150485, 0.04803878359649042, 0.9414966425754125, 0.028854380098706, 0.9514595700357766, 0.014440152426218378, 0.963913229262759, 0.009644051584596552, 0.9664039611606744, 0.0, 0.9813483522855719, 0.024058279125787046, 0.9539503018023948, 0.04803878359649042, 0.9414966425754125, 0.09593498045662457, 0.9290429832171331, 0.1342130506171901, 0.9265522514505149, 0.21560417910748478, 0.9116078603256174, 0.301843258209457, 0.8842098098424404, 0.3259404246627859, 0.8493395637968144, 0.3643740440022216, 0.8169600496491038, 0.39320249921589967, 0.7920527310638421, 0.4459984963264701, 0.7571824850182163, 0.4844191533546889, 0.727293702768421, 0.5228786975791525, 0.6899327248248799]]
    :param
        data_list:
            its data structure might be:
                [number, [(a,b), (a,b)]]
                or
                [number, a,b,a,b]
    :return:
    """
    pass

def rf_TV_Te():
    label_list = [2, 4, 5, 6, 7, 8, 9]

    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    counter = 0
    rf_final_res_fn = get_date_prefix() + 'rf_final_res.txt'
    for tree_num in [200, 300, 350, 400, 450]:
        for i in range(10):
            rf = RF(unlabeled_dataset_list = test_data_list,
                    labeled_dataset_list = training_dataset + validation_dataset,
                    label_list = label_list, tree_num = tree_num)
            rf.create_forest()
            sample_label_prob_dict_list = rf.classify()

            # 1.2- 保存树及其性能
            # create a name for rf, like date_rf_tree=num_pickle_i.file
            rf_model_name = get_date_prefix()+'rf_tree='+str(tree_num)+'_pickle_'+str(i)+'.file'

            # calculate accuracy and kappa
            acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
            kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

            # record rf and its name, tree_num, accuracy, and kappa
            save_random_forest(random_forest = rf, filename = rf_model_name, filepath = './')

            with open(rf_final_res_fn, 'a+') as file:
                line_str = ','.join([rf_model_name, str(tree_num), str(i), str(acc), str(kappa), str(acc+kappa)]) + '\n'
                file.write(line_str)
            counter += 1
            print('Finished {0}, {1} left'.format(counter, 50 - counter))
# rf_TV_Te()

def get_rf_acc_on_first_3_predictions():
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    # 1- Load trained RF model
    rf = load_random_forest(filename='2020_06_29_rf_tree=200_pickle_0.file',\
                            filepath='../../../../large_files/dpfc/ml_sl/rf/rf_res/trained_on_TV_tested_on_test')
    rf.unlabeled_dataset_list = test_data_list
    sample_label_prob_dict_list = rf.classify()

    # calculate accuracy and kappa
    acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
    acc_on_2 = cal_accuracy_on_2(sample_label_prob_dict_list, test_label_list)
    acc_on_3 = cal_accuracy_on_3(sample_label_prob_dict_list, test_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

    print('Random Forest: Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(
        acc, acc_on_2, acc_on_3, kappa))
# get_rf_acc_on_first_3_predictions()
# Random Forest: Accuracy on 1 = 0.5824175824175825, Accuracy on 2 = 0.7802197802197802,
#                Accuracy on 3 = 0.8791208791208791, Kappa=0.46010928961748637

def interpret_RF_by_LIME(mode):
    """
    Function
    :param
        mode
            'fig': 把每个数据的解释都画出来
            ‘txt’：把每个数据的解释都存储在对应txt文件中
        labeled_data_list:
            直接输入有标签的数据，才能知道预测的是否正确
    :return:
    """
    global label_list

    # 1- Load trained RF model
    rf = load_random_forest(filename='2020_06_29_rf_tree=200_pickle_0.file',
                            filepath='../../../../dpfc_large_files/ml_sl/rf/rf_res/trained_on_TV_tested_on_test')

    """
    2- Create the explainer
        Refer: https://marcotcr.github.io/lime/tutorials/Tutorial%20-%20continuous%20and%20categorical%20features.html
        tabular explainers need a training set. The reason for this is because we compute statistics on each feature (column). 
        If the feature is numerical, we compute the mean and std, and discretize it into quartiles. 
        If the feature is categorical, we compute the frequency of each value. For this tutorial, we'll only look at numerical features.
        
        We use these computed statistics for two things:
            1- To scale the data, so that we can meaningfully compute distances when the attributes are not on the same scale
            2- To sample perturbed instances - which we do by sampling from a Normal(0,1), multiplying by the std and adding back the mean.
    lime.lime_tabular.LimeTabularExplainer
        __init__ Args:
            training_data: numpy 2d array
            mode: "classification" or "regression"
            training_labels: labels for training data. Not required, but may be used by discretizer.
            feature_names: list of names (strings) corresponding to the columns in the training data.
    """
    # Transform Training+Validation-datasets into 2D numpy array = (n_samples, n_features_dimension)
    # tr_va_dataset = [ [label (int), [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)]], ...]
    tr_va_dataset, test_dataset = get_TV_T_dataset(file_path='../../datasets/ml_datasets/normed')

    # tr_va_reformed_dataset = [[(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)], ...]
    tr_va_reformed_dataset = [d[1] for d in tr_va_dataset]
    tr_va_arr = np.array(pack_list_2_list(tr_va_reformed_dataset))

    feature_names_list = ['point-'+str(n+1) for n in range(tr_va_arr.shape[1])]
    label_str_list = ['ECM'+str(label) for label in label_list]
    lime_explainer = lime.lime_tabular.LimeTabularExplainer(training_data=tr_va_arr, mode="classification",
                                                            feature_names=feature_names_list, class_names=label_str_list,
                                                            discretize_continuous=True)

    """
    3- Explaining an instance
        Since this is a multi-class classification problem, we set the top_labels parameter, so that we only explain the top class
    
    lime.lime_tabular.LimeTabularExplainer
        explain_instance Args:
            data_row: 1d numpy array or scipy.sparse matrix, corresponding to a row
            predict_fn: prediction function. 
                For classifiers, this should be a function that takes a numpy array and outputs prediction probabilities. 
                For regressors, this takes a numpy array and returns the predictions.
                For ScikitClassifiers, this is `classifier.predict_proba()`. 
                For ScikitRegressors, this is `regressor.predict()`. 
                The prediction function needs to work on multiple feature vectors (the vectors randomly perturbed
                from the data_row).
            labels: iterable with labels to be explained.
            top_labels: if not None, ignore labels and produce explanations for
                the K labels with highest prediction probabilities, where K is
                this parameter.
            num_features: maximum number of features present in explanation
            num_samples: size of the neighborhood to learn the linear model
            distance_metric: the distance metric to use for weights.
            model_regressor: sklearn regressor to use in explanation. Defaults to Ridge regression in LimeBase. 
                Must have model_regressor.coef_ and 'sample_weight' as a parameter to model_regressor.fit()
    """
    # 使用整个数据集
    labeled_data_list = tr_va_dataset+test_dataset
    if mode == 'txt':
        lime_explain_txt_fn = get_date_prefix()+'lime_explain_res.txt'
    for labeled_data in labeled_data_list:
        label, data_list = labeled_data
        data_arr = np.array(pack_list_2_list([data_list])).reshape(-1) # data_arr = (160, )
        pre_arr = rf.predict_proba(data_arr.reshape(1, -1)) # pre_arr = (1, 160)

        # 如果RF做出了正确的预测，再保存结果
        pre_label = label_list[np.argsort(pre_arr, axis=1)[0][-1]]
        if label == pre_label:
            print('Predict label correctly', label, ',\tPrediction array:', pre_arr)
            explainer = lime_explainer.explain_instance(data_row=data_arr,predict_fn=rf.predict_proba,
                                                        labels=(1,), top_labels=1,
                                                        num_features=data_arr.shape[0], num_samples=5000,
                                                        distance_metric='euclidean', model_regressor=None)
            if mode == 'fig':
                fig = explainer.as_pyplot_figure(label=explainer.top_labels[0])
                plt.show()
            elif mode == 'txt':
                """
                ans_list
                    [('0.37 < point-2 <= 0.76', 0.007448379167140092), 
                     ('0.33 < point-8 <= 0.74', 0.007098509255750589), ...]
                """
                ans_list = explainer.as_list(label = explainer.top_labels[0])
                with open(lime_explain_txt_fn, 'a+') as file:
                    header = '----------Start----------\n'
                    ecm_str = 'ecm={0}\n'.format(label)
                    line_str = header+ecm_str
                    for ans in ans_list:
                        point_info_str, weight = ans
                        line_str += point_info_str+','+str(weight)+'\n'
                    tail = '----------End----------\n'
                    line_str += tail
                    file.write(line_str)
# interpret_RF_by_LIME(mode='fig')
# interpret_RF_by_LIME(mode='txt')
"""
generate results file: 2021_01_24_lime_explain_res.txt
    file format:
    ----------Start----------
    ecm=4
    point-1 <= 0.00,0.009118543048854835
    point-89 > 0.59,0.006520717675691324
    point-26 <= 0.28,-0.0004287661951428752
    ...
    0.23 < point-139 <= 0.59,4.581205726336095e-05
    point-71 > 0.52,-4.013844195403033e-05
    point-100 <= 0.22,2.6580348141840243e-05
    ----------End----------
    ----------Start----------
    ecm=5
    0.11 < point-23 <= 0.18,0.004791066417417636
    point-148 <= 0.09,0.00477684971170501
    point-89 > 0.59,0.0047605699459234325
    ...
    0.18 < point-35 <= 0.27,0.004494778172658124
    0.17 < point-33 <= 0.26,0.00441815115254862
    point-91 > 0.60,0.0044054511347660025
    ----------End----------
    ...
"""

def plot_RF_Lime_Weight_Avg_Var(fp, fn, w_type):
    """
    :param
        fp:
        fn:
        ecm_num:
        w_type: str
            'w', raw/original weight
            'abs', the Abs(w)
            'positive', only keep the weight with positive value
    :return:
    """
    global label_list
    for label in label_list:
        print(label)
        if w_type == 'abs':
            abs_avg_list, abs_var_list = lime_explain_1Kind_RF_instances(fp, fn, ecm_num=label, w_type=w_type)
            # my_shareX_2Y_plot_4_RF_0(x_arr=np.array(range(len(abs_avg_list))), y1_arr=np.array(abs_avg_list),
            my_shareX_2Y_plot_4_RF_1(x_arr=np.array(range(len(abs_avg_list))), y1_arr=np.array(abs_avg_list),
                              y2_arr=np.array(abs_var_list), w_type=w_type)
        elif w_type == 'positive':
            p_avg_list, p_var_list = lime_explain_1Kind_RF_instances(fp, fn, ecm_num=label, w_type=w_type)
            # my_shareX_2Y_plot_4_RF_0(x_arr=np.array(range(len(p_avg_list))), y1_arr=np.array(p_avg_list),
            my_shareX_2Y_plot_4_RF_1(x_arr=np.array(range(len(p_avg_list))), y1_arr=np.array(p_avg_list),
                              y2_arr=np.array(p_var_list), w_type=w_type)
        elif w_type == 'w':
            avg_list, var_list = lime_explain_1Kind_RF_instances(fp, fn, ecm_num=label, w_type=w_type)
            # my_shareX_2Y_plot_4_RF_0(x_arr=np.array(range(len(avg_list))), y1_arr=np.array(avg_list),
            my_shareX_2Y_plot_4_RF_1(x_arr=np.array(range(len(avg_list))), y1_arr=np.array(avg_list),
                              y2_arr=np.array(var_list), w_type=w_type)
# plot_RF_Lime_Weight_Avg_Var(fp='/rf_res/lime_res', fn='2021_01_24_lime_explain_res.txt',
#                             w_type='w')
plot_RF_Lime_Weight_Avg_Var(fp='../../ml_sl/rf/rf_res/lime_res', fn='2021_01_24_lime_explain_res.txt',
                            w_type='abs')
# plot_RF_Lime_Weight_Avg_Var(fp='../../ml_sl/rf/rf_res/lime_res', fn='2021_01_24_lime_explain_res.txt',
#                             w_type='positive')