import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression

from ml_sl.test.load_testSet import read_testSet, read_testSet_with_label
from ml_sl.ml_data_wrapper import single_point_list_2_list
from ml_sl.logistic.lrc_0 import create_lrc, binary_lrc_classify
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_TV_T_dataset, get_T_V_T_dataset, get_normed_dataset_by_ecm_num

# -------------------------- Compare LRC with LogisticRegression(sklearn.linear_model) on testSet (Machine learning in Action chapter 5) --------------------------
data_i_list, data_j_list = read_testSet(filepath='../../test')
# -------------------------- LogisticRegression(sklearn.linear_model) on testSet --------------------------
def SK_learn_LRC_on_testSet(data_i_list, data_j_list):
    label_list = [0 for i in range(len(data_i_list))] + [1 for i in range(len(data_j_list))]

    data_i_list.extend(data_j_list)
    data_list = [[d[0][0], d[0][1]] for d in data_i_list]

    """
    tol : float, default=1e-4, Tolerance for stopping criteria.
    C :   float, default=1.0
            Inverse of regularization strength; must be a positive float.
            Like in support vector machines, smaller values specify stronger
            regularization.
    class_weight (similar to my ratio): 
            dict or 'balanced', default=None
            Weights associated with classes in the form ``{class_label: weight}``.
            If not given, all classes are supposed to have weight one.
    
            The "balanced" mode uses the values of y to automatically adjust
            weights inversely proportional to class frequencies in the input data
            as ``n_samples / (n_classes * np.bincount(y))``.
    
            Note that these weights will be multiplied with sample_weight (passed
            through the fit method) if sample_weight is specified.
    multi_class : 
            'ovr': One VS Rest
            {'auto', 'ovr', 'multinomial'}, default='auto'
            If the option chosen is 'ovr', then a binary problem is fit for each
            label. For 'multinomial' the loss minimised is the multinomial loss fit
            across the entire probability distribution, *even when the data is
            binary*. 'multinomial' is unavailable when solver='liblinear'.
            'auto' selects 'ovr' if the data is binary, or if solver='liblinear',
            and otherwise selects 'multinomial'.
    """
    sk_lrc = LogisticRegression(penalty='l2', tol=1e-4, C=1.0,\
                                fit_intercept=True, intercept_scaling=1, class_weight='balanced',\
                                random_state=None, solver='liblinear', max_iter=100,\
                                multi_class='ovr')
    # List ==> np.array
    x_arr = np.array(data_list)
    y_arr = np.array(label_list)

    sk_lrc.fit(X=x_arr, y=y_arr)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x0_min, x0_max], [x1_min, x1_max].
    x0_min, x0_max = x_arr[:, 0].min(), x_arr[:, 0].max()
    x1_min, x1_max = x_arr[:, 1].min(), x_arr[:, 1].max()

    # step size in the mesh
    step_size = 0.02
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, step_size), np.arange(x1_min, x1_max, step_size))
    Z = sk_lrc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(1, figsize=(4, 3))
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x_arr[:, 0], x_arr[:, 1], c=y_arr, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.title('SK-Learn-LogisticRegression')
    plt.show()
# -------------------------- LogisticRegression(sklearn.linear_model) on testSet --------------------------

# -------------------------- Mine LogisticRegression on testSet --------------------------
def mine_LRC_on_testSet(data_i_list, data_j_list, max_iter, plot_title, stable_alpha, alpha_init=None):
    w_list, b, ratio = create_lrc(data_i_list, data_j_list, stable_alpha=stable_alpha, max_iter=max_iter, alpha_init=alpha_init)
    label_list = [0 for i in range(len(data_i_list))] + [1 for i in range(len(data_j_list))]
    print('w:{0}, b:{1}'.format(w_list, b))

    data_list = [[d[0][0], d[0][1]] for d in data_i_list + data_j_list]
    d0_list = [d[0] for d in data_list]
    d1_list = [d[1] for d in data_list]
    x0_min, x0_max = min(d0_list), max(d0_list)
    x1_min, x1_max = min(d1_list), max(d1_list)

    # x0_list = [x0_min + (x0_max - x0_min) * i / 1000 for i in range(100)]
    # x1_list = [x1_min + (x1_max - x1_min) * i / 1000 for i in range(100)]
    # x_pair_list = [[x0, x1] for x0, x1 in zip(x0_list, x1_list)]
    # for x_pair in x_pair_list:
    #     z = binary_lrc_classify(w_list=w_list, b=b, ratio=ratio, x_list=x_pair)
    #     mine_Z_list.append(z)

    step_size = 0.02
    xx0, xx1 = np.meshgrid(np.arange(x0_min, x0_max, step_size), np.arange(x1_min, x1_max, step_size))
    xx0_flat = xx0.flatten()
    xx1_flat = xx1.flatten()

    mine_Z_list = []
    ratio = len(data_j_list) / len(data_i_list)
    for x0, x1 in zip(xx0_flat, xx1_flat):
        x_pair = [x0, x1]
        z = binary_lrc_classify(w_list=w_list, b=b, ratio=ratio, x_list=x_pair)
        mine_Z_list.append(z)
    mine_Z_arr = np.array(mine_Z_list).reshape(xx0.shape)

    # plt.figure(1, figsize=(4, 3))
    plt.figure(1, figsize=(8, 6))
    # xx
    plt.pcolormesh(xx0, xx1, mine_Z_arr, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(d0_list, d1_list, c=label_list, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('Sepal length')
    plt.ylabel('Sepal width')

    plt.xlim(x0_min, x0_max)
    plt.ylim(x1_min, x1_max)
    plt.xticks(())
    plt.yticks(())

    plot_title = plot_title + str(max_iter) + ', alpha_init = ' + str(alpha_init)
    # plt.title('LRC: stable alpha, max_iter = {}'.format(max_iter))
    plt.title(plot_title)
    plt.show()
"""
Mine LRC
Stable Alpha
    Max_iter 
        1000: 
            w:[0.43107436726703063, -0.21254604692953194], b:0.7001745813682995
            w:[0.7744522454236528, -0.20743242849322832], b:0.4692740612852083
        5000
            w:[0.4616839140249583, -0.20849779468929427], b:0.6486567275808679
        10000
            w:[0.6496803023664373, -0.2656591475477251], b:1.0698874369408053
            w:[0.4271956029380892, -0.2199189953279137], b:0.7684294794350277
            w:[0.7673647907321551, -0.22611602154468666], b:0.6452182562503187
        50000
            w:[0.5117417217852617, -0.1793314999921118], b:0.3567557735897706
Linear
    Max_iter 1000; alpha_init=0.1
        w:[0.4653680010380619, -0.5956368968591074], b:3.955590256614393
    Max_iter 1000; alpha_init=1
        w:[1.0198555618391554, -1.5561675476247667], b:11.387752391142527
        w:[1.0197567540589054, -1.5559840254108752], b:11.386360464753931
    Max_iter 5000; alpha_init=1
        w:[1.2038024196913788, -1.905236407738687], b:14.021599573413233
在testSet的二分类问题上，
    Stable alpha 效果很差，目测正确率在70%左右
    Linear alpha 效果很好，目测正确率在95%左右，基本和Scikit-learn-LogisticRegression效果一样
"""
# -------------------------- Mine LogisticRegression on testSet --------------------------
# if __name__ == '__main__':
    # SK_learn_LRC_on_testSet(data_i_list, data_j_list)
    # mine_LRC_on_testSet(data_i_list, data_j_list, \
    #          max_iter = 1000, plot_title = 'LRC: stable alpha, max_iter = ', \
    #          stable_alpha=True, alpha_init=None)
    # mine_LRC_on_testSet(data_i_list, data_j_list, \
    #          max_iter=1000, plot_title='LRC: linear alpha, max_iter = ', \
    #          stable_alpha=False, alpha_init=0.1)
# -------------------------- Compare LRC with LogisticRegression(sklearn.linear_model) on testSet (Machine learning in Action chapter 5) --------------------------

# -------------------------- Compare LRC with LogisticRegression(sklearn.linear_model) on EIS dataset --------------------------
def SK_learn_LRC_on_EIS(ecm_num_pair):
    """
    :param
        ecm_num_pair:
            list[ecm_num_a(int), ecm_num_b]
    :return:
    """
    # 1-Load two kinds of EIS data according to ecm_num_pair
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    eis_normed_data_list_0 = get_normed_dataset_by_ecm_num(training_dataset, validation_dataset, test_dataset, ecm_num = ecm_num_pair[0])
    eis_normed_data_list_1 = get_normed_dataset_by_ecm_num(training_dataset, validation_dataset, test_dataset, ecm_num = ecm_num_pair[1])

    eis_normed_data_list_0 = [single_point_list_2_list(d) for d in eis_normed_data_list_0]
    eis_normed_data_list_1 = [single_point_list_2_list(d) for d in eis_normed_data_list_1]
    label_list = [0 for i in range(len(eis_normed_data_list_0))] + [1 for i in range(len(eis_normed_data_list_1))]

    # 2-Transform EIS-data from list to np.array
    eis_normed_arr_0 = np.array(eis_normed_data_list_0)
    eis_normed_arr_1 = np.array(eis_normed_data_list_1)
    eis_arr = np.concatenate((eis_normed_arr_0, eis_normed_arr_1), axis=0)
    label_arr = np.array(label_list)

    sk_lrc = LogisticRegression(penalty='l2', tol=1e-4, C=1.0,\
                                fit_intercept=True, intercept_scaling=1, class_weight='balanced',\
                                random_state=None, solver='liblinear', max_iter=100,\
                                multi_class='ovr')
    sk_lrc.fit(X=eis_arr, y=label_arr)

    # 3-Calculate accuracy of sk_lrc
    eis_prediction_arr = sk_lrc.predict(X=eis_arr)
    # Count the amount of the same number in eis_prediction_arr and label_arr
    correct_count = 0
    for e_p, label in zip(eis_prediction_arr, label_arr):
        # print(e_p, label)
        if e_p - label == 0.0:
            correct_count += 1
    accuracy = correct_count / len(label_list)
    return accuracy

def mine_LRC_on_EIS(ecm_num_pair, stable_alpha=True, max_iter=1000, alpha_init=None):
    # 1-Load two kinds of EIS data according to ecm_num_pair
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
    eis_normed_data_list_0 = get_normed_dataset_by_ecm_num(training_dataset, validation_dataset,\
                                                           test_dataset, ecm_num=ecm_num_pair[0])
    eis_normed_data_list_1 = get_normed_dataset_by_ecm_num(training_dataset, validation_dataset,\
                                                           test_dataset, ecm_num=ecm_num_pair[1])
    label_list = [0 for i in range(len(eis_normed_data_list_0))] + [1 for i in range(len(eis_normed_data_list_1))]

    # 2- Train model
    w_list, b, ratio = create_lrc(data_i_list = eis_normed_data_list_0, data_j_list = eis_normed_data_list_1,\
                                  stable_alpha=stable_alpha, max_iter=max_iter, alpha_init=alpha_init)
    # 3- Classify samples
    pre_list = []
    for e_n_d_list in eis_normed_data_list_0 + eis_normed_data_list_1:
        e_n_d_list = single_point_list_2_list(e_n_d_list)
        prediction = binary_lrc_classify(w_list, b, ratio, x_list = e_n_d_list)
        pre_list.append(prediction)

    # 4- Calculate accuracy
    correct_count = 0
    for pre, label in zip(pre_list, label_list):
        # print(pre, label)
        if pre - label == 0.0:
            correct_count += 1
    accuracy = correct_count / len(label_list)
    return accuracy

if __name__ == '__main__':
    # ecm_num_pair = [4, 9]
    # sk_acc = SK_learn_LRC_on_EIS(ecm_num_pair)
    # print('ecm_num_pair', ecm_num_pair)
    # print('SK-Learn-LR Accuracy:', sk_acc)
    """
    Accuracy
        SK-Learn-LR
            ecm_num_pair[2, 4]
                Accuracy: 1.0
            ecm_num_pair[4, 2]
                Accuracy: 1.0
            ecm_num_pair[2, 9]
                Accuracy: 0.8125, it does not change no matter how many trials
            ecm_num_pair[9, 2]
                Accuracy: 0.8125, the accuracy is the same when the data is input inversely
            ecm_num_pair[5, 4]
                Accuracy: 0.7278911564625851
            ecm_num_pair[6, 4]
                Accuracy: 0.7949640287769785
            ecm_num_pair[7, 4]
                Accuracy: 0.781021897810219
            ecm_num_pair[4, 9]
                Accuracy: 1.0
            ecm_num_pair[9, 4]
                Accuracy: 1.0
        Mine-LRC
            stable_alpha = True
                ecm_num_pair = [9, 4], max_iter = 1000
                    Accuracy: 0.6435643564356436
                ecm_num_pair = [4, 9], max_iter = 1000
                    Accuracy: 0.3564356435643564
                ecm_num_pair = [2, 4], max_iter = 1000
                    Accuracy: 0.9069767441860465
                ecm_num_pair = [4, 2], max_iter = 1000
                    Accuracy: 0.09302325581395349
            stable_alpha = False, linear alpha
                ecm_num_pair = [2, 4], max_iter = 1000, alpha_init = 1
                    Accuracy: 0.9813953488372092
                ecm_num_pair = [4, 2], max_iter = 1000, alpha_init = 1
                    Accuracy: 0.986046511627907, it is DIFFERENT from the above result.
                ecm_num_pair = [4, 9], max_iter = 1000, alpha_init = 1
                    Accuracy: 1.0
                ecm_num_pair = [9, 4], max_iter = 1000, alpha_init = 1
                    Accuracy: 1.0
                ecm_num_pair = [6, 7], max_iter = 1000, alpha_init = 1
                    Accuracy: 0.845679012345679
                ecm_num_pair = [7, 6], max_iter = 1000, alpha_init = 1
                    Accuracy: 0.8395061728395061
                ecm_num_pair = [, ], max_iter = 1000, alpha_init = 1
                ecm_num_pair = [, ], max_iter = 1000, alpha_init = 1
                ecm_num_pair = [, ], max_iter = 1000, alpha_init = 1
                ecm_num_pair = [, ], max_iter = 1000, alpha_init = 1
    """

    ecm_num_pair = [7,6]
    max_iter = 1000

    # stable_alpha = True
    # alpha_init = None

    stable_alpha = False
    alpha_init = 1

    lr_acc = mine_LRC_on_EIS(ecm_num_pair=ecm_num_pair,\
                             stable_alpha=stable_alpha, max_iter=max_iter, alpha_init=alpha_init)
    print('ecm_num_pair', ecm_num_pair)
    print('stable_alpha', stable_alpha, '\nmax_iter', max_iter, '\nalpha_init', alpha_init)
    print('Mine LRC Accuracy:', lr_acc)
# -------------------------- Compare LRC with LogisticRegression(sklearn.linear_model) on EIS dataset --------------------------