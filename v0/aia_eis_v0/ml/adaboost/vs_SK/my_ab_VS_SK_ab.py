import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import AdaBoostClassifier

from ml_sl.test.load_testSet import read_testSet, read_testSet_with_label
from ml_sl.adaboost.ab_0 import AB
from utils.file_utils.filename_utils import get_date_prefix

# -------------------------- Compare my Adaboost with SKlearn-Adaboost (meat larner logistic) on testSet (Machine learning in Action chapter 5) --------------------------
def SK_AB_on_testSet(data_i_list, data_j_list):
    label_list = [0 for i in range(len(data_i_list))] + [1 for i in range(len(data_j_list))]

    data_i_list.extend(data_j_list)
    data_list = [[d[0][0], d[0][1]] for d in data_i_list]

    n_estimators = 200
    # Create and fit an AdaBoosted logistic regression classifer
    sk_ab_lrc = AdaBoostClassifier(LogisticRegression(penalty='l2', tol=1e-4, C=1.0, fit_intercept=True,\
                                                      intercept_scaling=1, class_weight='balanced', random_state=None,\
                                                      solver='liblinear', max_iter=100, multi_class='ovr'),\
                                    algorithm="SAMME", n_estimators=n_estimators)
    # List ==> np.array
    x_arr = np.array(data_list)
    y_arr = np.array(label_list)

    sk_ab_lrc.fit(X=x_arr, y=y_arr)

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x0_min, x0_max], [x1_min, x1_max].
    x0_min, x0_max = x_arr[:, 0].min(), x_arr[:, 0].max()
    x1_min, x1_max = x_arr[:, 1].min(), x_arr[:, 1].max()

    # step size in the mesh
    step_size = 0.02
    xx, yy = np.meshgrid(np.arange(x0_min, x0_max, step_size), np.arange(x1_min, x1_max, step_size))
    Z = sk_ab_lrc.predict(np.c_[xx.ravel(), yy.ravel()])

    # Put the result into a color plot
    Z = Z.reshape(xx.shape)
    # plt.figure(1, figsize=(4, 3))
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx, yy, Z, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(x_arr[:, 0], x_arr[:, 1], c=y_arr, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x')
    plt.ylabel('y')

    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.xticks(())
    plt.yticks(())

    plt.title('SK-Learn-AdaBoost(LogisticRegression) with {} estimator'.format(n_estimators))
    plt.show()

def my_AB_on_testSet(labeled_data_i_list, labeled_data_j_list):
    label_list = [0 for i in range(len(labeled_data_i_list))] + [1 for i in range(len(labeled_data_j_list))]
    labeled_data_list = labeled_data_i_list + labeled_data_j_list

    d0_list = [d[1][0][0] for d in labeled_data_list]
    d1_list = [d[1][0][1] for d in labeled_data_list]
    x0_min, x0_max = min(d0_list), max(d0_list)
    x1_min, x1_max = min(d1_list), max(d1_list)
    step_size = 0.02
    xx0, xx1 = np.meshgrid(np.arange(x0_min - 1, x0_max + 1, step_size), np.arange(x1_min - 1, x1_max + 1, step_size))
    unlabeled_data_list = [[(x0, x1)] for x0, x1 in zip(xx0.flatten().tolist(), xx1.flatten().tolist())]

    boost_num = 50
    ab = AB(boost_num = boost_num, resample_num = 10, alpha_init = 0.1, max_iter = 1000,\
            unlabeled_dataset_list = unlabeled_data_list, labeled_dataset_list = labeled_data_list,\
            label_list = [0,1])
    ab_model_name = get_date_prefix()+'ab_boost_num='+str(boost_num)+'_on_testSet_pickle.file'
    ab.create_ab_classifer(ab_model_name=ab_model_name)
    ab_sample_label_prob_dict_list = ab.classify(ab_model_name=ab_model_name)
    pre_list = []
    for sample_label_prob_dict in ab_sample_label_prob_dict_list:
        for k, v in sample_label_prob_dict.items():
            if v == max(sample_label_prob_dict.values()):
                pre_list.append(k)
                break
    pre_arr = np.array(pre_list).reshape(xx0.shape)

    # 5-Draw plot / Calculate Accuracy
    plt.figure(1, figsize=(8, 6))
    plt.pcolormesh(xx0, xx1, pre_arr, cmap=plt.cm.Paired)

    # Plot also the training points
    plt.scatter(d0_list, d1_list, c=label_list, edgecolors='k', cmap=plt.cm.Paired)
    plt.xlabel('x0')
    plt.ylabel('x1')

    plt.xlim(xx0.min(), xx0.max())
    plt.ylim(xx1.min(), xx1.max())
    plt.xticks(())
    plt.yticks(())

    plt.title('My-AdaBoost(Logistic) with {} estimator'.format(boost_num))
    plt.show()

if __name__ == '__main__':
    # data_i_list, data_j_list = read_testSet(filepath='../../test')
    # SK_AB_on_testSet(data_i_list, data_j_list)

    labeled_data_i_list, labeled_data_j_list = read_testSet_with_label(filepath='../../test')
    my_AB_on_testSet(labeled_data_i_list, labeled_data_j_list)
# -------------------------- Compare my Adaboost with SKlearn-Adaboost (meat larner logistic) on testSet (Machine learning in Action chapter 5) --------------------------