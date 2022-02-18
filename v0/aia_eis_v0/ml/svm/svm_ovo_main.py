import os

from ml_sl.ml_critrions import cal_accuracy, cal_kappa, cal_accuracy_on_2, cal_accuracy_on_3
from utils.file_utils.filename_utils import get_date_prefix
from utils.file_utils.dataset_reader_pack.ml_dataset_reader import get_T_V_T_dataset
from ml_sl.svm.multiclass_svm_0 import Multiclass_SVM

"""
SVM
    Linear
        Adjustable parameters:
            C: 1e-5 ~ 1e5, step factor 10
            tol, default 0.01
            max_iter: 1000 ~ 9000, step size 2000
    Poly
        Adjustable parameters:
            C: 1e-5 ~ 1e5, step factor 10
            tol, default 0.01
            max_iter: 1000 ~ 9000, step size 2000
            power: default as 2. [Original design: 2 ~ 10, step size 1]
            constant: default 1
            qua_coe: 1e-5 ~ 1e5, step factor 10
    Rbf
        Adjustable parameters:
            C: 1e-5 ~ 1e5, step factor 10
            tol: default 0.01
            max_iter: 1000 ~ 9000, step size 2000
            sigma: 1e-5 ~ 1e5, step factor 10
"""
training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path='../../datasets/ml_datasets/normed')
# ---------------- casual test of Multiclass-SVM (linear kernel)----------------
def svm_ovo_linear_tr_va(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                               unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset,\
                               label_list=[2,4,5,6,7,8,9])
    svm_model_pickle_name = get_date_prefix()+'svm_linear_test_pickle.file'
    multi_svm.create_svm_ovo_classifer(svm_model_pickle_name)
    sample_label_prob_dict_list = multi_svm.classify_ovo(svm_model_pickle_name)

    acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

# if __name__ == '__main__':
#     svm_para_dict = {'C': 10, 'max_iter':1000}
#     kernel_para_dict = {'type':'linear', 'paras':None}
#     svm_linear_tr_va(svm_para_dict, kernel_para_dict)
# ---------------- casual test of Multiclass-SVM (linear kernel)----------------

# ---------------- casual test of Multiclass-SVM (Poly kernel)----------------
def svm_ovo_poly_tr_va(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                               unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset,\
                               label_list=[2,4,5,6,7,8,9])
    svm_model_pickle_name = get_date_prefix()+'svm_linear_test_pickle.file'
    multi_svm.create_svm_ovo_classifer(svm_model_pickle_name)
    sample_label_prob_dict_list = multi_svm.classify_ovo(svm_model_pickle_name)

    acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

# if __name__ == '__main__':
#     svm_para_dict = {'C': 10, 'max_iter':1000}
#     kernel_para_dict = {'type':'poly', 'paras':[2,1,1]}
#     svm_poly_tr_va(svm_para_dict, kernel_para_dict)
# R(RC)_IS_lin-kk_res.txt: Accuracy=0.26595744680851063, Kappa=0.13230769230769235
# ---------------- casual test of Multiclass-SVM (Poly kernel)----------------

# ---------------- casual test of Multiclass-SVM (Rbf kernel)----------------
def svm_ovo_rbf_tr_va(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                               unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset,\
                               label_list=[2,4,5,6,7,8,9])
    svm_model_pickle_name = get_date_prefix() + 'svm_linear_test_pickle.file'
    multi_svm.create_svm_ovo_classifer(svm_model_pickle_name)
    sample_label_prob_dict_list = multi_svm.classify_ovo(svm_model_pickle_name)

    acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

# if __name__ == '__main__':
#     svm_para_dict = {'C': 10, 'max_iter':1000}
#     kernel_para_dict = {'type':'rbf', 'paras': 10}
#     svm_rbf_tr_va(svm_para_dict, kernel_para_dict)
# ---------------- casual test of Multiclass-SVM (Rbf kernel)----------------

# ---------------- Train model on Train+Vali datasets and evaluate it on test dataset ----------------
def svm_ovo_linear_TV_te():
    """
    Function:
        After a through grid search of hyperparameters of SVM, the final results, the best five hyperparameters setting are:
            Iteration: 1000, C: 0.001
            Iteration: 3000, C: 0.001
            Iteration: 5000, C: 0.001
            Iteration: 7000, C: 0.001
            Iteration: 9000, C: 0.001
    :param
        svm_para_dict:
        kernel_para_dict:
    :return:
    """
    global training_dataset, validation_dataset, test_dataset
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    svm_ovo_linear_final_res_fn = get_date_prefix() + 'svm_ovo_linear_final_res.txt'
    counter = 0
    for i in range(5):
        max_iter = 1000 + 2000 * i
        svm_para_dict = {'C': 0.001, 'max_iter': max_iter}
        kernel_para_dict = {'type' : 'linear', 'paras' : None}
        for j in range(10):
            multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                                       unlabeled_dataset_list=test_data_list,
                                       labeled_dataset_list=training_dataset+validation_dataset,\
                                       label_list=[2,4,5,6,7,8,9])
            svm_model_pickle_name = get_date_prefix()+'svm_ovo_linear_final_C=0.001_iter='+str(max_iter)+'_'+str(j)+'_pickle.file'
            multi_svm.create_svm_ovo_classifer(svm_model_pickle_name)
            sample_label_prob_dict_list = multi_svm.classify_ovo(svm_model_pickle_name)

            acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
            kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

            with open(svm_ovo_linear_final_res_fn, 'a+') as file:
                line = ','.join([svm_model_pickle_name, str(svm_para_dict['C']), str(max_iter), str(acc), str(kappa), str(acc+kappa)]) + '\n'
                file.write(line)
            counter += 1
            print('Finished {0}, {1} left'.format(counter, 50 - counter))
# svm_ovo_linear_TV_te()
# ---------------- Train model on Train+Vali datasets and evaluate it on test dataset ----------------

"""
OvR-Linear/Poly/RBF的效果都很差，AK最高都没有超过0.4，可悲啊
OvO的整体效果相对好一些，Linear(0.69) > Poly(~0.53) > RBF(~0.5) 
"""
def svm_ovo_acc_on_first_3_predictions():
    # Load datasets
    global training_dataset, validation_dataset, test_dataset
    test_data_list = []
    test_label_list = []
    for te in test_dataset:
        test_label_list.append(te[0])
        test_data_list.append(te[1])

    # load trained SVM
    svm_ovo_linear_final_model_path = os.path.join('ovo_models/final',\
                                                   '2020_06_26_svm_ovo_linear_final_C=0.001_iter=5000_3_pickle.file')
    max_iter = 5000
    svm_para_dict = {'C': 0.001, 'max_iter': max_iter}
    kernel_para_dict = {'type': 'linear', 'paras': None}
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict, \
                               unlabeled_dataset_list=test_data_list,\
                               labeled_dataset_list=training_dataset + validation_dataset, \
                               label_list=[2, 4, 5, 6, 7, 8, 9])
    # SVM classify test-dataset
    sample_label_prob_dict_list = multi_svm.classify_ovo(svm_ovo_model_pickle_name=svm_ovo_linear_final_model_path)

    # get accuracy on the first 1/2/3 predictions
    acc = cal_accuracy(sample_label_prob_dict_list, test_label_list)
    acc_on_2 = cal_accuracy_on_2(sample_label_prob_dict_list, test_label_list)
    acc_on_3 = cal_accuracy_on_3(sample_label_prob_dict_list, test_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, test_label_list)

    print('Accuracy on 1 = {0}, Accuracy on 2 = {1}, Accuracy on 3 = {2}, Kappa={3}'.format(acc, acc_on_2, acc_on_3, kappa))
# svm_ovo_acc_on_first_3_predictions()
# Results:
#   Accuracy on 1 = 0.46153846153846156, Accuracy on 2 = 0.6043956043956044,
#   Accuracy on 3 = 0.7142857142857143, Kappa=0.24755315558555518