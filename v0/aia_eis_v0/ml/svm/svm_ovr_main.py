import sys
sys.path.append('../../')

from ml_sl.ml_critrions import cal_accuracy, cal_kappa
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
            power: 2 ~ 10, step size 1
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
def svm_ovr_linear_tr_va(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                               unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset,\
                               label_list=[2,4,5,6,7,8,9])
    svm_model_pickle_name = get_date_prefix()+'svm_ovr_linear_test_pickle.file'
    multi_svm.create_svm_ovr_classifer(svm_model_pickle_name)
    sample_label_prob_dict_list = multi_svm.classify_ovr(svm_model_pickle_name)

    acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

# if __name__ == '__main__':
#     svm_para_dict = {'C': 10, 'max_iter':1000}
#     kernel_para_dict = {'type':'linear', 'paras':None}
#     svm_ovr_linear_tr_va(svm_para_dict, kernel_para_dict)
# R(RC)_IS_lin-kk_res.txt: Accuracy=0.13829787234042554, Kappa=0.01957249549317538
# ---------------- casual test of Multiclass-SVM (linear kernel)----------------

# ---------------- casual test of Multiclass-SVM (Poly kernel)----------------
def svm_ovr_poly_tr_va(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                               unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset,\
                               label_list=[2,4,5,6,7,8,9])
    svm_model_pickle_name = get_date_prefix()+'svm_ovr_poly_test_pickle.file'
    multi_svm.create_svm_ovr_classifer(svm_model_pickle_name)
    sample_label_prob_dict_list = multi_svm.classify_ovr(svm_model_pickle_name)

    acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

# if __name__ == '__main__':
#     svm_para_dict = {'C': 10, 'max_iter':1000}
#     kernel_para_dict = {'type':'poly', 'paras':[2,1,1]}
#     svm_ovr_poly_tr_va(svm_para_dict, kernel_para_dict)
# R(RC)_IS_lin-kk_res.txt: Accuracy=0.1276595744680851, Kappa=-0.020251489080079454
# ---------------- casual test of Multiclass-SVM (Poly kernel)----------------

# ---------------- casual test of Multiclass-SVM (Rbf kernel)----------------
def svm_ovr_rbf_tr_va(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset

    vali_data_list = []
    vali_label_list = []
    for vali in validation_dataset:
        vali_label_list.append(vali[0])
        vali_data_list.append(vali[1])
    multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict,\
                               unlabeled_dataset_list=vali_data_list, labeled_dataset_list=training_dataset,\
                               label_list=[2,4,5,6,7,8,9])
    svm_model_pickle_name = get_date_prefix() + 'svm_ovr_rbf_test_pickle.file'
    multi_svm.create_svm_ovr_classifer(svm_model_pickle_name)
    sample_label_prob_dict_list = multi_svm.classify_ovr(svm_model_pickle_name)

    acc = cal_accuracy(sample_label_prob_dict_list, vali_label_list)
    kappa = cal_kappa(sample_label_prob_dict_list, vali_label_list)
    print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

# if __name__ == '__main__':
#     svm_para_dict = {'C': 10, 'max_iter':1000}
#     kernel_para_dict = {'type':'rbf', 'paras': 10}
#     svm_ovr_rbf_tr_va(svm_para_dict, kernel_para_dict)
# R(RC)_IS_lin-kk_res.txt: Accuracy=0.22340425531914893, Kappa=0.1066267413097253
# ---------------- casual test of Multiclass-SVM (Rbf kernel)----------------

# ---------------- Train SVM on TV and Test on Test-dataset ----------------
def svm_ovr_TV_te(svm_para_dict, kernel_para_dict):
    global training_dataset, validation_dataset, test_dataset
    TV_dataset = training_dataset + validation_dataset

    te_data_list = []
    te_label_list = []
    for te in test_dataset:
        te_label_list.append(te[0])
        te_data_list.append(te[1])

    # Repeat for 10 times
    for i in range(10):
        """
        2020_05_08_svm_linear_C=1e-05_iter=1000_pickle_0.file
        2020_05_08_svm_poly_C=0.01_iter=1000_P=2_q=1_pickle_0.file
        2020_05_08_svm_rbf_C=0.0001_iter=1000_sigma=0.0001_pickle_9.file
        """
        multi_svm = Multiclass_SVM(svm_para_dict=svm_para_dict, kernel_para_dict=kernel_para_dict, \
                                   unlabeled_dataset_list=te_data_list, labeled_dataset_list=TV_dataset, \
                                   label_list=[2, 4, 5, 6, 7, 8, 9])
        kernel_type = kernel_para_dict['type']
        svm_model_pickle_name = get_date_prefix()
        if kernel_type == 'linear':
            C_str = str(svm_para_dict['C'])
            iter_str = str(svm_para_dict['max_iter'])
            svm_model_pickle_name += 'svm_ovr_final_{0}_C={1}_iter={2}_pickle_{3}.file'.format(kernel_type, C_str, iter_str, str(i))
        elif kernel_type == 'poly':
            C_str = str(svm_para_dict['C'])
            iter_str = str(svm_para_dict['max_iter'])
            para_list = kernel_para_dict['paras']
            q_str = str(para_list[1])
            svm_model_pickle_name += 'svm_ovr_final_{0}_C={1}_iter={2}_q={3}_pickle_{4}.file'.format(kernel_type, C_str, iter_str, q_str, str(i))
        elif kernel_type == 'rbf':
            C_str = str(svm_para_dict['C'])
            iter_str = str(svm_para_dict['max_iter'])
            sigma_str = str(kernel_para_dict['paras'])
            svm_model_pickle_name += 'svm_ovr_final_{0}_C={1}_iter={2}_sigma={3}_pickle_{4}.file'.format(kernel_type, C_str, sigma_str, iter_str, str(i))

        multi_svm.create_svm_ovr_classifer(svm_model_pickle_name)
        sample_label_prob_dict_list = multi_svm.classify_ovr(svm_model_pickle_name)

        acc = cal_accuracy(sample_label_prob_dict_list, te_label_list)
        kappa = cal_kappa(sample_label_prob_dict_list, te_label_list)
        print('Accuracy={0}, Kappa={1}'.format(acc, kappa))

if __name__ == '__wmain__':
    # ---------------------- SVM_OvR-Linear ----------------------
    # ------------- iter = 5000, C = 0.1 -------------
    # svm_para_dict = {'C': 0.1, 'max_iter': 5000}
    # kernel_para_dict = {'type':'linear', 'paras':None}
    # svm_ovr_TV_te(svm_para_dict, kernel_para_dict)
    # ------------- iter = 5000, C = 0.1 -------------
    # ---------------------- SVM_OvR-Linear ----------------------

    # ---------------------- SVM_OvR-Poly ----------------------
    # ------------- iter = 3000, C = 1, q = 100 -------------
    # In poly, the power is default as 2, () ** q(=2); and the constant is default as 1
    # svm_para_dict = {'C': 1, 'max_iter' : 3000}
    # kernel_para_dict = {'type' : 'poly', 'paras' : [2, 1, 100]}
    # svm_ovr_TV_te(svm_para_dict, kernel_para_dict)
    # ------------- iter = 3000, C = 1, q = 100 -------------
    # ---------------------- SVM_OvR-Poly ----------------------

    # ---------------------- SVM_OvR-RBF ----------------------
    # ------------- iter = 7000, C = 0.01, sigma = 100000 -------------
    # svm_para_dict = {'C' : 0.01, 'max_iter' : 7000}
    # kernel_para_dict = {'type' : 'rbf', 'paras' : 100000}
    # svm_ovr_TV_te(svm_para_dict, kernel_para_dict)
    # ------------- iter = 7000, C = 0.01, sigma = 100000 -------------

    # ------------- iter = 5000, C = 0.001, sigma = 10 -------------
    # svm_para_dict = {'C': 0.001, 'max_iter': 5000}
    # kernel_para_dict = {'type': 'rbf', 'paras': 10}
    # svm_ovr_TV_te(svm_para_dict, kernel_para_dict)
    # ------------- iter = 5000, C = 0.001, sigma = 10 -------------

    # ------------- iter = 5000, C = 0.01, sigma = 0.001 -------------
    svm_para_dict = {'C': 0.01, 'max_iter': 5000}
    kernel_para_dict = {'type': 'rbf', 'paras': 0.001}
    svm_ovr_TV_te(svm_para_dict, kernel_para_dict)
    # ------------- iter = 5000, C = 0.01, sigma = 0.001 -------------
    # ---------------------- SVM_OvR-RBF ----------------------
# python svm_ovr_main.py    
# ---------------- Train SVM on TV and Test on Test-dataset ----------------