import os
from utils.file_utils.filename_utils import get_date_prefix

def lrc_para_generator():
    """
    为LRC的网格搜索设置参数
    :return:
        返回对应的网格搜索的参数组合列表
        para_list = [max_iter, alpha_init, i]
    """
    para_list = []
    # max_iter = 1000 ~ 15000, step size 2000
    for i in range(8):
        max_iter = 1000 + 2000 * i
        for j in range(-5, 4):
            alpha_init = 10 ** j
            for k in range(10):
                para_list.append([max_iter, alpha_init, k])
    return para_list
# lrc_para_generator()

def lrc_res_checker(folder):
    para_list = lrc_para_generator()
    existed_para_list = []

    filenames = os.listdir(folder)
    # 至少要存在训练的结果
    if len(filenames) >= 1:
        for fn in filenames:
            file_path = os.path.join(folder, fn)
            with open(file_path, 'r') as file:
                for line in file.readlines():
                    # 2020_05_04_lrc_ovr_linear_iter=13000_alpha_init=0.0001_classifer_dict_pickle_3.file,13000,0.0001,3,0.14893617021276595,0.007915567282321892
                    para_str_list = line.strip().split(',')
                    max_iter = int(para_str_list[1])
                    alpha_init = float(para_str_list[2])
                    i = int(para_str_list[3])
                    existed_para_list.append([max_iter, alpha_init, i])

    for e_p in existed_para_list:
        if e_p in para_list:
            para_list.remove(e_p)
    return para_list
# lrc_res_checker(folder='../../ml_sl/logistic/ovr_res/linear/ovo_txt_res')

def svm_para_generator():
    def linear_svm_para_generator():
        """
        Linear
            Adjustable parameters:
                C: 1e-5 ~ 1e5, step factor 10
                tol, default 0.01
                max_iter: 1000 ~ 9000, step size 2000
        """
        linear_svm_para_list = []
        kernel_para_dict = {'type':'linear', 'paras':None}
        for c_i in range(-5, 6):
            c = float(10 ** c_i)
            for j in range(5):
                iter = int(1000 + 2000 * j)
                svm_para_dict = {'C': c, 'max_iter': iter}
                for i in range(10):
                    linear_svm_para_list.append([i, svm_para_dict, kernel_para_dict])
        return linear_svm_para_list

    def poly_svm_para_generator():
        """
        Poly
            Adjustable parameters:
                C: 1e-5 ~ 1e5, step factor 10
                tol, default 0.01
                max_iter: 1000 ~ 9000, step size 2000
                -----------
                power: 2 ~ 10, step size 1
                constant: default 1
                qua_coe: 1e-5 ~ 1e5, step factor 10
        """
        poly_svm_para_list = []
        # C
        for c_i in range(-3, 3):
            c = 10 ** c_i
            # Iteration
            for j in range(5):
                iter = 1000 + 2000 * j
                # Power
                # for p in range(2, 11, 2): # len 5
                # 只保留Power = 2
                p = 2
                # qua_coe
                for q_i in range(-3, 3):
                    q = 10 ** q_i
                    # Repeat for 10 times
                    for i in range(10):
                        svm_para_dict = {'C': c, 'max_iter': iter}
                        kernel_para_dict = {'type': 'poly', 'paras': [p, 1, q]}
                        poly_svm_para_list.append([i, svm_para_dict, kernel_para_dict])
        return poly_svm_para_list

    def rbf_svm_para_generator():
        """
        Rbf
            Adjustable parameters:
                C: 1e-5 ~ 1e5, step factor 10
                tol: default 0.01
                max_iter: 1000 ~ 9000, step size 2000
                -----------
                sigma: 1e-5 ~ 1e5, step factor 10
        """
        rbf_svm_para_list = []
        # C
        for c_i in range(-5, 6): # 11
            c = float(10 ** c_i)
            # Iteration
            for j in range(5): # 5
                iter = int(1000 + 2000 * j)
                for s_i in range(-5, 6): # 11
                    sigma = float(10 ** s_i)
                    for i in range(10): # 10
                        svm_para_dict = {'C': c, 'max_iter': iter}
                        kernel_para_dict = {'type': 'rbf', 'paras': sigma}
                        rbf_svm_para_list.append([i, svm_para_dict, kernel_para_dict])
        return rbf_svm_para_list

    linear_svm_para_list = linear_svm_para_generator() # len = 550
    poly_svm_para_list = poly_svm_para_generator() # len = 9000
    rbf_svm_para_list = rbf_svm_para_generator() # len = 6050
    return linear_svm_para_list, poly_svm_para_list, rbf_svm_para_list
# svm_para_generator()

def svm_res_checker(folder):
    linear_svm_para_list, poly_svm_para_list, rbf_svm_para_list = svm_para_generator()

    filenames = os.listdir(folder)

    existed_linear_para_list = []
    existed_poly_para_list = []
    existed_rbf_para_list = []

    for fn in filenames:
        file_path = os.path.join(folder, fn)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                pickle_model_name = line.strip().split(',')[0]
                # 1-Ensure the kernel type
                # Linear, 2020_05_01_svm_linear_C=1e-05_iter=1000_pickle_0.file
                if 'linear' in pickle_model_name:
                    # wanted paras, C, iter, repeated time
                    para_str_list = pickle_model_name.split('linear')[1].split('_')
                    C = float(para_str_list[1].split('=')[1])
                    max_iter = int(para_str_list[2].split('=')[1])
                    repeated_time = int(para_str_list[-1].split('.')[0])
                    svm_para_dict = {'C': C, 'max_iter': max_iter}
                    kernel_para_dict = {'type' : 'linear', 'paras' : None}
                    existed_linear_para_list.append([repeated_time, svm_para_dict, kernel_para_dict])

                # Poly, 2020_05_01_svm_poly_C=0.01_iter=1000_P=2_q=1_pickle_0.file
                elif 'poly' in pickle_model_name:
                    # wanted paras: C, max_iter, Power of polynomial, coefficient of X.T * X, repeated time
                    para_str_list = pickle_model_name.split('poly')[1].split('_')
                    C = float(para_str_list[1].split('=')[1])
                    max_iter = int(para_str_list[2].split('=')[1])
                    power = int(para_str_list[3].split('=')[1])
                    q = float(para_str_list[4].split('=')[1])
                    repeated_time = int(para_str_list[-1].split('.')[0])
                    svm_para_dict = {'C': C, 'max_iter': max_iter}
                    kernel_para_dict = {'type': 'poly', 'paras': [power, 1, q]}
                    existed_poly_para_list.append([repeated_time, svm_para_dict, kernel_para_dict])

                # Rbf, 2020_05_01_svm_rbf_C=0.0001_iter=5000_sigma=1000_pickle_0.file
                elif 'rbf' in pickle_model_name:
                    # wanted paras: C, max_iter, sigma, repeated time
                    para_str_list = pickle_model_name.split('rbf')[1].split('_')
                    C = float(para_str_list[1].split('=')[1])
                    max_iter = int(para_str_list[2].split('=')[1])
                    sigma = float(para_str_list[3].split('=')[1])
                    repeated_time = int(para_str_list[-1].split('.')[0])
                    svm_para_dict = {'C': C, 'max_iter': max_iter}
                    kernel_para_dict = {'type': 'rbf', 'paras': sigma}
                    existed_rbf_para_list.append([repeated_time, svm_para_dict, kernel_para_dict])

    # Remove existed linear para
    for e_l_p in existed_linear_para_list:
        if e_l_p in linear_svm_para_list:
            linear_svm_para_list.remove(e_l_p)
        # else:
        #     print('e_l_p = ', e_l_p)

    # Remove existed poly para
    for e_p_p in existed_poly_para_list:
        if e_p_p in poly_svm_para_list:
            poly_svm_para_list.remove(e_p_p)
        # else:
        #     print('e_p_p = ', e_p_p)

    # Remove existed rbf para
    for e_r_p in existed_rbf_para_list:
        if e_r_p in rbf_svm_para_list:
            rbf_svm_para_list.remove(e_r_p)
        # else:
        #     print('e_r_p = ', e_r_p)

    print('Linear left = {0}, Poly left = {1}, RBF = {2}'.format(len(linear_svm_para_list), len(poly_svm_para_list), len(rbf_svm_para_list) ))
    return linear_svm_para_list, poly_svm_para_list, rbf_svm_para_list
# svm_res_checker(folder='../../ml_sl/svm/ovo_txt_res')

def svm_res_sorter(folder, divide_type):
    """
    :param
        folder:
            folder stores the training results
        divide_type:
            multiclassification divide strategy
            str, 'OvO' or 'OvR'
    :return:
    """
    # divide_type:
    linear_para_list = []
    poly_para_list = []
    rbf_para_list = []

    filenames = os.listdir(folder)
    if len(filenames) >= 1:
        for fn in filenames:
            file_path = os.path.join(folder, fn)
            with open(file_path, 'r') as file:
                for line in file.readlines():
                    # 1-Judge kernel type
                    line = line.strip()
                    line_str_list = line.split(',')
                    model_name = line_str_list[0]
                    kernel_type = model_name.split('_')[4]
                    if kernel_type == 'linear':
                        # wanted para: C, Max_iter, i
                        # Exp: 2020_05_02_svm_linear_C=0.001_iter=9000_pickle_7.file,7,0.44680851063829785,0.24799999999999994
                        # model_name, Exp: 2020_05_02_svm_linear_C=0.001_iter=9000_pickle_7.file
                        C = float(model_name.split('_')[5].split('=')[1])
                        max_iter = int(model_name.split('_')[6].split('=')[1])
                        i = int(model_name.split('_')[-1].split('.')[0])

                        svm_para_dict = {'C': C, 'max_iter': max_iter}
                        kernel_para_dict = {'type': 'linear', 'paras': None}
                        linear_para_list.append([i, svm_para_dict, kernel_para_dict, line])

                    elif kernel_type == 'poly':
                        # wanted para: C, Max_iter, i, power, q
                        # Exp: 2020_05_01_svm_poly_C=0.01_iter=1000_P=4_q=0.001_pickle_1.file,1,0.26595744680851063,0.14770039421813405
                        # model_name, Exp: 2020_05_01_svm_poly_C=0.01_iter=1000_P=4_q=0.001_pickle_1.file
                        C = float(model_name.split('_')[5].split('=')[1])
                        max_iter = int(model_name.split('_')[6].split('=')[1])
                        power = int(model_name.split('_')[7].split('=')[1])
                        q = float(model_name.split('_')[8].split('=')[1])
                        i = int(model_name.split('_')[-1].split('.')[0])

                        svm_para_dict = {'C': C, 'max_iter': max_iter}
                        kernel_para_dict = {'type': 'poly', 'paras': [power, 1, q]}
                        poly_para_list.append([i, svm_para_dict, kernel_para_dict, line])

                    elif kernel_type == 'rbf':
                        # wanted para: C, Max_iter, i, sigma
                        # Exp: 2020_05_02_svm_rbf_C=1_iter=7000_sigma=0.1_pickle_0.file,0,0.48936170212765956,0.37298499166203436
                        # model_name, Exp: 2020_05_02_svm_rbf_C=1_iter=7000_sigma=0.1_pickle_0.file
                        C = float(model_name.split('_')[5].split('=')[1])
                        max_iter = int(model_name.split('_')[6].split('=')[1])
                        sigma = float(model_name.split('_')[7].split('=')[1])
                        i = int(model_name.split('_')[-1].split('.')[0])

                        svm_para_dict = {'C': C, 'max_iter': max_iter}
                        kernel_para_dict = {'type': 'rbf', 'paras': sigma}
                        rbf_para_list.append([i, svm_para_dict, kernel_para_dict, line])

    linear_svm_para_list, poly_svm_para_list, rbf_svm_para_list = svm_para_generator()

    # Sort linear para and write sorted results in a file
    svm_linear_sorted_res_file = get_date_prefix() + divide_type + '_svm_linear_sorted_res.txt'
    for linear_para in linear_svm_para_list:
        for i in range(len(linear_para_list)):
            if linear_para == linear_para_list[i][:-1]:
                svm_para_dict = linear_para[1]
                C = svm_para_dict['C']
                max_iter = svm_para_dict['max_iter']
                line = linear_para_list[i][-1]
                line_str_list = line.strip().split(',')
                new_line = ','.join([line_str_list[0], str(max_iter), str(C)] + line_str_list[1:]) + '\n'
                with open(svm_linear_sorted_res_file, 'a+') as file:
                    file.write(new_line)
                break

    # Sort poly para and write sorted results in a file
    svm_poly_sorted_res_file = get_date_prefix() + divide_type + '_svm_poly_sorted_res.txt'
    for poly_para in poly_svm_para_list:
        for i in range(len(poly_para_list)):
            if poly_para == poly_para_list[i][:-1]:
                svm_para_dict = poly_para[1]
                C = svm_para_dict['C']
                max_iter = svm_para_dict['max_iter']

                kernel_para_dict = poly_para[2]
                power = kernel_para_dict['paras'][0]
                qua_coe = kernel_para_dict['paras'][2]

                line = poly_para_list[i][-1]
                line_str_list = line.strip().split(',')

                new_line = ','.join([line_str_list[0], str(max_iter), str(C), str(power), str(qua_coe)] + line_str_list[1:]) + '\n'
                with open(svm_poly_sorted_res_file, 'a+') as file:
                    file.write(new_line)
                break

    # Sort rbf para and write sorted results in a file
    svm_rbf_sorted_res_file = get_date_prefix() + divide_type + '_svm_rbf_sorted_res.txt'
    for rbf_para in rbf_svm_para_list:
        for i in range(len(rbf_para_list)):
            if rbf_para == rbf_para_list[i][:-1]:
                svm_para_dict = rbf_para[1]
                C = svm_para_dict['C']
                max_iter = svm_para_dict['max_iter']

                kernel_para_dict = rbf_para[2]
                sigma = kernel_para_dict['paras']

                line = rbf_para_list[i][-1]
                line_str_list = line.strip().split(',')

                new_line = ','.join([line_str_list[0], str(max_iter), str(C), str(sigma)] + line_str_list[1:]) + '\n'
                with open(svm_rbf_sorted_res_file, 'a+') as file:
                    file.write(new_line)
                break
# ------------------------ Sort OvO Results ------------------------
# svm_res_sorter(folder='../../ml_sl/svm/ovo_txt_res/trained_on_tr_tested_on_vali', divide_type='OvO')
# ------------------------ Sort OvO Results ------------------------

# ------------------------ Sort OvR Results ------------------------
# svm_res_sorter(folder='../../ml_sl/svm/ovr_txt_res/trained_on_tr_tested_on_vali', divide_type='OvR')
# ------------------------ Sort OvR Results ------------------------

def adaBoost_para_generator():
    """
    按照boost_num 从50 ~ 450，每种重复10次，一共90组参数
    :return:
    """
    para_list = []
    for i in range(1, 10):
        for j in range(10):
            para_list.append([j, 50 * i])
    return para_list

def adaBoost_final_para_generator():
    """
    按照boost_num 从150 ~ 350，每种重复10次，一共50组参数
    :return:
    """
    all_para_list = []
    for i in range(10):
        for b in range(5):
            boost_num = 150 + 50 * b
            all_para_list.append([i, boost_num])
    return all_para_list

def adaBoost_res_checker(folder):
    filenames = os.listdir(folder)

    existed_para_list = []
    for fn in filenames:
        file_path = os.path.join(folder, fn)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                line_list = line.strip().split(',')
                boost_num = int(line_list[0].split('=')[1].split('_')[0])
                iter = int(line_list[1])
                existed_para_list.append([iter, boost_num])
    para_list = adaBoost_para_generator()
    # left_para_list = list(set(para_list) - set(existed_para_list))
    for e_para in existed_para_list:
        if e_para in para_list:
            para_list.remove(e_para)
    return para_list

def adaBoost_final_res_checker(folder):
    filenames = os.listdir(folder)

    existed_para_list = []
    for fn in filenames:
        file_path = os.path.join(folder, fn)
        with open(file_path, 'r') as file:
            for line in file.readlines():
                # 2020_06_27_ab_final_boost_num=250_1_pickle.file,250,1,0.4065934065934066,0.27275418084948944,0.679347587442896
                line_list = line.strip().split(',')
                boost_num = int(line_list[0].split('=')[1].split('_')[0])
                i = int(line_list[2])
                existed_para_list.append([i, boost_num])
    all_para_list = adaBoost_final_para_generator()
    # left_para_list = list(set(para_list) - set(existed_para_list))
    # print('Existed para:', existed_para_list)
    for e_para in existed_para_list:
        if e_para in all_para_list:
            all_para_list.remove(e_para)
    # print('Left para:', all_para_list)
    return all_para_list

def adaBoost_res_sorter(folder):
    trained_para_list = []

    filenames = os.listdir(folder)
    if len(filenames) >= 1:
        for fn in filenames:
            file_path = os.path.join(folder, fn)
            with open(file_path, 'r') as file:
                for line in file.readlines():
                    line_str_list = line.strip().split(',')
                    model_file_name = line_str_list[0]
                    boost_num_str = model_file_name.split('.')[0].split('_')[5].split('=')[1]
                    boost_num = int(boost_num_str)
                    i = int(line_str_list[1])
                    trained_para_list.append([i, boost_num, line])

    planed_para_list = adaBoost_para_generator()
    adaBoost_sorted_res_fn = get_date_prefix() + 'adaBoost_sorted_res.txt'
    for p_para in planed_para_list:
        for i in range(len(trained_para_list)):
            if p_para == trained_para_list[i][:-1]:
                with open(adaBoost_sorted_res_fn, 'a+') as file:
                    boost_num = p_para[1]
                    line = trained_para_list[i][-1]
                    line_str_list = line.strip().split(',')
                    new_line = ','.join([line_str_list[0], str(boost_num)] + line_str_list[1:]) + '\n'
                    file.write(new_line)
                break
# --------------------------- Sort AdaBoost result ---------------------------
# adaBoost_res_sorter(folder='../../ml_sl/adaboost/txt_res/trained_on_tr_tested_on_vali')
# --------------------------- Sort AdaBoost result ---------------------------

def sort_by_triple_column(txt_path, col_index_list, output_filename='sorted.txt'):
    """
    function
        主要用于多进程戌训练LRC的时候，结果文件混乱排放，整理文件顺序的
    :param
        col_index_list:
    :return:
    """
    data_list = []
    filenames = os.listdir(txt_path)
    if len(filenames) >= 1:
        for fn in filenames:
            file_path = os.path.join(txt_path, fn)
            with open(file_path, 'r') as file:
                for line in file.readlines():
                    line1 = line.strip()
                    line_str_list = line1.split(',')
                    data = [line_str_list[0]] + [float(d) for d in line_str_list[1:]] + [line]
                    data_list.append(data)
    else:
        print('In the folder: {0}, there is no training results'.format(txt_path))

    c0_ele_list = sorted(list(set([d[col_index_list[0]] for d in data_list])))
    c1_ele_list = sorted(list(set([d[col_index_list[1]] for d in data_list])))
    c2_ele_list = sorted(list(set([d[col_index_list[2]] for d in data_list])))

    sorted_filename = get_date_prefix() + output_filename
    for c0 in c0_ele_list:
        for c1 in c1_ele_list:
            for c2 in c2_ele_list:
                for d in data_list:
                    if (d[col_index_list[0]] == c0) and (d[col_index_list[1]] == c1) and (d[col_index_list[2]] == c2):
                        with open(sorted_filename, 'a+') as file:
                            file.write(d[-1])
                        break
# sort_by_triple_column(txt_path='2020_04_22_linear_final.txt', col_index_list=[1,2,3])
# ---------------------------------------------------------------------
# output_filename = 'lrc_ovr_linear_sorted.txt'
# sort_by_triple_column(txt_path='../../ml_sl/logistic/ovr_res/linear/ovo_txt_res', col_index_list=[1,2,3], output_filename=output_filename)
# ---------------------------------------------------------------------
# sort_by_triple_column(txt_path='../../ml_sl/logistic/ovr_res/linear/ovo_txt_res', col_index_list=[1,2,3], output_filename='lrc_ovr_linear_sorted.txt')

# def sort_by_four_column(txt_path, col_index_list, output_filename='sorted.txt'):
#     """
#     function
#         主要用于多进程戌训练LRC的时候，结果文件混乱排放，整理文件顺序的
#     :param
#         col_index_list:
#     :return:
#     """
#     data_list = []
#     filenames = os.listdir(txt_path)
#     if len(filenames) >= 1:
#         for fn in filenames:
#             file_path = os.path.join(txt_path, fn)
#             with open(file_path, 'r') as file:
#                 for line in file.readlines():
#                     line1 = line.strip()
#                     line_str_list = line1.split(',')
#                     data = [line_str_list[0]] + [float(d) for d in line_str_list[1:]] + [line]
#                     data_list.append(data)
#     else:
#         print('In the folder: {0}, there is no training results'.format(txt_path))
#
#     c0_ele_list = sorted(list(set([d[col_index_list[0]] for d in data_list])))
#     c1_ele_list = sorted(list(set([d[col_index_list[1]] for d in data_list])))
#     c2_ele_list = sorted(list(set([d[col_index_list[2]] for d in data_list])))
#
#     sorted_filename = get_date_prefix() + output_filename
#     for c0 in c0_ele_list:
#         for c1 in c1_ele_list:
#             for c2 in c2_ele_list:
#                 for d in data_list:
#                     if (d[col_index_list[0]] == c0) and (d[col_index_list[1]] == c1) and (d[col_index_list[2]] == c2):
#                         with open(sorted_filename, 'a+') as file:
#                             file.write(d[-1])
#                         break

# def sort_by_five_column(txt_path, col_index_list, output_filename='sorted.txt'):
#     """
#     function
#         主要用于多进程戌训练LRC的时候，结果文件混乱排放，整理文件顺序的
#     :param
#         col_index_list:
#     :return:
#     """
#     data_list = []
#     filenames = os.listdir(txt_path)
#     if len(filenames) >= 1:
#         for fn in filenames:
#             file_path = os.path.join(txt_path, fn)
#             with open(file_path, 'r') as file:
#                 for line in file.readlines():
#                     line1 = line.strip()
#                     line_str_list = line1.split(',')
#                     data = [line_str_list[0]] + [float(d) for d in line_str_list[1:]] + [line]
#                     data_list.append(data)
#     else:
#         print('In the folder: {0}, there is no training results'.format(txt_path))
#
#     c0_ele_list = sorted(list(set([d[col_index_list[0]] for d in data_list])))
#     c1_ele_list = sorted(list(set([d[col_index_list[1]] for d in data_list])))
#     c2_ele_list = sorted(list(set([d[col_index_list[2]] for d in data_list])))
#
#     sorted_filename = get_date_prefix() + output_filename
#     for c0 in c0_ele_list:
#         for c1 in c1_ele_list:
#             for c2 in c2_ele_list:
#                 for d in data_list:
#                     if (d[col_index_list[0]] == c0) and (d[col_index_list[1]] == c1) and (d[col_index_list[2]] == c2):
#                         with open(sorted_filename, 'a+') as file:
#                             file.write(d[-1])
#                         break