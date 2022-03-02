import os
import pickle

from ml_sl.ml_data_wrapper import point_2_x_y

"""
Dataset used for ML is stored at prj/datasets/ml_datasets/normed
    files
        ml_normed_test_data_list_pickle_2020_03_07.file
        ml_normed_train_data_list_pickle_2020_03_07.file
        ml_normed_vali_data_list_pickle_2020_03_07.file

    1-Some MLs use training and validation separately
        1.1 create function get_T_V_T_dataset()

    2-Some MLs use training and validation together
        2.1 create function get_TV_T_dataset()
"""

def get_T_V_T_dataset(file_path):
    """
    list[
            [label (int), [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)]]
        ]
    the range of x or y is 0 ~ 1
    """

    tr_file_date_str = '2020_04_12'
    va_file_date_str = '2020_03_07'
    te_file_date_str = '2020_03_07'
    tr_file_path = os.path.join(file_path, 'ml_normed_train_data_list_pickle_'+tr_file_date_str+'.file')
    va_file_path = os.path.join(file_path, 'ml_normed_vali_data_list_pickle_'+va_file_date_str+'.file')
    te_file_path = os.path.join(file_path, 'ml_normed_test_data_list_pickle_'+te_file_date_str+'.file')

    with open(tr_file_path, 'rb') as file:
        training_dataset = pickle.load(file)
    with open(va_file_path, 'rb') as file:
        validation_dataset = pickle.load(file)
    with open(te_file_path, 'rb') as file:
        test_dataset = pickle.load(file)
    return training_dataset, validation_dataset, test_dataset

def get_TV_T_dataset(file_path):
    training_dataset, validation_dataset, test_dataset = get_T_V_T_dataset(file_path)
    training_dataset.extend(validation_dataset)
    tr_va_dataset = training_dataset
    return tr_va_dataset, test_dataset

def get_raw_T_V_T_dataset(file_path, file_date_str='2020_03_07'):
    raw_training_dataset = None
    raw_validation_dataset = None
    raw_test_dataset = None

    tr_file_path = os.path.join(file_path, 'ml_raw_train_data_dict_pickle_' + file_date_str + '.file')
    va_file_path = os.path.join(file_path, 'ml_raw_vali_data_dict_pickle_' + file_date_str + '.file')
    te_file_path = os.path.join(file_path, 'ml_raw_test_data_dict_pickle_' + file_date_str + '.file')

    with open(tr_file_path, 'rb') as file:
        raw_training_dataset = pickle.load(file)
    with open(va_file_path, 'rb') as file:
        raw_validation_dataset = pickle.load(file)
    with open(te_file_path, 'rb') as file:
        raw_test_dataset = pickle.load(file)
    return raw_training_dataset, raw_validation_dataset, raw_test_dataset

def get_raw_dataset_by_ecm_num(raw_training_dataset, raw_validation_dataset, raw_test_dataset, ecm_num):
    """
    Running steps:
        1-Before use this function, u should run function [get_raw_T_V_T_dataset] to
        get raw_training_dataset, raw_validation_dataset, raw_test_dataset
        2-raw_training_dataset, raw_validation_dataset, raw_test_dataset are dicts,
        and get the desired part from each one of them by inputting key[ecm_num]
    :param
        raw_training_dataset
        raw_validation_dataset
        raw_test_dataset
            the above three data have the same data structure:
                dict{
                        label (int): [
                                        [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)]
                                        ...
                                        ]
                    }
        ecm_num:
            int
    :return:
        eis_raw_imp_list
    """
    tr_data_list = raw_training_dataset[ecm_num]
    va_data_list = raw_validation_dataset[ecm_num]
    te_data_list = raw_test_dataset[ecm_num]

    eis_raw_imp_list = []
    eis_raw_imp_list.extend(tr_data_list)
    eis_raw_imp_list.extend(va_data_list)
    eis_raw_imp_list.extend(te_data_list)
    return eis_raw_imp_list

def split_eis_imp_2_real_imag(eis_imp_list):
    """
    :param eis_raw_imp_list: result from function [get_raw_dataset_by_ecm_num]
    :return:
        list[
                [
                    z_real_list[],
                    z_imag_list[]
                ],
                ...
            ]
    """
    res_list = []
    for point_list in eis_imp_list:
        x_list, y_list = point_2_x_y(point_list)
        res_list.append([x_list, y_list])
    return res_list

def get_normed_dataset_by_ecm_num(training_dataset, validation_dataset, test_dataset, ecm_num):
    """
    :param
        training_dataset, validation_dataset, test_dataset:
            list[
                    [label (int), [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)]]
                ]
            the range of x or y is 0 ~ 1
        ecm_num:
    :return:
        eis_normed_imp_list
            list[
                    [(x0, y0), (x1, y1), ..., (xn-2, yn-2), (xn-1, yn-1)],
                    ...
                ]
    """
    tr_data_list = [t_d[1] for t_d in training_dataset if t_d[0] == ecm_num]
    va_data_list = [v_d[1] for v_d in validation_dataset if v_d[0] == ecm_num]
    te_data_list = [te_d[1] for te_d in test_dataset if te_d[0] == ecm_num]

    eis_normed_imp_list = []
    eis_normed_imp_list.extend(tr_data_list)
    eis_normed_imp_list.extend(va_data_list)
    eis_normed_imp_list.extend(te_data_list)
    return eis_normed_imp_list

# if __name__ == '__main__':
#     ml_dataset_pickle_file_path = '../../../datasets/ml_datasets/normed'
#     get_TV_T_dataset(file_path=ml_dataset_pickle_file_path)