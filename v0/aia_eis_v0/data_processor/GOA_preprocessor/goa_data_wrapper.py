import os
import xlrd
import pickle

from utils.file_utils.filename_utils import get_date_prefix
from playground.laiZhaoGui.goa.GOAs_fit_EIS_0 import get_para_range

# Read Lai's EIS data from 'whole_dataset.xlsx---Sheet2'
def read_Lai_EIS_data(excel_name, sheet_name):
    workbook = xlrd.open_workbook(excel_name)
    sheet2 = workbook.sheet_by_name(sheet_name)

    row_num = sheet2.nrows
    col_num = sheet2.ncols

    eis_data_dict_list = []
    for f_i in range(int(col_num / 3)):
        col_i = f_i * 3
        eis_data_dict = {}
        eis_data_dict['file_name'] = str(sheet2.cell_value(rowx=0, colx=col_i))
        eis_data_dict['ecm_num'] = int(sheet2.cell_value(rowx=1, colx=col_i))

        fre_list = []
        z_list = []
        for r_i in range(2, row_num):
            # 0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格 == ''）
            if (sheet2.cell_value(rowx=r_i, colx=col_i) != 0) and (sheet2.cell_value(rowx=r_i, colx=col_i) != ''):
                try:
                    fre_list.append(float(sheet2.cell_value(rowx=r_i, colx=col_i)))
                except ValueError as e:
                    print(e)
                    print('Row = {0}, Col = {1}, cell content = {2}'.format(r_i, col_i, sheet2.cell_value(rowx=r_i, colx=col_i)))
                z = float(sheet2.cell_value(rowx=r_i, colx=col_i+1)) + float(sheet2.cell_value(rowx=r_i, colx=col_i+2)) * 1j
                z_list.append(z)
        eis_data_dict['f'] = fre_list
        eis_data_dict['z_raw'] = z_list
        eis_data_dict_list.append(eis_data_dict)

    # --------------------- Generate file name: Exp: '2020_03_24_goa_lai_dataset_pickle.file' ---------------------
    # import datetime
    # year_str = str(datetime.datetime.now().year)
    # month_str = str(datetime.datetime.now().month)
    # day_str = str(datetime.datetime.now().day)
    # filename = year_str+'_'+month_str+'_'+day_str+'_goa_lai_dataset_pickle.file'
    filename = get_date_prefix() + 'goa_lai_dataset_pickle.file'
    # --------------------- Generate file name: Exp: '2020_03_24_goa_lai_dataset_pickle.file' ---------------------

    with open(filename, 'wb') as file:
        pickle.dump(eis_data_dict_list, file)
    return eis_data_dict_list

def load_Lai_EIS_data(file_path, file_name):
    file_abs_path = os.path.join(file_path, file_name)
    with open(file_abs_path, 'rb') as file:
        eis_data_dict_list = pickle.load(file)
    return eis_data_dict_list

# ----------------------------------- Generate Lai's EIS Raw data as pickle file ----------------------------------------------
# ------- Using 2020-07-22-阻抗类型整理2006.xlsx -------
# excel_name = '../../../datasets/experiement_data/processed_dataset/whole_dataset.xlsx'
# sheet_name = '2020_07_22_lai_all_EIS_data'
# read_Lai_EIS_data(excel_name, sheet_name)
# Generated EIS raw pickle dataset files: 2020_08_19_goa_lai_dataset_pickle.file
# ------- Using 2020-07-22-阻抗类型整理2006.xlsx -------

# ------- Using 2020-01-04-阻抗类型整理200103.xlsx -------
# excel_name = '../../../datasets/experiement_data/processed_dataset/whole_dataset.xlsx'
# sheet_name = '2020_01_04_lai_all_EIS_data'
# read_Lai_EIS_data(excel_name, sheet_name)
# Generated EIS raw pickle dataset files: 2020_03_26_goa_lai_dataset_pickle.file and 2020_03_24_goa_lai_dataset_pickle.file
# ------- Using 2020-01-04-阻抗类型整理200103.xlsx -------
# ----------------------------------- Generate Lai's EIS Raw data as pickle file ----------------------------------------------

def norm_lai_EIS_data():
    # The raw impedance multiply experiment area (1.01 * 1e-6) ==> normed impedance (/ cm^2)
    exp_area = 1.01 * 1e-6 # Lai's experimental area 1.01 * 1e-6 cm^2
    lai_raw_eis_data_dict_list = load_Lai_EIS_data(file_path='../../datasets/goa_datasets/raw',
                                                   file_name='2020_08_19_goa_lai_dataset_pickle.file')
    fn = get_date_prefix() + 'goa_lai_normed_dataset_pickle.file'
    for raw_eis_data_dict in lai_raw_eis_data_dict_list:
        # Wrong: Z / exp_area raw_eis_data_dict['z_raw'] = [z / exp_area for z in raw_eis_data_dict['z_raw']]
        # Right way:
        raw_eis_data_dict['z_raw'] = [z * exp_area for z in raw_eis_data_dict['z_raw']]
    with open(fn, 'wb') as file:
        pickle.dump(lai_raw_eis_data_dict_list, file)

# ----------------------------------- Generate Lai's EIS Normed data as pickle file ----------------------------------------------
# norm_lai_EIS_data()
# Generated EIS normed pickle dataset files: WRONG 2020_08_19_goa_lai_normed_dataset_pickle.file
# Generated EIS normed pickle dataset files: Right 2020_08_22_goa_lai_normed_dataset_pickle.file
# ----------------------------------- Generate Lai's EIS Normed data as pickle file ----------------------------------------------

def load_lai_manual_fitting_res(file_path, file_names, mag_num=2):
    """
    :param file_path:
    :param file_names:
    :param mag_num:
    :return:
        lai_manual_fit_res_dict{
            '1-14':{
                'para': [0.01839, 0.006388, 0.8688, 1.175, 0.002783, 0.798, 1371.0],
                'limit': [[0.0001, 1], [1e-05, 0.1], [0.3, 1.0], [0.01, 100], [1e-05, 0.1], [0.3, 1.0], [10, 100000]],
                'chi_square': 0.001314
            },
            '2-13':...,
        }
    """
    lai_manual_fit_res_dict = {}
    for fn in file_names:
        with open(os.path.join(file_path, fn)) as file:
            for line in file.readlines():
                line_str_list = line.strip().split(',')
                fn = line_str_list[1]

                # 处理不了 ValueError: could not convert string to float: '1.519E004.'
                # para_list = list(map(float, line_str_list[2:len(line_str_list)-1]))
                para_list = []
                limit_list = []
                for i, num_str in enumerate(line_str_list[2: len(line_str_list) - 1]):
                    try:
                        para = float(num_str)
                    except ValueError as e:
                        # ValueError: could not convert string to float: '1.519E004.'
                        num_str_list = num_str.split('E')
                        # ValueError: invalid literal for int() with base 10: '004.'
                        if '.' == num_str_list[1][-1]:
                            num_str_list[1] = num_str_list[1].split('.')[0]
                        para = float(num_str_list[0]) * pow(10, int(num_str_list[1]))
                    para_list.append(para)
                    # num_str_in_scientific_form =
                    # 2 and 5 are indexes of two CPE_ns in ECM2 or ECM9
                    if i in [2, 5]:
                        # limit_pair = [para - 0.1, 1.0]
                        limit_pair = [0.3, 1.0]
                        limit_list.append(limit_pair)
                    else:
                        limit_list.append(get_para_range(num=para, mag_num=mag_num))
                try:
                    chi_square = float(line_str_list[-1])
                except ValueError as e:
                    """
                    Lai的一些拟合结果没有Ch-Square误差，我在Excel中就用Null字符串代替，需要在此处特殊处理一下，
                    因为Lai的拟合误差大多数都是1e-3，所以就用1e-3来代替空缺值
                    """
                    if line_str_list[-1] == 'Null':
                        chi_square = 1e-3
                lai_manual_fit_res_dict[fn] = {'para': para_list, 'limit':limit_list, 'chi_square':chi_square}
    return lai_manual_fit_res_dict