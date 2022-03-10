import xlrd

# 1- Read Lai's manual fitting results from his excel '2020-07-22-阻抗类型整理2006.xlsx'
def read_lai_manual_fitting_res(ex_fp, sheet_name):
    workbook = xlrd.open_workbook(ex_fp)
    sheet = workbook.sheet_by_name(sheet_name=sheet_name)
    cols_num = 9
    ecm2_rows = 22

    lai_manual_fit_res_dict_list = []
    # ------------------------------ read table containing manual fitting results of ECM2 ------------------------------
    # Useful content starts from '1-14' row = 3, col = 2
    for r in range(ecm2_rows):
        lai_manual_fit_res_dict = {'ecm_num':2}
        r += 3
        for i, c in enumerate(range(cols_num)):
            c += 2
            if i == 0:
                fn = sheet.cell(rowx=r, colx=c).value
                lai_manual_fit_res_dict['fn'] = fn
            elif (i > 0) and (i < cols_num - 1):
                if 'para' not in lai_manual_fit_res_dict.keys():
                    lai_manual_fit_res_dict['para'] = [float(sheet.cell(rowx=r, colx=c).value)]
                else:
                    lai_manual_fit_res_dict['para'].append(float(sheet.cell(rowx=r, colx=c).value))
            elif i == cols_num - 1:
                chi_s = float(sheet.cell(rowx=r, colx=c).value)
                lai_manual_fit_res_dict['chi_square'] = chi_s
        lai_manual_fit_res_dict_list.append(lai_manual_fit_res_dict)
    # ------------------------------ read table containing manual fitting results of ECM2 ------------------------------

    # ------------------------------ read table containing manual fitting results of ECM9 ------------------------------
    ecm9_rows = 105
    # Useful content starts from '1-1' row = 3, col = 14
    for r in range(ecm9_rows):
        lai_manual_fit_res_dict = {'ecm_num':9}
        r += 3

        try:
            # 在Lai的拟合结果中，文件 1-16 和 2-1 的结果中没有记录Chi-Square，在对应位置上用“Null”代替，在此处跳过这两个文件
            for i, c in enumerate(range(cols_num)):
                c += 14
                if i == 0:
                    fn = sheet.cell(rowx=r, colx=c).value
                    lai_manual_fit_res_dict['fn'] = fn
                elif (i > 0) and (i < cols_num - 1):
                    if 'para' not in lai_manual_fit_res_dict.keys():
                        lai_manual_fit_res_dict['para'] = [float(sheet.cell(rowx=r, colx=c).value)]
                    else:
                        lai_manual_fit_res_dict['para'].append(float(sheet.cell(rowx=r, colx=c).value))
                elif i == cols_num - 1:
                    chi_s = float(sheet.cell(rowx=r, colx=c).value)
                    lai_manual_fit_res_dict['chi_square'] = chi_s
        except ValueError as e:
            print(e, fn, '在Lai的拟合结果中，文件 1-16 和 2-1 的结果中没有记录Chi-Square，在对应位置上用“Null”代替，在此处跳过这两个文件')
            continue

        lai_manual_fit_res_dict_list.append(lai_manual_fit_res_dict)
    # ------------------------------ read table containing manual fitting results of ECM9 ------------------------------
    return lai_manual_fit_res_dict_list

def read_lai_test_coordinate(ex_fp, sheet_name):
    sheet = xlrd.open_workbook(ex_fp).sheet_by_name(sheet_name)
    row_num = sheet.nrows
    col_num = sheet.ncols

    coor_dict_list = []
    # There are five rows of test points
    for i in range(5):
        col = i * 3
        for r in range(1, row_num):
            coor_dict = {'coor':[]}
            for c in range(3):
                if c == 0:
                    fn = sheet.cell(rowx=r, colx=c + col).value
                    coor_dict['fn'] = fn
                else:
                    coor_dict['coor'].append(sheet.cell(rowx=r, colx=c + col).value)
            coor_dict_list.append(coor_dict)
    return coor_dict_list

def pack_lai_manual_fitting_res(lai_manual_fit_res_dict_list, coor_dict_list):
    for lai_manual_fit_res_dict in lai_manual_fit_res_dict_list:
        fn = lai_manual_fit_res_dict['fn']
        for coor_dict in coor_dict_list:
            if fn == coor_dict['fn']:
                lai_manual_fit_res_dict['coor'] = coor_dict['coor']
                break
    return lai_manual_fit_res_dict_list

def wrap_lai_data_4_contour(lai_manual_fit_res_dict_list):
    x_list = []
    y_list = []
    z_list = []
    for lai_manual_fit_res_dict in lai_manual_fit_res_dict_list:
        x, y = lai_manual_fit_res_dict['coor']
        z = lai_manual_fit_res_dict['chi_square']
        x_list.append(x)
        y_list.append(y)
        z_list.append(z)
    return x_list, y_list, z_list