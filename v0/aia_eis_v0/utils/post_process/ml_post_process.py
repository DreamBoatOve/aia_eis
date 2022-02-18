import xlrd
from utils.file_utils.filename_utils import get_date_prefix

def single_para_boxplot_data_wrapper(table_start_row, table_start_col, acc_kappa_margin, sheet_name, excel_abs_path):
    """
        ML with one adjustable parameter, read and sort results in excel into a format can be used to plot boxplot
        :param:
            table_start_row
                the first filename's row
            table_start_col
                the first filename's col
            acc_kappa_margin
                the distance between the filename_column[0] and the acc_kappa_column
                acc_kappa_column = filename_column[0] + acc_kappa_margin
            sheet_name
            excel_name
            txt_filename
        :return:
            data_list
                list[
                    list[float, ...]
                ]
            label_list
                list[para, ...]
    """
    """
    每次运行程序前，要移除这段注释，否则会出现 SyntaxError: (unicode error) 'unicodeescape' codec can't decode bytes in position 392-393: truncated \xXX escape
    1- raise XLRDError(FILE_FORMAT_DESCRIPTIONS[file_format]+'; not supported') xlrd.biffh.XLRDError: Excel xlsx file; not supported
        目前xlrd=2.0.1的新版本是不支持‘xlsx’格式的文件：在xlrd的python\install\lib\site-packages\xlrd\__init__.py", line 170, in open_workbook
            if file_format and file_format != 'xls':
                raise XLRDError(FILE_FORMAT_DESCRIPTIONS[file_format]+'; not supported')
    2- xlsx和xls有什么区别？
        区别：
            1、xls是excel 2007之前版本的使用的默认格式；xlsx是excel 2007之后的版本使用的默认格式，包括2007的版本。
            2、XLSX格式的占用空间比XLS的小
    """
    # 1-Load excel
    workbook = xlrd.open_workbook(excel_abs_path)
    # 2-load sheet
    sheet = workbook.sheet_by_name(sheet_name=sheet_name)

    # print(sheet.cell_value(rowx = table_start_row, colx = table_start_col))
    # Get the row number (length) of the table
    table_len = 1
    try:
        # cell_type: 0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格）
        while sheet.cell_type(rowx=table_start_row + table_len, colx=table_start_col) != 0:
            print(sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col))
            table_len += 1
    except IndexError as e:
        print('Reach the bottom of the table')

    data_list = []
    label_list = []
    for i in range(int(table_len / 10)):
        # Take a parameter's result as a block, use the left-up corner's coordinate to locate the block
        left_up_coor = [table_start_row + i * 10, table_start_col]
        para = int(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1] + 1))
        acc_kappa_list = [float(sheet.cell_value(rowx=left_up_coor[0] + j, colx=left_up_coor[1] + acc_kappa_margin)) for j in range(10)]
        data_list.append(acc_kappa_list)
        label_list.append(para)
    return data_list, label_list

def single_para_sorter(table_start_row, table_start_col, sheet_name, excel_abs_path, txt_filename):
    """
    Function
        Some ML use grid search to find optimal parameters, and store results in excel
        this function read grid search results from excel, and sort them out
        1- For a parameter setting, it was tested for 10 times, remove the best and the worst performance, then average
            the left 8 results (Acc + Kappa)
        2- Store these results at the left side of the original data, and copy them into a txt file for heatmap plot
        3- Pick out the first 5 parameter setting for further investigation
        4- The results in the table should the following data distribution:
            filename, para_0, para_1, No., Acc, Kappa, Acc+Kappa
        5- Use the first filename's coordinate (table_start_row, table_start_col) to locate the whole table
    :param:
        table_start_row
            the first filename's row
        table_start_col
            the first filename's col
        sheet_name
        excel_name
        txt_filename
    :return:
        read above content into a txt file
    """
    # 1-Load excel
    workbook = xlrd.open_workbook(excel_abs_path)
    # 2-load sheet
    sheet = workbook.sheet_by_name(sheet_name=sheet_name)

    # print(sheet.cell_value(rowx = table_start_row, colx = table_start_col))
    # Get the row number (length) of the table
    table_len = 1
    try:
        # cell_type: 0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格）
        while sheet.cell_type(rowx=table_start_row + table_len, colx=table_start_col) != 0:
            # print(sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col))
            table_len += 1
    except IndexError as e:
        print('Reach the bottom of the table')

    for i in range(int(table_len / 10)):
        # Take a parameter's result as a block, use the left-up corner's coordinate to locate the block
        left_up_coor = [table_start_row + i * 10, table_start_col]
        para = int(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1] + 1))
        acc_kappa_list = [float(sheet.cell_value(rowx=left_up_coor[0] + j, colx=left_up_coor[1] + 4)) for j in range(10)]
        # Remove the biggest and smallest acc_kappa
        acc_kappa_list.remove(max(acc_kappa_list))
        acc_kappa_list.remove(min(acc_kappa_list))

        avg_acc_kappa = sum(acc_kappa_list) / len(acc_kappa_list)
        with open(txt_filename, 'a+') as file:
            line = ','.join([str(para), str(avg_acc_kappa)]) + '\n'
            file.write(line)

def binary_para_sorter(table_start_row, table_start_col, sheet_name, excel_path, txt_filename,
                       para0_margin=1, para1_margin=2):
    """
    Function
        Some ML use grid search to find optimal parameters, and store results in excel
        this function read grid search results from excel, and sort them out
        1- For a parameter setting, it was tested for 10 times, remove the best and the worst performance, then average
            the left 8 results (Acc + Kappa)
        2- Store these results at the left side of the original data, and copy them into a txt file for heatmap plot
        3- Pick out the first 5 parameter setting for further investigation
        4-The results in the table should the following data distribution:
            filename, para_0, para_1, No., Acc, Kappa, Acc+Kappa
        5-Use the first filename's coordinate (table_start_row, table_start_col) to locate the whole table
    :param:
        table_start_row
            the first model filename's row
        table_start_col
            the first model filename's col
        sheet_name
        excel_name
        txt_filename
    :return:
        read above content into a txt file
    """
    # 1-Load excel
    workbook = xlrd.open_workbook(excel_path)
    # 2-load sheet
    sheet = workbook.sheet_by_name(sheet_name=sheet_name)

    # print(sheet.cell_value(rowx=table_start_row, colx=table_start_col))
    # Get the row number (length) of the table
    table_len = 1
    try:
        # cell_type: 0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格）
        while sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col) != 0:
            # print(sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col))
            table_len += 1
    except IndexError as e:
        print('Reach the bottom of the table')

    for i in range(int(table_len / 10)):
        # Take a parameter's result(model' pickle filename) as a block, use the left-up corner's coordinate to locate the block
        left_up_coor = [table_start_row + i * 10, table_start_col]
        para_0 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para0_margin))
        para_1 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para1_margin))
        acc_kappa_list = [float(sheet.cell_value(rowx=left_up_coor[0] + j, colx=left_up_coor[1] + 6)) for j in range(10)]
        # Remove the biggest and smallest acc_kappa
        acc_kappa_list.remove(max(acc_kappa_list))
        acc_kappa_list.remove(min(acc_kappa_list))

        avg_acc_kappa = sum(acc_kappa_list) / len(acc_kappa_list)
        with open(txt_filename, 'a+') as file:
            line = ','.join([str(para_0), str(para_1), str(avg_acc_kappa)]) + '\n'
            file.write(line)
# ------------------------------ Average LRC grid search results --------------------------
# 整理Lrc-Ovr的mpi运行出的结果
# txt_filename = get_date_prefix() + 'lrc_ovr_linear_avg_res.txt'
# binary_para_sorter(table_start_row=3, table_start_col=21, sheet_name='LRC', excel_path='../../ml_sl/ml_training_records.xlsx',\
#                    txt_filename=txt_filename, para1_margin=1, para2_margin=2)
# ------------------------------ Average LRC grid search results --------------------------

# ------------------------------ Average SVM-OvO-Linear grid search results --------------------------
# txt_filename = get_date_prefix() + 'svm_ovo_linear_gs_avg_res.txt'
# binary_para_sorter(table_start_row=3, table_start_col=0, sheet_name='SVM', excel_path='../../ml_sl/ml_training_records.xlsx',\
#                    txt_filename=txt_filename, para1_margin=1, para2_margin=2)
# ------------------------------ Average SVM-OvO-Linear grid search results --------------------------

# ------------------------------ Average SVM-OvR-Linear grid search results --------------------------
# txt_filename = get_date_prefix() + 'svm_ovr_linear_gs_avg_res.txt'
# binary_para_sorter(table_start_row=3, table_start_col=27, sheet_name='SVM', excel_path='../../ml_sl/ml_training_records.xlsx',\
#                    txt_filename=txt_filename, para1_margin=1, para2_margin=2)
# ------------------------------ Average SVM-OvR-Linear grid search results --------------------------

def triple_para_sorter(table_start_row, table_start_col, sheet_name, excel_path, txt_filename,
                       para0_margin=1, para1_margin=2, para2_margin=3, target_margin=7):
    """
    Function
        Some ML use grid search to find optimal parameters, and store results in excel
        this function read grid search results from excel, and sort them out
        1- For a parameter setting, it was tested for 10 times, remove the best and the worst performance, then average
            the left 8 results (Acc + Kappa)
        2- Store these results at the left side of the original data, and copy them into a txt file for Parallel Coordinates plot
        3- Pick out the first 8 parameter setting for further investigation
        4- The results in the table should the following data distribution:
            filename, para_0, para_1, para_2, No., Acc, Kappa, Acc + Kappa
        5- Use the first filename's coordinate (table_start_row, table_start_col) to locate the whole table
    :param:
        table_start_row
            the first model filename's row
        table_start_col
            the first model filename's col
        sheet_name
        excel_name
        txt_filename
    :return:
        read above content into a txt file
    """
    # 1-Load excel
    workbook = xlrd.open_workbook(excel_path)
    # 2-load sheet
    sheet = workbook.sheet_by_name(sheet_name=sheet_name)

    # print(sheet.cell_value(rowx=table_start_row, colx=table_start_col))
    # Get the row number (length) of the table
    table_len = 1
    try:
        # cell_type: 0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格）
        while sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col) != 0:
            # print(sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col))
            table_len += 1
    except IndexError as e:
        print('Reach the bottom of the table')

    for i in range(int(table_len / 10)):
        # Take a parameter's result(model' pickle filename) as a block/anchor, use the left-up corner's coordinate to locate the block
        left_up_coor = [table_start_row + i * 10, table_start_col]
        para_0 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para0_margin))
        para_1 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para1_margin))
        para_2 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para2_margin))
        acc_kappa_list = [float(sheet.cell_value(rowx=left_up_coor[0] + j, colx=left_up_coor[1] + target_margin)) for j in range(10)]
        # Remove the biggest and smallest acc_kappa
        acc_kappa_list.remove(max(acc_kappa_list))
        acc_kappa_list.remove(min(acc_kappa_list))

        avg_acc_kappa = sum(acc_kappa_list) / len(acc_kappa_list)
        with open(txt_filename, 'a+') as file:
            line = ','.join([str(para_0), str(para_1), str(para_2), str(avg_acc_kappa)]) + '\n'
            file.write(line)
# ------------------------------ Averaged SVM-OvO-RBF grid search results ------------------------------
# txt_filename = get_date_prefix() + 'svm_ovo_rbf_gs_avg_res.txt'
# triple_para_sorter(table_start_row=3, table_start_col=18, sheet_name='SVM', excel_path='../../ml_sl/ml_training_records.xlsx',
#                    txt_filename=txt_filename, para0_margin=1, para1_margin=2, para2_margin=3, target_margin=7)
# Three column in txt are : Iteration, C, Sigma
# ------------------------------ Averaged SVM-OvO-RBF grid search results ------------------------------

# ------------------------------ Averaged SVM-OvR-RBF grid search results ------------------------------
# txt_filename = get_date_prefix() + 'svm_ovr_rbf_gs_avg_res.txt'
# triple_para_sorter(table_start_row=3, table_start_col=45, sheet_name='SVM', excel_path='../../ml_sl/ml_training_records.xlsx',
#                    txt_filename=txt_filename, para0_margin=1, para1_margin=2, para2_margin=3, target_margin=7)
# ------------------------------ Averaged SVM-OvR-RBF grid search results ------------------------------

# ------------------------------ Averaged SVM-OvO-Poly grid search results ------------------------------
# txt_filename = get_date_prefix() + 'svm_ovo_poly_gs_avg_res.txt'
# triple_para_sorter(table_start_row=3, table_start_col=8, sheet_name='SVM', excel_path='../../ml_sl/ml_training_records.xlsx',
#                     txt_filename=txt_filename, para0_margin=1, para1_margin=2, para2_margin=4, target_margin=8)
# ------------------------------ Averaged SVM-OvO-Poly grid search results ------------------------------

# ------------------------------ Averaged SVM-OvR-Poly grid search results ------------------------------
# txt_filename = get_date_prefix() + 'svm_ovr_poly_gs_avg_res.txt'
# triple_para_sorter(table_start_row=3, table_start_col=35, sheet_name='SVM', excel_path='../../ml_sl/ml_training_records.xlsx',
#                     txt_filename=txt_filename, para0_margin=1, para1_margin=2, para2_margin=4, target_margin=8)
# ------------------------------ Averaged SVM-OvR-Poly grid search results ------------------------------

def qua_para_sorter(table_start_row, table_start_col, sheet_name, excel_path, txt_filename,
                    para0_margin=1, para1_margin=2, para2_margin=3, para3_margin=4, target_margin=8):
    """
    Function
        Some ML use grid search to find optimal parameters, and store results in excel
        this function read grid search results from excel, and sort them out
        1- For a parameter setting, it was tested for 10 times, remove the best and the worst performance, then average
            the left 8 results (Acc + Kappa)
        2- Store these results at the left side of the original data, and copy them into a txt file for Parallel Coordinates plot
        3- Pick out the first 9 parameter setting for further investigation
        4- The results in the table should the following data distribution:
            filename, para_0, para_1, para_2, para_3, No., Acc, Kappa, Acc + Kappa
        5- Use the first filename's coordinate (table_start_row, table_start_col) to locate the whole table
    :param:
        table_start_row
            the first model filename's row
        table_start_col
            the first model filename's col
        sheet_name
        excel_name
        txt_filename
    :return:
        read above content into a txt file
    """
    # 1-Load excel
    workbook = xlrd.open_workbook(excel_path)
    # 2-load sheet
    sheet = workbook.sheet_by_name(sheet_name=sheet_name)

    # print(sheet.cell_value(rowx=table_start_row, colx=table_start_col))
    # Get the row number (length) of the table
    table_len = 1
    try:
        # cell_type: 0. empty（空的）,1 string（text）, 2 number, 3 date, 4 boolean, 5 error， 6 blank（空白表格）
        while sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col) != 0:
            # print(sheet.cell_type(rowx = table_start_row + table_len, colx = table_start_col))
            table_len += 1
    except IndexError as e:
        print('Reach the bottom of the table')

    for i in range(int(table_len / 10)):
        # Take a parameter's result(model' pickle filename) as a block/anchor, use the left-up corner's coordinate to locate the block
        left_up_coor = [table_start_row + i * 10, table_start_col]
        para_0 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para0_margin))
        para_1 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para1_margin))
        para_2 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para2_margin))
        para_3 = float(sheet.cell_value(rowx=left_up_coor[0], colx=left_up_coor[1]+para3_margin))
        acc_kappa_list = [float(sheet.cell_value(rowx=left_up_coor[0] + j, colx=left_up_coor[1] + target_margin)) for j in range(10)]
        # Remove the biggest and smallest acc_kappa
        acc_kappa_list.remove(max(acc_kappa_list))
        acc_kappa_list.remove(min(acc_kappa_list))

        avg_acc_kappa = sum(acc_kappa_list) / len(acc_kappa_list)
        with open(txt_filename, 'a+') as file:
            line = ','.join([str(para_0), str(para_1), str(para_2), str(para_3), str(avg_acc_kappa)]) + '\n'
            file.write(line)