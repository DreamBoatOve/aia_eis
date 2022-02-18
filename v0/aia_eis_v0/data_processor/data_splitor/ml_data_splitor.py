import copy
import os
import random
import pickle
import xlrd

class ML_data_splitor:
    def __init__(self, file_path, excel_name, sheet1_name, sheet2_name):
        """
        :param
            file_path:
                str
                the relative file path in the project of excel file
            excel_name:
                str
                excel file name with appendix
            sheet1_name:
                str
                the sheet name of the sheet recording the data from paper [Equivalent circuit model recognition of electrochemical impedance spectroscopy via machine learning]
            sheet2_name:
                str
                the sheet name of the sheet recording the data from Lai
            data_dict
                {
                    label(int): [
                                    [(x0, y0), (x1, y1), ..., (xn-1, yn-1)]
                                ]
                }
        """
        self.file_path = file_path
        self.excel_name = excel_name

        self.book = self.get_workbook()
        self.sheet1_name = sheet1_name
        self.sheet2_name = sheet2_name

        self.data_dict = {}
        self.read_sheet1()
        self.read_sheet2()

        self.train_data_dict, self.vali_data_dict, self.test_data_dict = self.split_raw()
        self.train_data_list, self.vali_data_list, self.test_data_list, self.normed_train_data_list, self.normed_vali_data_list, self.normed_test_data_list= self.dict_2_list()

    def get_workbook(self):
        file_abs_path = os.path.join(self.file_path, self.excel_name)
        book = xlrd.open_workbook(filename = file_abs_path)
        return book

    def read_sheet1(self):
        sheet1 = self.book.sheet_by_name(self.sheet1_name)
        col_num = sheet1.ncols

        for c_i in range(col_num):
            label = sheet1.cell_value(rowx=0, colx=c_i)
            z_list = []
            for i in range(1, 81):
                z_real = float(sheet1.cell_value(rowx=i, colx=c_i))
                z_img = float(sheet1.cell_value(rowx=i+80, colx=c_i))
                z_list.append((z_real, z_img))
            if int(label) in self.data_dict.keys():
                self.data_dict[int(label)].append(z_list)
            else:
                self.data_dict[int(label)] = [z_list]

    def read_sheet2(self):
        sheet2 = self.book.sheet_by_name(self.sheet2_name)
        row_num = sheet2.nrows
        col_num = sheet2.ncols

        for c_i in range(0, col_num, 2):
            label = sheet2.cell_value(rowx=0, colx=c_i)
            z_list = []
            for r_i in range(1, row_num):
                z_real = float(sheet2.cell_value(rowx=r_i, colx=c_i))
                z_img = float(sheet2.cell_value(rowx=r_i, colx=c_i+1))
                z_list.append((z_real, z_img))
            if int(label) in self.data_dict.keys():
                self.data_dict[int(label)].append(z_list)
            else:
                self.data_dict[int(label)] = [z_list]

    def split_raw(self):
        """
        每种标签的数据集随机抽出70%， 15%， 15%放入训练，验证，测试数据集中
        """
        train_data_dict = {}
        vali_data_dict = {}
        test_data_dict = {}

        for k in self.data_dict.keys():
            train_data_dict[k] = []
            vali_data_dict[k] = []
            test_data_dict[k] = []

        for k in self.data_dict.keys():
            data_list = copy.deepcopy(self.data_dict[k])
            data_len = len(data_list)
            while len(data_list) > 0:
                random_index = random.randint(0, len(data_list)-1)
                if len(data_list) > data_len * 0.3:
                    train_data_dict[k].append(data_list.pop(random_index))
                elif len(data_list) > data_len * 0.15:
                    vali_data_dict[k].append(data_list.pop(random_index))
                else:
                    test_data_dict[k].extend(data_list)
                    break
        return train_data_dict, vali_data_dict, test_data_dict

    def dict_2_list(self):
        train_data_list = []
        vali_data_list = []
        test_data_list = []

        normed_train_data_list = []
        normed_vali_data_list = []
        normed_test_data_list = []

        for k, data in self.train_data_dict.items():
            for d in data:
                train_data_list.append([k, d])
                normed_train_data_list.append([k, normalize(d)])
        for k, data in self.vali_data_dict.items():
            for d in data:
                vali_data_list.append([k, d])
                normed_vali_data_list.append([k, normalize(d)])
        for k, data in self.test_data_dict.items():
            for d in data:
                test_data_list.append([k, d])
                normed_test_data_list.append([k, normalize(d)])
        return train_data_list, vali_data_list, test_data_list, normed_train_data_list, normed_vali_data_list, normed_test_data_list

    def serialize(self, data, file_name):
        with open(file_name, 'wb') as file:
            pickle.dump(data, file)

    def serialize_2_pickle(self):
        train_name = 'ml_raw_train_data_dict_pickle_2020_03_07.file'
        vali_name = 'ml_raw_vali_data_dict_pickle_2020_03_07.file'
        test_name = 'ml_raw_test_data_dict_pickle_2020_03_07.file'
        with open(train_name, 'wb') as file:
            pickle.dump(self.train_data_dict, file)
        with open(vali_name, 'wb') as file:
            pickle.dump(self.vali_data_dict, file)
        with open(test_name, 'wb') as file:
            pickle.dump(self.test_data_dict, file)

def normalize(data_list):
    """
    :param
        data_list:
            [(x0, y0), (x1, y1), ..., (xn-1, yn-1)]
    """
    x_list = []
    y_list = []
    for pair in data_list:
        x_list.append(pair[0])
        y_list.append(pair[1])

    x_min = min(x_list)
    x_max = max(x_list)
    x_range = x_max - x_min

    y_min = min(y_list)
    y_max = max(y_list)
    y_range = y_max - y_min

    r = max(x_range, y_range)

    normed_data_list = [((x - x_min) / r, (y - y_min) / r) for x, y in zip(x_list, y_list)]
    return normed_data_list

def serialize_by_pickle(data_list, filename):
    with open(filename, 'wb') as file:
        pickle.dump(data_list, file)

# if __name__ == '__main__':
#     mds = ML_data_splitor(file_path='../../../datasets/experiement_data/processed_dataset',\
#                           excel_name='whole_dataset.xlsx',\
#                           sheet1_name='Sheet1_real', \
#                           sheet2_name='Sheet2_pad')
#     # mds.serialize_2_pickle()
#     mds.serialize(mds.normed_train_data_list, 'ml_normed_train_data_list_pickle_2020_03_07.file')
#     mds.serialize(mds.normed_vali_data_list, 'ml_normed_vali_data_list_pickle_2020_03_07.file')
#     mds.serialize(mds.normed_test_data_list, 'ml_normed_test_data_list_pickle_2020_03_07.file')