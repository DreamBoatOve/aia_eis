import xlrd
import xlwt
import os
import matplotlib.pyplot as plt

from data_processor.ml_preprocessor.quadratic_fun_fitter import func as qua_func
from data_processor.ml_preprocessor.quadratic_fun_fitter import qua_fun_fit
from data_processor.ml_preprocessor.cubic_fun_fitter import func as cub_func
from data_processor.ml_preprocessor.cubic_fun_fitter import cub_fun_fit

class low_dim_2_high_dim:
    def __init__(self, file_path, excel_name, sheet_name):
        self.file_path = file_path
        self.excel_name = excel_name
        self.sheet_name = sheet_name

        self.raw_data_list, self.label_list = self.read_excel()
        self.fir_pad_data_list = self.pad_data_31_2_41_1()
        self.sec_pad_data_list = self.pad_data_41_2_80_0()

    def read_excel(self):
        book_path = os.path.join(self.file_path, self.excel_name)
        book = xlrd.open_workbook(filename=book_path)
        sheet = book.sheet_by_name(self.sheet_name)
        row_num = sheet.nrows
        col_num = sheet.ncols

        data_list = []
        for c_i in range(col_num):
            if c_i % 3 == 0:
                chunk_list = []
                for r_i in range(1, row_num):
                    try:
                        cell_0_value = float(sheet.cell_value(rowx=r_i, colx=c_i))
                        cell_1_value = float(sheet.cell_value(rowx=r_i, colx=c_i+1))
                        cell_2_value = float(sheet.cell_value(rowx=r_i, colx=c_i+2))
                    except ValueError as e:
                        print(e)
                        cell = sheet.cell(rowx=r_i, colx=c_i)
                        # cell的数据类型有: 0 empty, 1 string, 2 number, 3 date, 4 boolean, 5 error
                        print('cell data type:', cell.ctype)
                        print('cell position: row ', r_i, 'col ', c_i)
                    tmp_list = [cell_0_value, cell_1_value, cell_2_value]
                    chunk_list.append(tmp_list)
                data_list.append(chunk_list)
            else:
                continue

        label_list = [sheet.cell_value(rowx=0, colx=c_i) for c_i in range(col_num) if sheet.cell(rowx=0, colx=c_i).ctype != 0]
        return data_list, label_list

    def pad_data_31_2_41_0(self):
        all_z_pair_list = []
        """
        chunk 
            一个chunk代表一个EIS数据
            [
                [impedance, Zreal, Zimg],
                [impedance, Zreal, Zimg],
                ...
            ]
        """
        for chunk in self.raw_data_list:
            z_pair_list = []
            # length 是31，range(0, 30) = [0,3,6,...,27] 10 points
            for r_i in range(0, len(chunk)-1, 3):
                # 首先判断点3，4(p2, p3)是否呈V型
                p0 = chunk[r_i][1:]
                p1 = chunk[r_i+1][1:]
                p2 = chunk[r_i+2][1:]
                p3 = chunk[r_i+3][1:]

                try:
                    k12 = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    k23 = (p3[1] - p2[1]) / (p3[0] - p2[0])
                except IndexError as e:
                    print('row', r_i)
                    print('p1', p1)
                    print('p2', p2)
                    print(e)
                x_list = [p[0] for p in [p0, p1, p2]]
                y_list = [p[1] for p in [p0, p1, p2]]
                para = qua_fun_fit(x_list, y_list)[0]
                if k12 * k23 < 0:
                    x = 0.5 * (p1[0] + p2[0])
                    y = qua_func(para, x)
                    z_pair_list.extend([p0, p1, [x, y], p2])
                else:
                    x = 0.5 * (p2[0] + p3[0])
                    y = qua_func(para, x)
                    z_pair_list.extend([p0, p1, p2, [x, y]])
            z_pair_list.append(chunk[-1][1:])
            all_z_pair_list.append(z_pair_list)
        return all_z_pair_list

    def pad_data_31_2_41_1(self):
        all_z_pair_list = []
        """
        chunk 
            一个chunk代表一个EIS数据
            [
                [impedance, Zreal, Zimg],
                [impedance, Zreal, Zimg],
                ...
            ]
        """
        for chunk in self.raw_data_list:
            z_pair_list = []
            # length 是31，range(0, 30) = [0,3,6,...,27] 10 points
            for r_i in range(0, len(chunk)-1, 3):
                # 首先判断点3，4(p2, p3)是否呈V型
                p0 = chunk[r_i][1:]
                p1 = chunk[r_i+1][1:]
                p2 = chunk[r_i+2][1:]
                p3 = chunk[r_i+3][1:]

                try:
                    k01 = (p1[1] - p0[1]) / (p1[0] - p0[0])
                    k12 = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    k23 = (p3[1] - p2[1]) / (p3[0] - p2[0])
                except IndexError as e:
                    print('row', r_i)
                    print('p0', p0)
                    print('p1', p1)
                    print('p2', p2)
                    print(e)

                if (k01 * k12 >= 0) and (k12 * k23 >= 0):
                    x_list = [p[0] for p in [p0, p1, p2]]
                    y_list = [p[1] for p in [p0, p1, p2]]
                    para = qua_fun_fit(x_list, y_list)[0]
                    x = 0.5 * (p2[0] + p3[0])
                    y = qua_func(para, x)
                    z_pair_list.extend([p0, p1, p2, [x,y]])

                elif (k01 * k12 >= 0) and (k12 * k23 < 0):
                    x_list = [p[0] for p in [p0, p1, p2]]
                    y_list = [p[1] for p in [p0, p1, p2]]
                    para = qua_fun_fit(x_list, y_list)[0]
                    x = 0.5 * (p1[0] + p2[0])
                    y = qua_func(para, x)
                    z_pair_list.extend([p0, p1, [x, y], p2])

                elif (k01 * k12 < 0) and (k12 * k23 >= 0):
                    x_list = [p[0] for p in [p1, p2, p3]]
                    y_list = [p[1] for p in [p1, p2, p3]]
                    para = qua_fun_fit(x_list, y_list)[0]
                    x = 0.5 * (p2[0] + p3[0])
                    y = qua_func(para, x)
                    z_pair_list.extend([p0, p1, p2, [x, y]])

                elif (k01 * k12 < 0) and (k12 * k23 < 0):
                    x_list = [p[0] for p in [p0, p1, p2, p3]]
                    y_list = [p[1] for p in [p0, p1, p2, p3]]
                    para = qua_fun_fit(x_list, y_list)[0]
                    x = 0.5 * (p0[0] + p1[0])
                    y = qua_func(para, x)
                    z_pair_list.extend([p0, [x, y], p1, p2])

            z_pair_list.append(chunk[-1][1:])
            all_z_pair_list.append(z_pair_list)
        return all_z_pair_list

    def pad_data_41_2_80_0(self):
        all_z_pair_list = []
        """
        chunk 
            一个chunk代表一个EIS数据
            [
                [impedance, Zreal, Zimg],
                [impedance, Zreal, Zimg],
                ...
            ]
        """
        for chunk in self.fir_pad_data_list:
            z_pair_list = []
            # length 是41-->+39-->80，range(0, 41) = [0,3,6,...,36(yes, 13), 39(no, 14)] 10 points
            for r_i in range(0, len(chunk) - 2, 3):
                # 首先判断点3，4(p2, p3)是否呈V型
                p0 = chunk[r_i]
                p1 = chunk[r_i + 1]
                p2 = chunk[r_i + 2]
                p3 = chunk[r_i + 3]

                try:
                    k01 = (p1[1] - p0[1]) / (p1[0] - p0[0])
                    k12 = (p2[1] - p1[1]) / (p2[0] - p1[0])
                    k23 = (p3[1] - p2[1]) / (p3[0] - p2[0])
                except IndexError as e:
                    print('row', r_i)
                    print('p0', p0)
                    print('p1', p1)
                    print('p2', p2)
                    print(e)

                if (k01 * k12 >= 0) and (k12 * k23 >= 0):
                    x_list = [p[0] for p in [p0, p1, p2, p3]]
                    y_list = [p[1] for p in [p0, p1, p2, p3]]
                    para = cub_fun_fit(x_list, y_list)[0]

                    x0 = 0.5 * (p0[0] + p1[0])
                    y0 = cub_func(para, x0)

                    x1 = 0.5 * (p1[0] + p2[0])
                    y1 = cub_func(para, x1)

                    x2 = 0.5 * (p2[0] + p3[0])
                    y2 = cub_func(para, x2)
                    z_pair_list.extend([p0, [x0, y0], p1, [x1, y1], p2, [x2, y2]])

                elif (k01 * k12 >= 0) and (k12 * k23 < 0):
                    x_list = [p[0] for p in [p0, p1, p2]]
                    y_list = [p[1] for p in [p0, p1, p2]]
                    para = qua_fun_fit(x_list, y_list)[0]

                    x0 = p0[0] + (p2[0] - p1[0]) * 1 / 4
                    x1 = p0[0] + (p2[0] - p1[0]) * 2 / 4
                    x2 = p0[0] + (p2[0] - p1[0]) * 3 / 4

                    y0 = qua_func(para, x0)
                    y1 = qua_func(para, x1)
                    y2 = qua_func(para, x2)

                    if p1[0] < x0:
                        z_pair_list.extend([p0, p1, [x0, y0], [x1, y1], [x2, y2], p2])
                    elif p1[0] < x1:
                        z_pair_list.extend([p0, [x0, y0], p1, [x1, y1], [x2, y2], p2])
                    elif p1[0] < x2:
                        z_pair_list.extend([p0, [x0, y0], [x1, y1], p1, [x2, y2], p2])
                    else:
                        z_pair_list.extend([p0, [x0, y0], [x1, y1], [x2, y2], p1, p2])

                elif (k01 * k12 < 0) and (k12 * k23 >= 0):
                    x_list = [p[0] for p in [p1, p2, p3]]
                    y_list = [p[1] for p in [p1, p2, p3]]
                    para = qua_fun_fit(x_list, y_list)[0]

                    x0 = p1[0] + (p3[0] - p1[0]) * 1 / 4
                    x1 = p1[0] + (p3[0] - p1[0]) * 2 / 4
                    x2 = p1[0] + (p3[0] - p1[0]) * 3 / 4

                    y0 = qua_func(para, x0)
                    y1 = qua_func(para, x1)
                    y2 = qua_func(para, x2)

                    if p2[0] < x0:
                        z_pair_list.extend([p0, p1, p2, [x0, y0], [x1, y1], [x2, y2]])
                    elif p2[0] < x1:
                        z_pair_list.extend([p0, p1, [x0, y0], p2, [x1, y1], [x2, y2]])
                    elif p2[0] < x2:
                        z_pair_list.extend([p0, p1, [x0, y0], [x1, y1], p2, [x2, y2]])
                    else:
                        z_pair_list.extend([p0, p1, [x0, y0], [x1, y1], [x2, y2], p2])

                elif (k01 * k12 < 0) and (k12 * k23 < 0):
                    x_list = [p[0] for p in [p0, p1, p2, p3]]
                    y_list = [p[1] for p in [p0, p1, p2, p3]]
                    para = cub_fun_fit(x_list, y_list)[0]

                    x0 = 0.5 * (p0[0] + p1[0])
                    x1 = 0.5 * (p1[0] + p2[0])
                    x2 = 0.5 * (p2[0] + p3[0])

                    y0 = cub_func(para, x0)
                    y1 = cub_func(para, x1)
                    y2 = cub_func(para, x2)

                    z_pair_list.extend([p0, [x0, y0], p1, [x1, y1], p2, [x2, y2]])

            z_pair_list.extend(chunk[39:])
            all_z_pair_list.append(z_pair_list)
        return all_z_pair_list

    def plot(self, chunk_index):
        # Original data
        chunk = self.raw_data_list[chunk_index]
        # First pad data, 31-->41
        first_pad = self.fir_pad_data_list[chunk_index]
        # Second pad data, 41-->80
        sec_pad = self.sec_pad_data_list[chunk_index]
        print('original length', len(chunk))
        print('first pad length', len(first_pad))
        print('second pad length', len(sec_pad))

        raw_x_list = [c[1] for c in chunk]
        raw_y_list = [c[2] for c in chunk]
        plt.scatter(raw_x_list, raw_y_list, color="green", label="Raw data", marker='o',linewidth=4)

        fir_x_list = [z[0] for z in first_pad]
        fir_y_list = [z[1] for z in first_pad]
        # Line
        # plt.plot(new_x_list, new_y_list, color="red", label="New data", linewidth=2)
        # Scatter
        plt.scatter(fir_x_list, fir_y_list, color="red", label="First pad data", marker='x', linewidth=4)

        sec_x_list = [z[0] for z in sec_pad]
        sec_y_list = [z[1] for z in sec_pad]
        plt.scatter(sec_x_list, sec_y_list, color="red", label="Second pad data", marker='s', linewidth=1)
        plt.legend()  # 绘制图例
        plt.show()

    def serialize_2_excel(self, excel_filename):
        # Get the label list
        book = xlwt.Workbook(encoding='utf-8')
        sheet1 = book.add_sheet(sheetname='Sheet1')

        for i in range(len(self.label_list)):
            label = self.label_list[i]
            data_list = self.sec_pad_data_list[i]

            sheet1.write(0, 2 * i, label)
            for r_i, data in zip(range(1, len(data_list) + 1), data_list):
                sheet1.write(r_i, 2 * i, data[0])
                sheet1.write(r_i, 2 * i + 1, data[1])
        book.save(excel_filename)

if __name__ == '__main__':
    ldhd = low_dim_2_high_dim(file_path='../../../datasets/experiement_data/processed_dataset', \
                              excel_name='whole_dataset.xlsx', \
                              sheet_name='Sheet2_trim')
    # ldhd.plot(chunk_index=16)
    ldhd.serialize_2_excel(excel_filename='lai_80.xls')