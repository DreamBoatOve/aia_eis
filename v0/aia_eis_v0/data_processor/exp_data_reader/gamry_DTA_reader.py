import os

"""
Module function
    Read Lai's EIS experiment DTA files
"""

def target_file_seq_list_generator():
    target_file_dict = {}

    # Select useful EIS data from 2020-01-04-阻抗类型整理200103.xlsx
    # seq_1_set = set(range(1, 34)) - set([5, 13, 15, 17, 28])
    # seq_2_set = set(range(1, 35)) - set([14, 18, 21, 22, 25, 27])
    # seq_3_set = set(range(1, 35)) - set([3, 4, 16, 18, 28])
    # seq_4_set = set(range(1, 35)) - set([13, 15, 16, 17, 18, 20, 23, 24, 26, 27, 30, 32, 33])
    # seq_5_set = set(range(1, 34)) - set([10, 13, 14, 20, 21, 23, 25, 30])

    # Select useful EIS data from 2020-07-22-阻抗类型整理2006.xlsx, there are some modifications in raw EIS data,
    # Details of modification is recorded in 2020-07-22-阻抗类型整理2006.xlsx-Sheet 'revision'
    # 27 + 28 + 28 + 21 + 23 ==> 127 files in total
    seq_1_set = set(range(1, 34)) - set([5, 13, 15, 17, 23, 28]) # remove file 1-23 ==> 27 files
    seq_2_set = set(range(1, 35)) - set([14, 18, 21, 22, 25, 27]) # ==> 28 files
    seq_3_set = set(range(1, 35)) - set([2, 3, 4, 16, 18, 28]) # remove file 3-2 ==> 28 files
    seq_4_set = set(range(1, 35)) - set([13, 15, 16, 17, 18, 20, 23, 24, 26, 27, 30, 32, 33]) # ==> 21 files
    seq_5_set = set(range(1, 34)) - set([10, 13, 14, 15, 20, 21, 23, 24, 25, 30]) # remove file 5-15, 5-24 ==> 23 files
    # print(len(seq_1_set))
    # print(len(seq_2_set))
    # print(len(seq_3_set))
    # print(len(seq_4_set))
    # print(len(seq_5_set))

    seq_set_list = [seq_1_set, seq_2_set, seq_3_set, seq_4_set, seq_5_set]
    for s, seq in zip([str(a) for a in range(1, 6)], seq_set_list):
        target_file_dict[s] = list(seq)
    return target_file_dict

def decimal_converter(decimal_str_in_file):
    # decimal_str_in_file = '5.00000E+000' or '1.00000E+005' or '1.00000E-001' or '3.00000E+002'
    # 这种10进制的表示形式，用int函数无法直接转换
    decimal_str_list = decimal_str_in_file.split('E')
    integer_str = decimal_str_list[0]
    decimal_str = decimal_str_list[1]
    return int(float(integer_str)) * pow(10, int(decimal_str))

def gamry_DTA_OCP_EIS_parser(file_path):
    data_dict = {}
    # 将文件读成二进制
    # with open(file_path, 'rb') as file:

    # 对文件用【UTF-8】格式读取
    # with open(file_path, 'r', encoding='GBK') as file:

    # 对文件用【UTF-8】格式读取
    # with open(file_path, 'r', encoding='UTF-8') as file:
    # UnicodeDecodeError: 'utf-8' codec can't decode byte 0xb0 in position 3682: invalid start byte

    # 对文件用【ascii】格式读取
    # with open(file_path, 'r', encoding='ascii') as file:
    # UnicodeDecodeError: 'ascii' codec can't decode byte 0xb0 in position 3682: ordinal not in range(128)

    # with open(file_path, 'r', encoding='gb18030') as file:
    with open(file_path, 'r', encoding='UTF-8', errors='ignore') as file:
        line_num = 0
        fre_list = []
        zreal_list = []
        zimg_list = []
        z_pointer = 1000000
        for line in file.readlines():
            # print(line)
            line_str_list = line.strip().split('\t')
            # Get the 'points per decade'
            if line_str_list[0] == 'PTSPERDEC':
                data_dict['points_decade'] = int(float(line_str_list[2]))
                continue
            # Get AC Amplitude
            if line_str_list[0] == 'VAC':
                data_dict['AC'] = float(line_str_list[2])
                continue
            # Get Frequency Zreal Zimg
            if line_str_list[0] == 'ZCURVE':
                z_pointer = line_num + 3
            if line_num >= z_pointer:
                try:
                    fre_list.append(float(line_str_list[2]))
                    zreal_list.append(float(line_str_list[3]))
                    zimg_list.append(float(line_str_list[4]))
                except ValueError as e:
                    print(e)
                    print('The error is in file', file_path)
            line_num += 1
        data_dict['frequency'] = fre_list
        data_dict['zreal'] = zreal_list
        data_dict['zimg'] = zimg_list
    return data_dict

class lai_eis_reader:
    def __init__(self, parent_folder):
        self.parent_folder = parent_folder

    def generate_txt(self):
        target_file_dict = target_file_seq_list_generator()
        for sub_folder_str in [str(a) for a in range(1, 6)]:
            folder = self.parent_folder + '/' + sub_folder_str
            target_seq_list = target_file_dict[sub_folder_str]
            for target_seq in target_seq_list:
                filename_head = '20170415-EIS-'
                filename_tail = sub_folder_str + '-' + str(target_seq) + '.DTA'
                filename = filename_head + filename_tail
                file_path = os.path.join(folder, filename)

                data_dict = gamry_DTA_OCP_EIS_parser(file_path)
                frequency_list = data_dict['frequency']
                zreal_list = data_dict['zreal']
                zimg_list = data_dict['zimg']

                output_filename = sub_folder_str + '-' + str(target_seq) + '.txt'
                output_str = ''
                for f, zr, zi in zip(frequency_list, zreal_list, zimg_list):
                    output_str += str(f)+'\t'+str(zr)+'\t'+str(zi)+'\n'
                with open(output_filename, 'w') as file:
                    file.write(output_str)

if __name__ == '__main__':
    # ---------- Test FUNCTION target_file_seq_list_generator ----------
    # target_file_dict = target_file_seq_list_generator()
    # print(target_file_dict)
    # ---------- Test FUNCTION target_file_seq_list_generator ----------

    # ---------- Test FUNCTION decimal_converter(decimal_str_in_file) ----------
    # num = decimal_converter(decimal_str_in_file='5.00000E+002')
    # print(num)
    # ---------- Test FUNCTION decimal_converter(decimal_str_in_file) ----------

    # ---------- Test FUNCTION gamry_DTA_OCP_EIS_parser(file_path) ----------
    # data_dict = gamry_DTA_OCP_EIS_parser(file_path='./20170415-EIS-1-1.DTA')
    # print(data_dict)
    # ---------- Test FUNCTION gamry_DTA_OCP_EIS_parser(file_path) ----------

    # ---------- Test CLASS lai_eis_reader ----------
    lai_eis_reader(parent_folder='../../../datasets/experiement_data/laiZhaoGui/eis').generate_txt()
    # generated files are stored at dpfc\datasets\experiement_data\laiZhaoGui\eis\exp_data_txt
    # ---------- Test CLASS lai_eis_reader ----------