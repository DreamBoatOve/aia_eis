import os

def read_testSet(filepath, filename = './testSet.txt'):
    data_i_list = []
    data_j_list = []
    file_abs_path = os.path.join(filepath, filename)
    with open(file_abs_path, 'r') as file:
        for line in file.readlines():
            line_str_list = line.strip().split('\t')
            if line_str_list[-1] == '0':
                data_i_list.append([(float(line_str_list[0]), float(line_str_list[1]))])
            else:
                data_j_list.append([(float(line_str_list[0]), float(line_str_list[1]) )])
    return data_i_list, data_j_list

def read_testSet_with_label(filepath, filename = './testSet.txt'):
    labeled_data_i_list = []
    labeled_data_j_list = []
    file_abs_path = os.path.join(filepath, filename)
    with open(file_abs_path, 'r') as file:
        for line in file.readlines():
            line_str_list = line.strip().split('\t')
            if line_str_list[-1] == '0':
                labeled_data_i_list.append([0, [(float(line_str_list[0]), float(line_str_list[1]))]])
            else:
                labeled_data_j_list.append([1, [(float(line_str_list[0]), float(line_str_list[1]) )]])
    return labeled_data_i_list, labeled_data_j_list