import os
"""
本模块中的函数主要用于解析Lai的EIS原始数据
"""
def read_raw_lai_eis(folder):
    """
    function
        主要读取文件夹中的所有EIS实验文件
    :param
        folder
            default: prj\datasets\experiement_data\lai\eis\exp_data_txt
    :return:
        以文件名为key，value:{‘fre’:fre_list, 'z':z_list(complex)}的字典
    """
    raw_eis_dict = {}
    filenames = os.listdir(folder)
    for fn in filenames:
        key = fn.split('.')[0]
        file_path = os.path.join(folder, fn)
        fre_list = []
        z_list = []
        with open(file_path, 'r') as file:
            for line in file.readlines():
                line_str_list = line.strip().split('\t')
                fre = float(line_str_list[0])
                z = float(line_str_list[1]) + float(line_str_list[-1]) * 1j
                fre_list.append(fre)
                z_list.append(z)
        raw_eis_dict[key] = {'fre':fre_list, 'z':z_list}
    return raw_eis_dict

def normed_lai_eis(folder, sample_area = 1.01 * 1e-6):
    """
    function
        将raw-EIS数据 乘以 实验区域面积 得到 单位面积（cm * cm）上的阻抗数据
    :param
        folder:
        sample_area: 1.01 * 1e-6 cm^2
    :return:
    """
    raw_eis_dict = read_raw_lai_eis(folder)
    normed_eis_dict = {}
    for k in raw_eis_dict.keys():
        normed_eis_dict[k] = {'fre': raw_eis_dict, 'z':[z * sample_area for z in raw_eis_dict[k]['z']]}
    return normed_eis_dict
# normed_eis_dict = normed_lai_eis(folder='../../../../../datasets/experiement_data/laiZhaoGui/eis/exp_data_txt')