import os

from utils.gui_utils.exp_data_judger_gui import exp_data_judger

def DTA_fn_judge(fn, required_str, fn_suffix='.DTA'):
    """
    Function
    :param
        required_str:
            文件名中必须有的一段字符串
            如必须要有‘eis’，大小写均可
                刘鹏的实验文件：‘EISPOT-R01-C01.DTA’
                文超的实验文件：‘lwc-345-0.1Meis.DTA’
        fn_suffix:
            文件名的后缀
    :return:
        符合要求 True
        不符合要求 False
    """
    # str.upper() / lower()
    required_str_list = [s.lower() for s in required_str]
    required_str_len = len(required_str_list)
    if fn.endswith(fn_suffix):
        fn_header = fn.strip().split(fn_suffix)[0]
        fn_header_str_list = [s.lower() for s in fn_header]
        fn_header_str_len = len(fn_header_str_list)

        for i in range(fn_header_str_len - required_str_len + 1):
            tmp_fn_list = fn_header_str_list[i : i + required_str_len]
            compare_bool_list = [a == b for a, b in zip(required_str_list, tmp_fn_list)]
            if False not in compare_bool_list:
                return True
        return False
# DTA_fn_judge(fn='EISPOT-R01-C01.DTA', required_str='EIS', fn_suffix='.DTA')
# DTA_fn_judge(fn='lwc-345-0.1Meis.DTA', required_str='eis', fn_suffix='.DTA')

def read_folder(folder_fp, required_str):
    fn_list = os.listdir(folder_fp)
    eis_fp_list = []
    for fn in fn_list:
        fp = os.path.join(folder_fp, fn)

        # a file
        if os.path.isfile(fp):
            if DTA_fn_judge(fn, required_str, fn_suffix='.DTA'):
                eis_fp_list.append(fp)

        # a folder
        else:
            tmp_eis_fp_list = read_folder(folder_fp=fp, required_str=required_str)
            eis_fp_list.extend(tmp_eis_fp_list)
    return eis_fp_list
# eis_fp_list = read_folder(folder_fp='E:\ms\liuPeng\电化学', required_str='eis')

"""
1-用户自身输入文件中包含的数据
2-根据用户输入选择读取文件的函数
3-最终返回结果为字典（需要包含各种属性）
"""
class DTA_reader:
    """
    Function
        DTA是Gamry实验设备对应的文件后缀，这个类专门用来读取Gamry
    """
    def __init__(self, file_path, data_type_selection=False):
        """
        :param
            file_path: 
                这里的文件路径要确保是直接能找到“单个实验文件”的相对路径
            data_type_selection: 
        """""
        if data_type_selection == False:
            # 0 == EIS， 1 == OCP + EIS，默认为1，即（OCP + EIS）
            self.data_type_int = 1
        else:
            self.data_type_int = 0
            # self.data_type_int = exp_data_judger().create_gui()
        self.file_path = file_path

    def fn_configer(self):
        pass

    def fn_judge(self):
        pass

    def eis_reader(self):
        """
        eis_dict 包含以下信息
            eis
                初始频率        freq_init       float
                    该行例句：FREQINIT	QUANT	1.00000E+005	Initial Freq. (Hz)
                终止频率        freq_final      float
                    该行例句：FREQFINAL	QUANT	1.00000E-002	Final Freq. (Hz)
                交流电压的振幅   ac_voltage (mV)      float
                    该行例句：VAC	QUANT	1.00000E+001	AC Voltage (mV rms)
                阻抗实部        z_real          list[float]
                    该行例句：	0	3	100078.1	[121.8645]	-158.9782	1	200.3123	-52.52809	-1.795593E-007	-0.2048679	8
                阻抗虚部        z_img           list[float]
                    该行例句：	0	3	100078.1	121.8645	[-158.9782]	1	200.3123	-52.52809	-1.795593E-007	-0.2048679	8
        """
        eis_dict = {}
        freq_init = 0.0
        freq_final = 0.0
        ac_voltage = 0.0
        fre_list = []
        z_real_list = []
        z_img_list = []
        # Error: UnicodeDecodeError: 'gbk/utf-8' codec can 't decode byte 0xb0 in position 1967: illegal multibyte sequence
        # Solution: 用rb二进制模式读取文件，在分别对每一行的二进制数据进行utf-8编码成字符串
        with open(self.file_path, 'rb') as file:
            line_num = 0
            ZCURVE_line_num = 0
            for line in file.readlines():
                # print('binary line', line)
                # line_utf_8 = line.decode('utf-8')
                try:
                    line_utf_8 = line.decode('utf-8')
                except UnicodeDecodeError:
                    print('line:', line_num, line, 'occured UnicodeDecodeError')
                    line_num += 1
                    continue
                # print('utf-8 line', line_utf_8)
                line_str_list = line_utf_8.strip().split('\t')
                if ZCURVE_line_num == 0:
                    if line_str_list[0] == 'FREQINIT':
                        freq_init = float(line_str_list[2])
                        # print('Testing: freq_init:',freq_init)
                        eis_dict['freq_init'] = freq_init
                        line_num += 1
                        continue
                    elif line_str_list[0] == 'FREQFINAL':
                        freq_final = float(line_str_list[2])
                        # print('Testing: freq_final:',freq_final)
                        eis_dict['freq_final'] = freq_final
                        line_num += 1
                        continue
                    elif line_str_list[0] == 'VAC':
                        ac_voltage = float(line_str_list[2])
                        # print('Testing: ac_voltage:', ac_voltage)
                        eis_dict['ac_voltage'] = ac_voltage
                        line_num += 1
                        continue
                    elif line_str_list[0] == 'ZCURVE':
                        ZCURVE_line_num = line_num
                        line_num += 1
                        continue
                    else:
                        line_num += 1
                        continue
                elif line_num > (ZCURVE_line_num + 2):
                    # print('Testing: line_str_list:', line_str_list)
                    fre_list.append(float(line_str_list[2]))
                    z_real_list.append(float(line_str_list[3]))
                    z_img_list.append(float(line_str_list[4]))
                    line_num += 1
                    continue
                else:
                    line_num += 1
                    continue

            eis_dict['f'] = fre_list
            eis_dict['z_real'] = z_real_list
            eis_dict['z_img'] = z_img_list
        return eis_dict

    def ocp_eis_reader(self):
        """
        ocp_eis_dict
        ocp_dict 包含以下信息
            ocp
                时间(s)           t               list[float]
                    该行例句：	0	[0.25]	1.83442E-001	1.83442E-001	5.87918E-004	...........
                电位(V)           vf              list[float]
                    该行例句：	1	0.5	[1.83435E-001]	1.83071E-001	5.93751E-004	...........
        eis_dict 包含以下信息
            eis
                初始频率        freq_init       float
                    该行例句：FREQINIT	QUANT	1.00000E+005	Initial Freq. (Hz)
                终止频率        freq_final      float
                    该行例句：FREQFINAL	QUANT	1.00000E-002	Final Freq. (Hz)
                交流电压的振幅   ac_voltage (mV)      float
                    该行例句：VAC	QUANT	1.00000E+001	AC Voltage (mV rms)
                阻抗实部        z_real          list[float]
                    该行例句：	0	3	100078.1	[121.8645]	-158.9782	1	200.3123	-52.52809	-1.795593E-007	-0.2048679	8
                阻抗虚部        z_img           list[float]
                    该行例句：	0	3	100078.1	121.8645	[-158.9782]	1	200.3123	-52.52809	-1.795593E-007	-0.2048679	8
        """
        ocp_eis_dict = {}

        ocp_dict = {}
        t_list = []
        vf_list = []

        eis_dict = {}
        z_real_list = []
        z_img_list = []

        with open(self.file_path, 'rb') as file:
            line_num = 0
            OCVCURVE_line_num = 0
            ZCURVE_line_num = 0
            for line in file.readlines():
                try:
                    line_utf_8 = line.decode('utf-8')
                except UnicodeDecodeError:
                    line_num += 1
                    continue
                line_str_list = line_utf_8.strip().split('\t')
                if (OCVCURVE_line_num == 0) & (ZCURVE_line_num == 0):
                    if line_str_list[0] == 'FREQINIT':
                        freq_init = float(line_str_list[2])
                        print('Testing: freq_init:',freq_init)
                        eis_dict['freq_init'] = freq_init
                        line_num += 1
                    elif line_str_list[0] == 'FREQFINAL':
                        freq_final = float(line_str_list[2])
                        print('Testing: freq_final:',freq_final)
                        eis_dict['freq_final'] = freq_final
                        line_num += 1
                    elif line_str_list[0] == 'VAC':
                        ac_voltage = float(line_str_list[2])
                        print('Testing: ac_voltage:', ac_voltage)
                        eis_dict['ac_voltage'] = ac_voltage
                        line_num += 1
                    elif line_str_list[0] == 'OCVCURVE':
                        OCVCURVE_line_num = line_num
                        line_num += 1
                    else:
                        line_num += 1
                elif (line_num > OCVCURVE_line_num + 2) & (ZCURVE_line_num == 0):
                    if line_str_list[0] == 'EOC':
                        ZCURVE_line_num = 1
                    else:
                        t_list.append(float(line_str_list[1]))
                        vf_list.append(float(line_str_list[2]))
                    line_num += 1
                elif (ZCURVE_line_num != 0) & (ZCURVE_line_num != 1) & (line_num > (ZCURVE_line_num + 2)):
                    z_real_list.append(float(line_str_list[3]))
                    z_img_list.append(float(line_str_list[4]))
                    line_num += 1
                else:
                    if line_str_list[0] == 'ZCURVE':
                        ZCURVE_line_num = line_num
                    line_num += 1
        ocp_dict['t'] = t_list
        ocp_dict['vf'] = vf_list

        eis_dict['z_real'] = z_real_list
        eis_dict['z_img'] = z_img_list

        ocp_eis_dict['ocp'] = ocp_dict
        ocp_eis_dict['eis'] = eis_dict
        return ocp_eis_dict

    def reader(self):
        if self.data_type_int == 0:
            return self.eis_reader()
        else:
            return self.ocp_eis_reader()

# eis_file_path = 'E:\WorkSpaceOfGit\distributed_parallel_fitting_circuit\datasets\eis_files_in_different_formats\eis.DTA'
# eis_dict = exp_data_reader(file_path=eis_file_path).eis_reader()
# eis_dict = exp_data_reader(file_path=eis_file_path, data_type_selection=True).reader()
# print(eis_dict)

# ocp_eis_file_path = 'E:\WorkSpaceOfGit\distributed_parallel_fitting_circuit\datasets\eis_files_in_different_formats\ocp_eis.DTA'
# ocp_eis_dict = exp_data_reader(file_path=ocp_eis_file_path).ocp_eis_reader()
# ocp_eis_dict = exp_data_reader(file_path=ocp_eis_file_path, data_type_selection=True).reader()
# print(ocp_eis_dict)