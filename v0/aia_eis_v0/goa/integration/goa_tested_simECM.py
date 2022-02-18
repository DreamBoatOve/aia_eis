import sys
sys.path.append('../../')

import os

from data_processor.GOA_simulation.GOA_ECMs_simulation import load_sim_ecm_para_config_dict
"""
Module Function
    本次（第二次）计算Chi-Square
        参照ZSimpWin中的最终定义
            Tech Note 37: Least Squares Fit Formulation
                3. Definition of the chi-squared
                    Eq 10 Modulus weighting
                    就是在第一次的结果上在除以 v，
                    v为系统的自由度，严格讲 v = N - M - 1，此处为了和ZSimpWIn一样，令 v = N - M
                        N 为测试的阻抗点数，如赖师兄N=30
                        M 为等效电路中需要拟合的参数数目，CPE元件包含两个参数Y和n
    第一此计算Chi-Square计算错误
        def cal_w(raw_Z):
            return 1 / (raw_Z.real ** 2 + raw_Z.imag ** 2)
        CHi-Square = sum([cal_w(rz) * ((rz.real - sz.real)**2 + (rz.imag - sz.imag)**2)
                                 for rz, sz in zip(z_raw_complex_list, z_sim_complex_list)])
                                 
Routine
    因为20种GOA已分别在9种ECM上按照第一种Chi-Square，CS-1，计算好，正确的Chi-Square，CS2，和CS1差别仅在于v，
    所以将之前的结果中的CS1 除以 v即为 CS2
    在原先文本的倒数第一列和第二列中间添加CS2
    simECM_res folder中是几个测试文本
"""
def get_V(ecm_num, file_path):
    ecm_para_config_dict = load_sim_ecm_para_config_dict(ecm_num, file_path)
    # N 为测试的阻抗点数，如赖师兄N=30
    N = len(ecm_para_config_dict['f'])

    # M 为等效电路中需要拟合的参数数目，CPE元件包含两个参数Y和n
    M = len(ecm_para_config_dict['para'])

    # V为系统的自由度，严格讲 v = N - M - 1，此处为了和ZSimpWIn一样，令 v = N - M
    v = N - M
    return v

def reCal_Chi_Square(res_fp, sim_EIS_fp):
    fn_list = os.listdir(res_fp)
    for i, fn in enumerate(fn_list):
        print('Processsing {0} file {1}'.format(i, fn))
        # 1- txt ?
        if fn.endswith('.txt'):
            ecm_num = None
            # 2- 'bb_bc' ?
            if fn.split('_')[0] == 'bb':
                # bb_bc_ecm8_07.txt
                ecm_num = int(fn.split('.')[0].split('_')[2].split('ecm')[1])
            else:
                # gwo_ecm5_42.txt
                ecm_num = int(fn.split('.')[0].split('_')[1].split('ecm')[1])
            v = get_V(ecm_num, sim_EIS_fp)

            new_str = ''
            with open(os.path.join(res_fp, fn), 'r+') as f1:
                """
                Old Line:
                Iteration,  para_list[float, ...],                                                  Chi-Square(错误版本),     Cumulated Running Time
                9984,       [0.02012484780316051,0.0001799896579117755, ..., 52972.38087848809],    4.433244077492469e-06,  124.94990979999966
                New Line:
                Iteration,  para_list[float, ...],                                                  Chi-Square(错误版本),       Chi-Square(Right Version),    Cumulated Running Time
                9984,       [0.02012484780316051,0.0001799896579117755, ..., 52972.38087848809],    4.433244077492469e-06,      4.433244077492469e-06 / v,  124.94990979999966
                """
                for line in f1.readlines():
                    # 1- Get CS1, Chi-Square(错误版本)
                    line_str_list = line.strip().split(',')
                    cs1 = float(line_str_list[-2])
                    # 2- Calculate CS2 = CS1 / v, CS2: Chi-Square(Right Version)
                    cs2 = cs1 / v
                    # 3- Create new line
                    line_str_list.insert(-1, str(cs2))
                    tmp_new_line = ','.join(line_str_list)+'\n'
                    new_str += tmp_new_line

                # Empty the old content
                # truncate() 方法用于截断文件，如果指定了可选参数 size，则表示截断文件为 size 个字符。
                # 如果没有指定 size，则从当前位置起截断；截断之后 size 后面的所有字符被删除
                f1.truncate(0)

            # Write new content into the same file
            with open(os.path.join(res_fp, fn), 'w') as f2:
                f2.write(new_str)
# reCal_Chi_Square(res_fp='./simECM_res', sim_EIS_fp='../../datasets/goa_datasets')