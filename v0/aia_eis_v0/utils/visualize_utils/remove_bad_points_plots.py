import math
import matplotlib.pyplot as plt
import os

from circuits.circuit_pack import RaQRbaQRb
from data_processor.GOA_preprocessor.goa_data_wrapper import load_lai_manual_fitting_res
from utils.visualize_utils.impedance_plots import nyquist_plot_1
from utils.file_utils.pickle_utils import pickle_file, load_pickle_file

"""
Module Function
    为了对比GOA在 赖的原始数据 和 删除坏点后的对应数据 上的拟合效果
    需要在论文中绘制一组图，三张,把图ab合并的话，太挤，看不清
        图a
            赖的raw-5-18，ECM2类型的原始数据
            局部放大在100mHz和159mHz处的两个坏点
        图b
            raw + 【Lai】 + GOA-1~5（ECM2拟合性能的Top5），Chi-Square
            局部放大在高频处，主要凸显GOA拟合混乱。
        图c
            raw + Lai + GOA-1~5（ECM2拟合性能的Top5），Chi-Square
            局部放大在高频处，主要凸显高频处密集的点拟合的也好。
        绘图数据
            dpfc_src\plugins_test\jupyter_code\dpfc_files\tests_10\pkls\5-18
                goa1_GWO_raw_dict.pkl
                    GWO在原始数据上的拟合结果
                goa1_GWO_re_dict.pkl
                    GWO在删除坏点后数据上的拟合结果
"""
lai_normed_dataset = load_pickle_file(fp='../../datasets/goa_datasets/normed',
                                      fn='2020_08_22_goa_lai_normed_dataset_pickle.file')
lai_manual_fit_res_dict = load_lai_manual_fitting_res(file_path='../../datasets/goa_datasets/Lai_manual_fitting_res',
                                                      file_names=['2020_07_22_lai_ecm2_fitting_res.CSV',
                                                                  '2020_07_22_lai_ecm9_fitting_res.CSV'],
                                                      mag_num=5)
# a图是原始EIS的Nyquist
def a_5_18_plot():
    # '5-18'
    one_normed_dataset = lai_normed_dataset[117]
    raw_z_list = one_normed_dataset['z_raw']

    img_dict = {'fname': 'a', 'dpi': 300}
    # img_dict = {'fname': '', 'fmt': , 'dpi': 300}
    nyquist_plot_1(img_dict, z_list=raw_z_list, x_lim=[0, 7], y_lim=[0, 7],
                   grid_flag=False, plot_label='Raw')
# a_5_18_plot()

def a_5_31_plot():
    # '5-31' ECM-2
    one_normed_dataset = lai_normed_dataset[124]
    raw_z_list = one_normed_dataset['z_raw']

    img_dict = {'fname': 'a', 'dpi': 300}
    # img_dict = {'fname': '', 'fmt': , 'dpi': 300}
    nyquist_plot_1(img_dict, z_list=raw_z_list, x_lim=[0, 1400], y_lim=[0, 1400],
                   grid_flag=False, plot_label='Raw')
    print('done')
# a_5_31_plot()

# a图是原始EIS的Nyquist局部放大
def a_5_18_enlarge_plot():
    # '5-18'
    one_normed_dataset = lai_normed_dataset[117]
    raw_z_list = one_normed_dataset['z_raw']

    img_dict = {'fname': 'a_enlarge', 'dpi': 300}
    # img_dict = {'fname': '', 'fmt': , 'dpi': 300}
    nyquist_plot_1(img_dict, z_list=raw_z_list, x_lim=[0.0075, 0.02], y_lim=[0, 0.0125],
                   grid_flag=False, plot_label='Raw')
# a_5_18_enlarge_plot()

def a_5_31_enlarge_plot():
    # '5-31' ECM-2
    one_normed_dataset = lai_normed_dataset[124]
    raw_z_list = one_normed_dataset['z_raw']

    img_dict = {'fname': 'a_enlarge', 'dpi': 300}
    # img_dict = {'fname': '', 'fmt': , 'dpi': 300}

    # '5-31' enlarge 【中】频部的两个异常点
    # nyquist_plot_1(img_dict, z_list=raw_z_list, x_lim=[-30, 30], y_lim=[0, 60],
    #                grid_flag=False, plot_label='Raw')

    # '5-31' enlarge 【高】频部的两个异常点
    nyquist_plot_1(img_dict, z_list=raw_z_list, x_lim=[0, 0.8], y_lim=[0, 0.8],
                   grid_flag=False, plot_label='Raw')
    print('done')
# a_5_31_enlarge_plot()

def unpack_complexZ(complex_z_list):
    z_real_list = [z.real for z in complex_z_list]
    z_imag_list = [z.imag for z in complex_z_list]
    z_inv_imag_list = [-z_imag for z_imag in z_imag_list]
    return z_real_list, z_inv_imag_list

# b图
def b_5_18_plot():
    fig = plt.figure()
    # 保留小数点后5位
    round_num = 5

    # Read Data
    # Raw
    one_normed_dataset = lai_normed_dataset[117]
    raw_z_list = one_normed_dataset['z_raw']
    raw_z_real_list, raw_z_inv_imag_list = unpack_complexZ(raw_z_list)
    # chr(967) == 'χ'
    raw_label = 'Raw'
    # raw_label = 'Raw,'+chr(967)+'\u00B2'+'='+str(round())
    plt.plot(raw_z_real_list, raw_z_inv_imag_list, 'o--', label='Raw')

    goa_res_fp = '../../plugins_test/jupyter_code/dpfc_files/tests_10/pkls/5-18'
    # goa-GSO-1
    goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GSO_raw_dict.pkl')
    goa_gso_1_z_list = goa_gso_1_dict['z']
    goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list = unpack_complexZ(goa_gso_1_z_list)
    goa_gso_1_label = 'GSO, '+chr(967)+'\u00B2'+'='+str(round(goa_gso_1_dict['chi_square'], round_num))
    # Modify fmt
    plt.plot(goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list, 'ko--', label=goa_gso_1_label)

    # goa-GWO-2
    goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GWO_raw_dict.pkl')
    goa_gwo_2_z_list = goa_gwo_2_dict['z']
    goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list = unpack_complexZ(goa_gwo_2_z_list)
    goa_gwo_2_label = 'GWO, '+chr(967)+'\u00B2'+'='+str(round(goa_gwo_2_dict['chi_square'], round_num))
    plt.plot(goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list, 'gs:', label=goa_gwo_2_label)

    # goa-WOA-3
    goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_WOA_raw_dict.pkl')
    goa_woa_3_z_list = goa_woa_3_dict['z']
    goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list = unpack_complexZ(goa_woa_3_z_list)
    goa_woa_3_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_woa_3_dict['chi_square'], round_num))
    plt.plot(goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list, 'gh:', label=goa_woa_3_label)

    # goa-DE-4
    goa_de_4_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_DE_raw_dict.pkl')
    goa_de_4_z_list = goa_de_4_dict['z']
    goa_de_4_z_real_list, goa_de_4_z_inv_imag_list = unpack_complexZ(goa_de_4_z_list)
    goa_de_4_label = 'DE, '+chr(967)+'\u00B2'+'='+str(round(goa_de_4_dict['chi_square'], round_num))
    plt.plot(goa_de_4_z_real_list, goa_de_4_z_inv_imag_list, 'bo-', label=goa_de_4_label)

    # goa-ABC-5
    goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_ABC_raw_dict.pkl')
    goa_abc_5_z_list = goa_abc_5_dict['z']
    goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list = unpack_complexZ(goa_abc_5_z_list)
    goa_abc_5_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_abc_5_dict['chi_square'], round_num))
    plt.plot(goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list, 'gv:', label=goa_abc_5_label)

    # Configure plot
    # x_lim, y_lim = [0, 7], [0, 7]
    # x_lim, y_lim = [-0.5, 10], [-0.5, 10]
    x_lim, y_lim = [-0.5, 7], [-0.5, 7]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    plt.gca().set_aspect("equal")

    plt.grid(False)
    plt.legend()
    # plt.show()

    fname = 'b.tif'
    dpi = 300

    plt.show()
    # plt.savefig(fname=fname, dpi=dpi, format='tif')
# b_5_18_plot()

def b_5_31_plot():
    # '5-31' ECM-2
    fig = plt.figure()
    # 保留小数点后5位
    round_num = 5

    # Read Data
    # Raw
    one_normed_dataset = lai_normed_dataset[124]
    raw_z_list = one_normed_dataset['z_raw']
    raw_z_real_list, raw_z_inv_imag_list = unpack_complexZ(raw_z_list)
    # chr(967) == 'χ'
    raw_label = 'Raw'
    # raw_label = 'Raw,'+chr(967)+'\u00B2'+'='+str(round())
    plt.plot(raw_z_real_list, raw_z_inv_imag_list, 'o--', label='Raw')

    goa_res_fp = '../../plugins_test/jupyter_code/dpfc_files/tests_10/pkls/5-31'
    # goa-GSO-1
    # goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GSO_raw_dict.pkl')
    goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa4_GSO_raw_dict.pkl')
    goa_gso_1_z_list = goa_gso_1_dict['z']
    goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list = unpack_complexZ(goa_gso_1_z_list)
    goa_gso_1_label = 'GSO, '+chr(967)+'\u00B2'+'='+str(round(goa_gso_1_dict['chi_square'], round_num))
    # Modify fmt
    plt.plot(goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list, 'ko--', label=goa_gso_1_label)

    # goa-GWO-2
    # goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GWO_raw_dict.pkl')
    goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa5_GWO_raw_dict.pkl')
    goa_gwo_2_z_list = goa_gwo_2_dict['z']
    goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list = unpack_complexZ(goa_gwo_2_z_list)
    goa_gwo_2_label = 'GWO, '+chr(967)+'\u00B2'+'='+str(round(goa_gwo_2_dict['chi_square'], round_num))
    plt.plot(goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list, 'gs:', label=goa_gwo_2_label)

    # goa-WOA-3
    # goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_WOA_raw_dict.pkl')
    goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa2_WOA_raw_dict.pkl')
    goa_woa_3_z_list = goa_woa_3_dict['z']
    goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list = unpack_complexZ(goa_woa_3_z_list)
    goa_woa_3_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_woa_3_dict['chi_square'], round_num))
    plt.plot(goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list, 'gh:', label=goa_woa_3_label)

    # goa-DE-4
    goa_de_4_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_DE_raw_dict.pkl')
    goa_de_4_z_list = goa_de_4_dict['z']
    goa_de_4_z_real_list, goa_de_4_z_inv_imag_list = unpack_complexZ(goa_de_4_z_list)
    goa_de_4_label = 'DE, '+chr(967)+'\u00B2'+'='+str(round(goa_de_4_dict['chi_square'], round_num))
    plt.plot(goa_de_4_z_real_list, goa_de_4_z_inv_imag_list, 'bo-', label=goa_de_4_label)

    # goa-ABC-5
    # goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_ABC_raw_dict.pkl')
    goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa3_ABC_raw_dict.pkl')
    goa_abc_5_z_list = goa_abc_5_dict['z']
    goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list = unpack_complexZ(goa_abc_5_z_list)
    goa_abc_5_label = 'ABC, '+chr(967)+'\u00B2'+'='+str(round(goa_abc_5_dict['chi_square'], round_num))
    plt.plot(goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list, 'gv:', label=goa_abc_5_label)

    # Configure plot
    x_lim, y_lim = [-30, 1520], [-30, 1520]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    plt.gca().set_aspect("equal")

    plt.grid(False)
    plt.legend()
    # plt.show()

    fname = 'b.tif'
    dpi = 300

    plt.show()
    # plt.savefig(fname=fname, dpi=dpi, format='tif')
# b_5_31_plot()

def b_5_18_enlarge_plot():
    fig = plt.figure()
    # 保留小数点后5位
    round_num = 5

    # Read Data
    # Raw
    one_normed_dataset = lai_normed_dataset[117]
    raw_z_list = one_normed_dataset['z_raw']
    raw_z_real_list, raw_z_inv_imag_list = unpack_complexZ(raw_z_list)
    # chr(967) == 'χ'
    raw_label = 'Raw'
    # raw_label = 'Raw,'+chr(967)+'\u00B2'+'='+str(round())
    plt.plot(raw_z_real_list, raw_z_inv_imag_list, 'o--', label='Raw')

    goa_res_fp = '../../plugins_test/jupyter_code/dpfc_files/tests_10/pkls/5-18'
    # goa-GSO-1
    goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GSO_raw_dict.pkl')
    goa_gso_1_z_list = goa_gso_1_dict['z']
    goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list = unpack_complexZ(goa_gso_1_z_list)
    goa_gso_1_label = 'GSO, '+chr(967)+'\u00B2'+'='+str(round(goa_gso_1_dict['chi_square'], round_num))
    # Modify fmt
    plt.plot(goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list, 'ko--', label=goa_gso_1_label)

    # goa-GWO-2
    goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GWO_raw_dict.pkl')
    goa_gwo_2_z_list = goa_gwo_2_dict['z']
    goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list = unpack_complexZ(goa_gwo_2_z_list)
    goa_gwo_2_label = 'GWO, '+chr(967)+'\u00B2'+'='+str(round(goa_gwo_2_dict['chi_square'], round_num))
    plt.plot(goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list, 'gs:', label=goa_gwo_2_label)

    # goa-WOA-3
    goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_WOA_raw_dict.pkl')
    goa_woa_3_z_list = goa_woa_3_dict['z']
    goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list = unpack_complexZ(goa_woa_3_z_list)
    goa_woa_3_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_woa_3_dict['chi_square'], round_num))
    plt.plot(goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list, 'gh:', label=goa_woa_3_label)

    # goa-DE-4
    goa_de_4_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_DE_raw_dict.pkl')
    goa_de_4_z_list = goa_de_4_dict['z']
    goa_de_4_z_real_list, goa_de_4_z_inv_imag_list = unpack_complexZ(goa_de_4_z_list)
    goa_de_4_label = 'DE, '+chr(967)+'\u00B2'+'='+str(round(goa_de_4_dict['chi_square'], round_num))
    plt.plot(goa_de_4_z_real_list, goa_de_4_z_inv_imag_list, 'bo-', label=goa_de_4_label)

    # goa-ABC-5
    goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_ABC_raw_dict.pkl')
    goa_abc_5_z_list = goa_abc_5_dict['z']
    goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list = unpack_complexZ(goa_abc_5_z_list)
    goa_abc_5_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_abc_5_dict['chi_square'], round_num))
    plt.plot(goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list, 'gv:', label=goa_abc_5_label)

    # Configure plot
    # x_lim, y_lim = [0, 0.03], [0, 0.03]
    x_lim, y_lim = [0.0075, 0.03], [-0.0025, 0.02]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    plt.gca().set_aspect("equal")

    plt.grid(False)
    # plt.legend()
    # plt.show()

    fname = 'b_enlarge.tif'
    dpi = 300

    # plt.show()
    plt.savefig(fname=fname, dpi=dpi, format='tif')
# b_5_18_enlarge_plot()

def read_removed_eis_file(fn, fp='../../plugins_test/jupyter_code/dpfc_files/tests_10/files'):
    fre_list = []
    normed_z_list = []
    area = 1.01 * 1e-6
    with open(os.path.join(fp, fn), 'r', encoding='utf-8') as f:
        for line in f.readlines()[3:]:
            fre, zr, zi = [float(i) for i in line.strip().split(',')]
            normed_z = zr * area + 1j * zi * area
            fre_list.append(fre)
            normed_z_list.append(normed_z)
    return fre_list, normed_z_list

def get_ECM_para(para_list):
    # 这个函数适用于 ECM-2和ECM-9
    R0 = para_list[0]
    Q0_pair = para_list[1:3]
    R1 = para_list[3]
    Q1_pair = para_list[4:6]
    R2 = para_list[-1]
    return R0, Q0_pair, R1, Q1_pair, R2

def c_5_18_plot():
    fig = plt.figure()
    # 保留小数点后5位
    round_num = 5

    # Read Data
    # Raw-Re
    raw_re_fre_list, raw_re_normed_z_list = read_removed_eis_file(fn='eis-5-18-removed-Fre-Z.csv')
    raw_re_z_real_list, raw_re_z_inv_imag_list = unpack_complexZ(raw_re_normed_z_list)
    # chr(967) == 'χ'
    raw_label = 'Raw (removed)'
    # raw_label = 'Raw,'+chr(967)+'\u00B2'+'='+str(round())
    plt.plot(raw_re_z_real_list, raw_re_z_inv_imag_list, 'o--', label=raw_label)

    # Lai
    lai_data_5_18_info_dict = lai_manual_fit_res_dict['5-18']
    lai_5_18_para_list = lai_data_5_18_info_dict['para']
    lai_R0, lai_Q0_pair, lai_R1, lai_Q1_pair, lai_R2 = get_ECM_para(lai_5_18_para_list)
    w_list = [2 * math.pi * f for f in raw_re_fre_list]
    lai_z_list = [RaQRbaQRb(w, lai_R0, lai_Q0_pair, lai_R1, lai_Q1_pair, lai_R2) for w in w_list]
    lai_re_z_real_list, lai_re_z_inv_imag_list = unpack_complexZ(lai_z_list)
    lai_label = 'ZSimpWin, '+chr(967)+'\u00B2'+'='+str(round(lai_data_5_18_info_dict['chi_square'], round_num))
    plt.plot(lai_re_z_real_list, lai_re_z_inv_imag_list, 'm+-', label=lai_label)

    goa_res_fp = '../../plugins_test/jupyter_code/dpfc_files/tests_10/pkls/5-18'
    # goa-GSO-1
    goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GSO_re_dict.pkl')
    goa_gso_1_z_list = goa_gso_1_dict['z']
    goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list = unpack_complexZ(goa_gso_1_z_list)
    goa_gso_1_label = 'GSO, '+chr(967)+'\u00B2'+'='+str(round(goa_gso_1_dict['chi_square'], round_num))
    # Modify fmt
    plt.plot(goa_gso_1_z_real_list[:-2], goa_gso_1_z_inv_imag_list[:-2], 'ko--', label=goa_gso_1_label)

    # goa-GWO-2
    goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GWO_re_dict.pkl')
    goa_gwo_2_z_list = goa_gwo_2_dict['z']
    goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list = unpack_complexZ(goa_gwo_2_z_list)
    goa_gwo_2_label = 'GWO, '+chr(967)+'\u00B2'+'='+str(round(goa_gwo_2_dict['chi_square'], round_num))
    plt.plot(goa_gwo_2_z_real_list[:-2], goa_gwo_2_z_inv_imag_list[:-2], 'gs:', label=goa_gwo_2_label)

    # goa-WOA-3
    goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_WOA_re_dict.pkl')
    goa_woa_3_z_list = goa_woa_3_dict['z']
    goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list = unpack_complexZ(goa_woa_3_z_list)
    goa_woa_3_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_woa_3_dict['chi_square'], round_num))
    plt.plot(goa_woa_3_z_real_list[:-2], goa_woa_3_z_inv_imag_list[:-2], 'gh:', label=goa_woa_3_label)

    # goa-DE-4
    goa_de_4_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_DE_re_dict.pkl')
    goa_de_4_z_list = goa_de_4_dict['z']
    goa_de_4_z_real_list, goa_de_4_z_inv_imag_list = unpack_complexZ(goa_de_4_z_list)
    goa_de_4_label = 'DE, '+chr(967)+'\u00B2'+'='+str(round(goa_de_4_dict['chi_square'], round_num))
    plt.plot(goa_de_4_z_real_list[:-2], goa_de_4_z_inv_imag_list[:-2], 'bo-', label=goa_de_4_label)

    # goa-ABC-5
    goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_ABC_re_dict.pkl')
    goa_abc_5_z_list = goa_abc_5_dict['z']
    goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list = unpack_complexZ(goa_abc_5_z_list)
    goa_abc_5_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_abc_5_dict['chi_square'], round_num))
    plt.plot(goa_abc_5_z_real_list[:-2], goa_abc_5_z_inv_imag_list[:-2], 'gv:', label=goa_abc_5_label)

    # Configure plot
    x_lim, y_lim = [0, 7], [0, 7]
    # x_lim, y_lim = [-0.5, 10], [-0.5, 10]
    # x_lim, y_lim = [-0.5, 7], [-0.5, 7]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    plt.gca().set_aspect("equal")

    plt.grid(False)
    plt.legend()

    fname = 'c.tif'
    dpi = 300

    # plt.show()
    plt.savefig(fname=fname, dpi=dpi, format='tif')
# c_5_18_plot()

def c_5_31_plot():
    fig = plt.figure()
    # 保留小数点后5位
    round_num = 6

    # Read Data
    # Raw-Re
    raw_re_fre_list, raw_re_normed_z_list = read_removed_eis_file(fn='eis-5-31-removed-Fre-Z.csv')
    raw_re_z_real_list, raw_re_z_inv_imag_list = unpack_complexZ(raw_re_normed_z_list)
    # chr(967) == 'χ'
    raw_label = 'Raw (removed)'
    # raw_label = 'Raw,'+chr(967)+'\u00B2'+'='+str(round())
    plt.plot(raw_re_z_real_list, raw_re_z_inv_imag_list, 'o--', label=raw_label)

    # Lai
    lai_data_5_31_info_dict = lai_manual_fit_res_dict['5-31']
    lai_5_31_para_list = lai_data_5_31_info_dict['para']
    lai_R0, lai_Q0_pair, lai_R1, lai_Q1_pair, lai_R2 = get_ECM_para(lai_5_31_para_list)
    w_list = [2 * math.pi * f for f in raw_re_fre_list]
    lai_z_list = [RaQRbaQRb(w, lai_R0, lai_Q0_pair, lai_R1, lai_Q1_pair, lai_R2) for w in w_list]
    lai_re_z_real_list, lai_re_z_inv_imag_list = unpack_complexZ(lai_z_list)
    lai_label = 'ZSimpWin, '+chr(967)+'\u00B2'+'='+str(round(lai_data_5_31_info_dict['chi_square'], round_num))
    plt.plot(lai_re_z_real_list, lai_re_z_inv_imag_list, 'm+-', label=lai_label)

    goa_res_fp = '../../plugins_test/jupyter_code/dpfc_files/tests_10/pkls/5-31'
    # goa-GSO-1
    goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa4_GSO_re_dict.pkl')
    goa_gso_1_z_list = goa_gso_1_dict['z']
    goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list = unpack_complexZ(goa_gso_1_z_list)
    goa_gso_1_label = 'GSO, '+chr(967)+'\u00B2'+'='+str(round(goa_gso_1_dict['chi_square'], round_num))
    # Modify fmt
    plt.plot(goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list, 'ko--', label=goa_gso_1_label)

    # goa-GWO-2
    goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa5_GWO_re_dict.pkl')
    goa_gwo_2_z_list = goa_gwo_2_dict['z']
    goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list = unpack_complexZ(goa_gwo_2_z_list)
    goa_gwo_2_label = 'GWO, '+chr(967)+'\u00B2'+'='+str(round(goa_gwo_2_dict['chi_square'], round_num))
    plt.plot(goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list, 'gs:', label=goa_gwo_2_label)

    # goa-WOA-3
    goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa2_WOA_re_dict.pkl')
    goa_woa_3_z_list = goa_woa_3_dict['z']
    goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list = unpack_complexZ(goa_woa_3_z_list)
    goa_woa_3_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_woa_3_dict['chi_square'], round_num))
    plt.plot(goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list, 'gh:', label=goa_woa_3_label)

    # goa-DE-4
    goa_de_4_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_DE_re_dict.pkl')
    goa_de_4_z_list = goa_de_4_dict['z']
    goa_de_4_z_real_list, goa_de_4_z_inv_imag_list = unpack_complexZ(goa_de_4_z_list)
    goa_de_4_label = 'DE, '+chr(967)+'\u00B2'+'='+str(round(goa_de_4_dict['chi_square'], round_num))
    plt.plot(goa_de_4_z_real_list, goa_de_4_z_inv_imag_list, 'bo-', label=goa_de_4_label)

    # goa-ABC-5
    goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa3_ABC_re_dict.pkl')
    goa_abc_5_z_list = goa_abc_5_dict['z']
    goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list = unpack_complexZ(goa_abc_5_z_list)
    goa_abc_5_label = 'ABC, '+chr(967)+'\u00B2'+'='+str(round(goa_abc_5_dict['chi_square'], round_num))
    plt.plot(goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list, 'gv:', label=goa_abc_5_label)

    # Configure plot
    x_lim, y_lim = [-30, 1700], [-30, 1700]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    plt.gca().set_aspect("equal")

    plt.grid(False)
    plt.legend()

    fname = 'c.tif'
    dpi = 300

    plt.show()
    # plt.savefig(fname=fname, dpi=dpi, format='tif')
c_5_31_plot()

def c_5_18_enlarge_plot():
    fig = plt.figure()
    # 保留小数点后5位
    round_num = 5

    # Read Data
    # Raw-Re
    raw_re_fre_list, raw_re_normed_z_list = read_removed_eis_file(fn='eis-5-18-removed-Fre-Z.csv')
    raw_re_z_real_list, raw_re_z_inv_imag_list = unpack_complexZ(raw_re_normed_z_list)
    # chr(967) == 'χ'
    raw_label = 'Raw (removed)'
    # raw_label = 'Raw,'+chr(967)+'\u00B2'+'='+str(round())
    plt.plot(raw_re_z_real_list, raw_re_z_inv_imag_list, 'o--', label=raw_label)

    # Lai
    lai_data_5_18_info_dict = lai_manual_fit_res_dict['5-18']
    lai_5_18_para_list = lai_data_5_18_info_dict['para']
    lai_R0, lai_Q0_pair, lai_R1, lai_Q1_pair, lai_R2 = get_ECM_para(lai_5_18_para_list)
    w_list = [2 * math.pi * f for f in raw_re_fre_list]
    lai_z_list = [RaQRbaQRb(w, lai_R0, lai_Q0_pair, lai_R1, lai_Q1_pair, lai_R2) for w in w_list]
    lai_re_z_real_list, lai_re_z_inv_imag_list = unpack_complexZ(lai_z_list)
    lai_label = 'ZSimpWin, '+chr(967)+'\u00B2'+'='+str(round(lai_data_5_18_info_dict['chi_square'], round_num))
    plt.plot(lai_re_z_real_list, lai_re_z_inv_imag_list, 'm+-', label=lai_label)

    goa_res_fp = '../../plugins_test/jupyter_code/dpfc_files/tests_10/pkls/5-18'
    # goa-GSO-1
    goa_gso_1_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GSO_re_dict.pkl')
    goa_gso_1_z_list = goa_gso_1_dict['z']
    goa_gso_1_z_real_list, goa_gso_1_z_inv_imag_list = unpack_complexZ(goa_gso_1_z_list)
    goa_gso_1_label = 'GSO, '+chr(967)+'\u00B2'+'='+str(round(goa_gso_1_dict['chi_square'], round_num))
    # Modify fmt
    plt.plot(goa_gso_1_z_real_list[:-2], goa_gso_1_z_inv_imag_list[:-2], 'ko--', label=goa_gso_1_label)

    # goa-GWO-2
    goa_gwo_2_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_GWO_re_dict.pkl')
    goa_gwo_2_z_list = goa_gwo_2_dict['z']
    goa_gwo_2_z_real_list, goa_gwo_2_z_inv_imag_list = unpack_complexZ(goa_gwo_2_z_list)
    goa_gwo_2_label = 'GWO, '+chr(967)+'\u00B2'+'='+str(round(goa_gwo_2_dict['chi_square'], round_num))
    plt.plot(goa_gwo_2_z_real_list[:-2], goa_gwo_2_z_inv_imag_list[:-2], 'gs:', label=goa_gwo_2_label)

    # goa-WOA-3
    goa_woa_3_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_WOA_re_dict.pkl')
    goa_woa_3_z_list = goa_woa_3_dict['z']
    goa_woa_3_z_real_list, goa_woa_3_z_inv_imag_list = unpack_complexZ(goa_woa_3_z_list)
    goa_woa_3_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_woa_3_dict['chi_square'], round_num))
    plt.plot(goa_woa_3_z_real_list[:-2], goa_woa_3_z_inv_imag_list[:-2], 'gh:', label=goa_woa_3_label)

    # goa-DE-4
    goa_de_4_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_DE_re_dict.pkl')
    goa_de_4_z_list = goa_de_4_dict['z']
    goa_de_4_z_real_list, goa_de_4_z_inv_imag_list = unpack_complexZ(goa_de_4_z_list)
    goa_de_4_label = 'DE, '+chr(967)+'\u00B2'+'='+str(round(goa_de_4_dict['chi_square'], round_num))
    plt.plot(goa_de_4_z_real_list[:-2], goa_de_4_z_inv_imag_list[:-2], 'bo-', label=goa_de_4_label)

    # goa-ABC-5
    goa_abc_5_dict = load_pickle_file(fp=goa_res_fp, fn='goa1_ABC_re_dict.pkl')
    goa_abc_5_z_list = goa_abc_5_dict['z']
    goa_abc_5_z_real_list, goa_abc_5_z_inv_imag_list = unpack_complexZ(goa_abc_5_z_list)
    goa_abc_5_label = 'WOA, '+chr(967)+'\u00B2'+'='+str(round(goa_abc_5_dict['chi_square'], round_num))
    plt.plot(goa_abc_5_z_real_list[:-2], goa_abc_5_z_inv_imag_list[:-2], 'gv:', label=goa_abc_5_label)

    # Configure plot
    x_lim, y_lim = [0.005, 0.02], [0, 0.015]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    plt.gca().set_aspect("equal")

    plt.grid(False)
    # plt.legend()

    fname = 'c_enlarge_.tif'
    dpi = 300

    # plt.show()
    plt.savefig(fname=fname, dpi=dpi, format='tif')
# c_5_18_enlarge_plot()