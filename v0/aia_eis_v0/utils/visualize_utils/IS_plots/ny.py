import math
import matplotlib.pyplot as plt

def nyquist_multiPlots_1(z_pack_list, x_lim=[0, 50000], y_lim=[0, 50000], img_dict=None, grid_flag=False,
                         plot_label_list=None):
    """
    Function
        1-同时画出多条IS，起到对比作用
        2-根据索引，标出人为扰动的点
    """
    marker_list = ['bo-', 'g^-', 'rv-', 'c.-', 'm<-']

    fig, ax = plt.subplots()
    for i, z_list in enumerate(z_pack_list):
        z_real_list = [z.real for z in z_list]
        z_imag_list = [z.imag for z in z_list]
        z_inv_imag_list = [-z_imag for z_imag in z_imag_list]

        fmt = marker_list[i]
        if plot_label_list is not None:
            ax.plot(z_real_list, z_inv_imag_list, fmt, label=plot_label_list[i])
        else:
            ax.plot(z_real_list, z_inv_imag_list, fmt)
    #         z = z_list[i]
    #         if i == 0: # index = 0 --> Ideal IS:
    #             ax.annotate(text='Real:({0}, {1})'.format(z.real,-z.imag), xy=(z.real,-z.imag))
    #         else:
    #             ax.annotate(text='Disturb:({0}, {1})'.format(z.real,-z.imag), xy=(z.real,-z.imag), textcoords='offset points', fontsize=16,
    #                         arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))
    # r'$xxxx$'
    # xy=蓝色点位置
    # xytext：描述框相对xy位置
    # textcoords='offset points'，以xy为原点偏移xytext
    # arrowprops = 画弧线箭头，'---->', rad=.2-->0.2弧度
    # plt.annotate(r'$2x+1=%s$'%y0, xy=(x0,y0), xytext=(+30,-30), textcoords='offset points', fontsize=16,
    #              arrowprops=dict(arrowstyle='->',connectionstyle='arc3,rad=.2'))

    #         if plot_label_list is not None:
    #             ax.plot(z_real_list, z_inv_imag_list, fmt=marker_list[i], label=plot_label_list[i])
    #         else:
    #             ax.plot(z_real_list, z_inv_imag_list, fmt=marker_list[i])

    ax.set_xlim(x_lim)
    ax.set_ylim(y_lim)
    ax.set_xlabel('Z$_{real}$ [$\Omega$]')  # 设置x轴名称 x label
    ax.set_ylabel('-Z$_{imag}$ [$\Omega$]')  # 设置y轴名称 y label
    ax.set_aspect("equal")

    plt.legend()
    plt.show()


def nyquist_multiPlots_0(z_pack_list, x_lim=[0, 50000], y_lim=[0, 50000], img_dict=None, grid_flag=False,
                         plot_label_list=None):
    """
    Function:
        This function is the same as nyquist_plot, but is used for Fig.1
        1- Mark ECM No. on the up right corner
        2- Save plots as the highest quality
            Python目前可以生成的图的格式是：eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            **The output formats available depend on the backend being used.
            Choose tif format

            savefig(fname, dpi=None, facecolor=’w’, edgecolor=’w’,
                orientation=’portrait’, papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
                fname:
                    stored image's name, like qwe.png, asd.tif
                dpi:
                    resolution of img
                bbox_inches:
                    'tight' 将图表多余的空白区域裁减掉。如果要保留图表周围多余的空白区域，可省略这个实参
    :param
        z_list:
        x_lim:
        y_lim:
        img_dict:
            {
                'fname': str, the name of image, like abc.tif
                'fmt': 颜色+线性+marker
                'dpi': int
            }
        grid_flag:
        plot_label:
    :return:
    """
    marker_list = ['o--', '^--', 'v--']
    for i, z_list in enumerate(z_pack_list):
        z_real_list = [z.real for z in z_list]
        z_imag_list = [z.imag for z in z_list]
        z_inv_imag_list = [-z_imag for z_imag in z_imag_list]

        # fig = plt.figure()
        # if plot_label_list is not None:
        #     plt.plot(z_real_list, z_inv_imag_list, marker_list, label=plot_label_list[i])
        # else:
        #     plt.plot(z_real_list, z_inv_imag_list, marker_list)

        fig, ax = plt.subplots()
        if plot_label_list is not None:
            ax.plot(z_real_list, z_inv_imag_list, marker_list, label=plot_label_list[i])
        else:
            ax.plot(z_real_list, z_inv_imag_list, marker_list)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    # plt.axis('equal')
    # plt.axis('scaled')
    # plt.set_aspect('equal')
    plt.gca().set_aspect("equal")

    plt.grid(grid_flag)
    plt.legend()
    # plt.show()

    if img_dict is not None:
        fname = img_dict['fname']
        dpi = img_dict['dpi']
    # plt.savefig(fname=fname, dpi=dpi, format='tiff')
    plt.show()


def nyquist_plot(z_list, grid_flag=False, fig_title=''):
    """
    :param
        z_list:
        grid_flag:
        plot_label:
    :return:
    """
    z_real_list = [z.real for z in z_list]
    z_imag_list = [z.imag for z in z_list]
    z_inv_imag_list = [-z_imag for z_imag in z_imag_list]

    fig = plt.figure()
    plt.plot(z_real_list, z_inv_imag_list, 'o--', label='Nyquist')

    # ----------- 之前手动设定，x轴和y轴的数值范围 -----------
    # plt.xlim(x_lim)
    # plt.ylim(y_lim)
    # ----------- 之前手动设定，x轴和y轴的数值范围 -----------

    # ----------- 自动设定x轴和y轴的数值范围 -----------
    x_min, x_max = min(z_real_list), max(z_real_list)
    x_range = x_max - x_min
    y_min, y_max = min(z_inv_imag_list), max(z_inv_imag_list)
    y_range = y_max - y_min
    max_range = max(x_range, y_range)

    x_lim = [x_min - max_range*0.05, x_min + max_range*1.05]
    y_lim = [y_min - max_range*0.05, y_min + max_range*1.05]
    plt.xlim(x_lim)
    plt.ylim(y_lim)
    # ----------- 自动设定x轴和y轴的数值范围 -----------

    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    # 使x/y 轴 单位长度 表示相同的数值
    plt.gca().set_aspect("equal")

    plt.grid(grid_flag)
    plt.legend()
    plt.title(fig_title)
    plt.show()


def nyquist_plot_1(z_list, x_lim=[0, 50000], y_lim=[0, 50000], img_dict=None,
                   grid_flag=False, plot_label=''):
    """
    Function:
        This function is the same as nyquist_plot, but is used for Fig.1
        1- Mark ECM No. on the up right corner
        2- Save plots as the highest quality
            Python目前可以生成的图的格式是：eps, jpeg, jpg, pdf, pgf, png, ps, raw, rgba, svg, svgz, tif, tiff
            **The output formats available depend on the backend being used.
            Choose tif format

            savefig(fname, dpi=None, facecolor=’w’, edgecolor=’w’,
                orientation=’portrait’, papertype=None, format=None,
                transparent=False, bbox_inches=None, pad_inches=0.1,
                frameon=None)
                fname:
                    stored image's name, like qwe.png, asd.tif
                dpi:
                    resolution of img
                bbox_inches:
                    'tight' 将图表多余的空白区域裁减掉。如果要保留图表周围多余的空白区域，可省略这个实参
    :param
        z_list:
        x_lim:
        y_lim:
        img_dict:
            {
                'fname': str, the name of image, like abc.tif
                'fmt': 颜色+线性+marker
                'dpi': int
            }
        grid_flag:
        plot_label:
    :return:
    """
    z_real_list = [z.real for z in z_list]
    z_imag_list = [z.imag for z in z_list]
    z_inv_imag_list = [-z_imag for z_imag in z_imag_list]

    fig = plt.figure()
    plt.plot(z_real_list, z_inv_imag_list, 'o--', label=plot_label)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')

    # plt.axis('equal')
    # plt.axis('scaled')
    # plt.set_aspect('equal')
    plt.gca().set_aspect("equal")

    plt.grid(grid_flag)
    plt.legend()
    # plt.show()

    if img_dict is not None:
        fname = img_dict['fname']
        dpi = img_dict['dpi']
    # plt.savefig(fname=fname, dpi=dpi, format='tiff')

    plt.show()
    # fname += '.tif'
    # plt.savefig(fname=fname, dpi=dpi, format='tif')
# plot it to check whether it is the right status, then store it

def mRow_nCol_Nyquist_plot_0(z_raw_list, goa_res_dict):
    """
    Function:
        为了分析刘鹏芯片的高通量EIS对应的自动拟合结果 dpfc_src\playground\liuPeng\liuPeng_eis_para_03.txt
            EIS-0：
                ECM-0:
                    GOA-1
                        9,WOA,0,5000,[0.07569810035141292,0.00010280705732252737,0.7762615401847585,21359984977334.293,0.19446585770659836,0.5818032657251768,11194184001.65727],0.08155534888346437
                    GOA-2
                    GOA-3
                    GOA-4
                    （GOA-5），可能没有
                ECM-1
                ECM-2
        每个EIS实验文件被概率前三的ECM对应的5种GOA拟合一次
    Requirement
        matplotlib 多行多列
        3 row * 5 col
            没有内容的空着
        每张图
            原始数据（散点）--》标注出被删除的点（重点突出）
            拟合的曲线
                误差
    :return:
    """
    z_raw_real_list = [z.real for z in z_raw_list]
    z_raw_imag_list = [z.imag for z in z_raw_list]
    z_raw_inv_imag_list = [-z_imag for z_imag in z_raw_imag_list]

    # 按道理讲，应该自动判断goa_res_dict的数据有几行几列，这里省事，直接nrow=3, ncol=5
    nrow, ncol = 3, 5
    fig, ax = plt.subplots(nrow, ncol)
    for r in range(nrow):
        ecm_num = list(goa_res_dict.keys())[r]
        ecm_proba = goa_res_dict[ecm_num]['proba']
        print('ECM={0},\t{1}'.format(ecm_num, ecm_proba))
        data_list = goa_res_dict[ecm_num]['data']
        for c in range(len(data_list)):
            ax[r][c].scatter(z_raw_real_list, z_raw_inv_imag_list, label='Raw EIS')

            goa_name, z_sim_list, chi_square = data_list[c]
            z_sim_real_list = [z.real for z in z_sim_list]
            z_sim_imag_list = [z.imag for z in z_sim_list]
            z_sim_inv_imag_list = [-z_imag for z_imag in z_sim_imag_list]
            ax[r][c].plot(z_sim_real_list, z_sim_inv_imag_list, 'o--', color='red',
                          label='{0}={1}'.format(goa_name, round(chi_square, 6)))
            ax[r][c].legend()
    plt.xlim((min(z_raw_real_list), max(z_raw_real_list)))
    plt.ylim((min(z_raw_inv_imag_list), max(z_raw_inv_imag_list)))
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')
    plt.legend()
    plt.show()

def raw_fitted_nyquist_plot(raw_z_list, fitted_z_list, x_lim=[0, 50000], y_lim=[0, 50000]):
    """
    function
        Draw the raw-EIS and fitted-EIS in the same plot
    :param
        raw_z_list:
            raw eis impedance list (complex)
        fitted_z_list:
            fitted eis impedance list (complex)
        x_lim:
            the boundaries of x-axis
        y_lim:
            the boundaries of y-axis
        plot_label:
    :return:
    """
    raw_z_real_list = [z.real for z in raw_z_list]
    raw_z_imag_list = [z.imag for z in raw_z_list]
    raw_z_inv_imag_list = [-z_imag for z_imag in raw_z_imag_list]

    fitted_z_real_list = [z.real for z in fitted_z_list]
    fitted_z_imag_list = [z.imag for z in fitted_z_list]
    fitted_z_inv_imag_list = [-z_imag for z_imag in fitted_z_imag_list]

    fig = plt.figure()
    # plt.plot(raw_z_real_list, raw_z_inv_imag_list, 'o--', label=plot_label)
    plt.plot(raw_z_real_list, raw_z_inv_imag_list, color='blue', marker='o', linestyle='--', linewidth=1, markersize=3,
             label='Raw')

    # plt.plot(fitted_z_real_list, fitted_z_inv_imag_list, 'o--', label=plot_label)
    plt.plot(fitted_z_real_list, fitted_z_inv_imag_list, color='red', marker='^', linestyle='-.', linewidth=1,
             markersize=3, label='Fitted')

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')
    plt.legend()
    plt.show()

def ECM2_multi_nyquist_plot_configer(data_info_dict, lai_fitted_z_list, GOAs_z_list):
    """
    Function
        为下方的 multi_nyquist_plot_by_ECM_NUM(multi_Z_dict) 准备参数multi_Z_dict
    :param
        data_info_dict:{
            ’f‘: [f0(float), f1, f2, ...],
            'chi_square': chi_square float,
            'fmt': color + marker + linestyle
            'z_normed': experiment * area (1.01 * 1e-6 cm^2)
        }
        raw_z_list:
            Z_Raw means 实验设备测得的阻抗数据 * 实际实验面积 1.01 * 1e-6 后的标准面积（一平方厘米）上的阻抗
            [(Z_Raw_real-Complex, Z_Raw_imag-Complex), ...]
        lai_fitted_z_list:
            使用Lai手工拟合的结果，带入Fre和ECM重新模拟的阻抗数据
            [(Z_Lai_real-Complex, Z_Lai_imag-Complex), ...]
        GOAs_z_list:
            [
                [(Z_GOA1_real-Complex, Z_GOA1_imag-Complex), ...],
                [(Z_GOA2_real-Complex, Z_GOA2_imag-Complex), ...],
                [(Z_GOA3_real-Complex, Z_GOA3_imag-Complex), ...],
                [(Z_GOA4_real-Complex, Z_GOA4_imag-Complex), ...],
                [(Z_GOA5_real-Complex, Z_GOA5_imag-Complex), ...],
            ]
    :return:
        multi_Z_dict
    """
    fre_list = data_info_dict['f']
    lai_chi_s = data_info_dict['chi_square']
    raw_z_list = data_info_dict['z_normed']

    # 原始数据和Lai的拟合结果都作为参考，所以都画成黑色
    multi_Z_dict = {}
    multi_Z_dict['raw'] = {'z': raw_z_list,
                           'fmt': 'kx-'}

    pass

def multi_nyquist_plot_by_ECM_NUM(multi_Z_dict, x_lim=[0, 50000], y_lim=[0, 50000], grid_flag=True):
    """
    Function
        1- 将多条来自不同来源的 Nyquist曲线画在一幅图中
        2- 并标注每条数据的来源，比如 Raw、Lai、GOA-PSO，GOA-WOA。。。
        3- 每个来源的Chi-Square
        4- 颜色、符号
    :param
        fmt
            格式字符串，格式字符串只是用于快速设置基本行属性的缩写。所有这些以及更多这些都可以通过关键字参数来控制。此参数不能作为关键字传递
            fmt = 'go--' <==> color='green', marker='o', linestyle='dashed'
        multi_Z_dict{
            'raw':{
                ‘z’：z_list[z0(complex), z1(complex), ...]
                ‘color’: str,
                'fmt'
                    ‘color’: str,
                    'mark': str
            }
            'laiZhaoGui':{
                ‘z’：z_list[z0(complex), z1(complex), ...]
                'chi_square': float
                ‘color’: str,
                'mark': str
            }
            'goa-1':{
                ‘z’：z_list[z0(complex), z1(complex), ...]
                'chi_square': float
                'goa': goa name str,
                ‘color’: str,
                'mark': str
            }
            'goa-2':{
                ‘z’：z_list[z0(complex), z1(complex), ...]
                'chi_square': float
                'goa': goa name str,
                ‘color’: str,
                'mark': str
            }
            ...
            'goa-5':{
                ‘z’：z_list[z0(complex), z1(complex), ...]
                'chi_square': float
                'goa': goa name str,
                ‘color’: str,
                'mark': str
            }
        }
    :return:
    """
    raw_Z_dict = multi_Z_dict['raw']
    lai_Z_dict = multi_Z_dict['laiZhaoGui']
    goa1_Z_dict = multi_Z_dict['goa-1']
    goa2_Z_dict = multi_Z_dict['goa-2']
    goa3_Z_dict = multi_Z_dict['goa-3']
    goa4_Z_dict = multi_Z_dict['goa-4']
    goa5_Z_dict = multi_Z_dict['goa-5']

    markersize = 12
    linewidth = 2
    fig = plt.figure(figsize=(16, 9))

    # raw
    # plt.plot([z.real for z in raw_Z_dict['z']], [z.imag for z in raw_Z_dict['z']], color=raw_Z_dict['color'],
    #          fmt=raw_Z_dict['fmt'], label='Raw', linewidth=linewidth, markersize=markersize)
    plt.plot([z.real for z in raw_Z_dict['z']], [-z.imag for z in raw_Z_dict['z']], raw_Z_dict['fmt'],
             label='Raw', linewidth=linewidth, markersize=markersize)

    # laiZhaoGui
    # plt.plot([z.real for z in lai_Z_dict['z']], [z.imag for z in lai_Z_dict['z']], color=lai_Z_dict['color'],
    #          fmt=raw_Z_dict['fmt'], label='Lai, Chi-Square={}'.format(lai_Z_dict['chi_square']), linewidth=linewidth,
    #          markersize=markersize)
    plt.plot([z.real for z in lai_Z_dict['z']], [-z.imag for z in lai_Z_dict['z']], lai_Z_dict['fmt'],
             label='Lai, Chi-Square={}'.format(lai_Z_dict['chi_square']), linewidth=linewidth,
             markersize=markersize)
    # goa-1
    # plt.plot([z.real for z in goa1_Z_dict['z']], [z.imag for z in goa1_Z_dict['z']], color=goa1_Z_dict['color'],
    #          fmt=goa1_Z_dict['fmt'], label='{0}, Chi-Square={1}'.format(goa1_Z_dict['goa'], goa1_Z_dict['chi_square']),
    #          linewidth=linewidth, markersize=markersize)
    plt.plot([z.real for z in goa1_Z_dict['z']], [-z.imag for z in goa1_Z_dict['z']], goa1_Z_dict['fmt'],
             label='{0}, Chi-Square={1}'.format(goa1_Z_dict['goa'], goa1_Z_dict['chi_square']),
             linewidth=linewidth, markersize=markersize)

    # goa-2
    # plt.plot([z.real for z in goa2_Z_dict['z']], [z.imag for z in goa2_Z_dict['z']], color=goa2_Z_dict['color'],
    #          fmt=goa2_Z_dict['fmt'], label='{0}, Chi-Square={1}'.format(goa2_Z_dict['goa'], goa2_Z_dict['chi_square']),
    #          linewidth=linewidth, markersize=markersize)
    plt.plot([z.real for z in goa2_Z_dict['z']], [-z.imag for z in goa2_Z_dict['z']], goa2_Z_dict['fmt'],
             label='{0}, Chi-Square={1}'.format(goa2_Z_dict['goa'], goa2_Z_dict['chi_square']),
             linewidth=linewidth, markersize=markersize)

    # goa-3
    # plt.plot([z.real for z in goa3_Z_dict['z']], [z.imag for z in goa3_Z_dict['z']], color=goa3_Z_dict['color'],
    #          fmt=goa3_Z_dict['fmt'], label='{0}, Chi-Square={1}'.format(goa3_Z_dict['goa'], goa3_Z_dict['chi_square']),
    #          linewidth=linewidth, markersize=markersize)
    plt.plot([z.real for z in goa3_Z_dict['z']], [-z.imag for z in goa3_Z_dict['z']], goa3_Z_dict['fmt'],
             label='{0}, Chi-Square={1}'.format(goa3_Z_dict['goa'], goa3_Z_dict['chi_square']),
             linewidth=linewidth, markersize=markersize)

    # goa-4
    # plt.plot([z.real for z in goa4_Z_dict['z']], [z.imag for z in goa4_Z_dict['z']], color=goa4_Z_dict['color'],
    #          fmt=goa4_Z_dict['fmt'], label='{0}, Chi-Square={1}'.format(goa4_Z_dict['goa'], goa4_Z_dict['chi_square']),
    #          linewidth=linewidth, markersize=markersize)
    plt.plot([z.real for z in goa4_Z_dict['z']], [-z.imag for z in goa4_Z_dict['z']], goa4_Z_dict['fmt'],
             label='{0}, Chi-Square={1}'.format(goa4_Z_dict['goa'], goa4_Z_dict['chi_square']),
             linewidth=linewidth, markersize=markersize)

    # goa-5
    # plt.plot([z.real for z in goa5_Z_dict['z']], [z.imag for z in goa5_Z_dict['z']], color=goa5_Z_dict['color'],
    #          fmt=goa5_Z_dict['fmt'], label='{0}, Chi-Square={1}'.format(goa5_Z_dict['goa'], goa5_Z_dict['chi_square']),
    #          linewidth=linewidth, markersize=markersize)
    plt.plot([z.real for z in goa5_Z_dict['z']], [-z.imag for z in goa5_Z_dict['z']], goa5_Z_dict['fmt'],
             label='{0}, Chi-Square={1}'.format(goa5_Z_dict['goa'], goa5_Z_dict['chi_square']),
             linewidth=linewidth, markersize=markersize)

    plt.xlim(x_lim)
    plt.ylim(y_lim)
    plt.xlabel('Z$_{real}$ [$\Omega$]')
    plt.ylabel('-Z$_{imag}$ [$\Omega$]')
    plt.grid(grid_flag)
    plt.legend()
    plt.show()