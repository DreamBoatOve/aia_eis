import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.ticker import MaxNLocator
from matplotlib import ticker, cm

from playground.laiZhaoGui.goa.get_lai_manual_fitting_res import read_lai_manual_fitting_res, read_lai_test_coordinate, pack_lai_manual_fitting_res, wrap_lai_data_4_contour
from playground.laiZhaoGui.goa.get_GOAs_fitting_res import get_GOAs_best_fitting_res, pack_GOAs_fit_res, wrap_GOAs_data_4_contour

def two_vertical_contour_0(x1_list, y1_list, z1_list, x2_list, y2_list, z2_list):
    """
    Function:
        我用各种ECM上最优的前五种GOA拟合的Chi-Squared误差，与赖拟合的误差，分别画在上下两个等高图上
    Requirement：
        1- 各自 或 一起 拥有colorbar
        2- colorbar 上的刻度是对数分布的
        3- the range of x-axis is: 0 ~ 17 mm; the range of y-axis is: 0 ~ 2 mm.
    :return:
    """
    """
    Lai
        Z: Min              Max
            0.0004402       0.04055
    GOA
        Z: Min                          Max(1st, too big and weird, delete it)         Max(2nd, better and normal)         Max(3rd)
            0.0033192209598534358       13891082844.471136                              54.41158700914487                   27.29493804319961
    1- Delete the huge abnormal data in z2_list (GOAs' R(RC)_IS_lin-kk_res.txt), (x=1.3,y=0.458,z=13891082844.471136,fn=2-3,ECM-Num=9)
    2- Set the value range of colorbar as 1e-4 ~ 1e2, 6 margins
    """
    # 将z_min ~ z_max等分成15份，每个数值区间用一个颜色表示
    # z_min = min(min(z1_list), min(z2_list))
    # z_max = max(max(z1_list), max(z2_list))
    # print(z_min, z_max)
    # levels = MaxNLocator(nbins=15).tick_values(z.min(), z.max())
    # 1- Delete the huge abnormal data in z2_list (GOAs' R(RC)_IS_lin-kk_res.txt), (x=1.3,y=0.458,z=13891082844.471136,fn=2-3,ECM-Num=9)
    abnormal_index = z2_list.index(max(z2_list))
    del x2_list[abnormal_index]
    del y2_list[abnormal_index]
    del z2_list[abnormal_index]

    """
    The data format of x, y and z has to be 2D np.array
        laiZhaoGui has 125(Odd) pieces of data, in order to transfer them into 2D np.array, remove the last data(can not randomly add one piece of data).
        So laiZhaoGui's data will be 124 ==> 62, 2
        Goa has 126(even) pieces of data, in order to transfer them into 2D np.array.
        So Goa's data will be 126 ==> 63, 2
    """
    # x,y的数值要按照逐渐变大或变小的规律有序排列，不然图像会乱
    xyz1_list = [[x,y,z] for x,y,z in zip(x1_list, y1_list, z1_list)]
    # sort by y, then sort by x
    xyz1_list.sort(key=lambda xyz:(xyz[1], xyz[0]), reverse=False)
    x1_sorted_list = [xyz[0] for xyz in xyz1_list]
    y1_sorted_list = [xyz[1] for xyz in xyz1_list]
    z1_sorted_list = [xyz[2] for xyz in xyz1_list]

    x1_2D_arr = np.array(x1_sorted_list[:len(x1_sorted_list)-1]).reshape((62, 2))
    y1_2D_arr = np.array(y1_sorted_list[:len(y1_sorted_list)-1]).reshape((62, 2))
    z1_2D_arr = np.array(z1_sorted_list[:len(z1_sorted_list)-1]).reshape((62, 2))

    # x1_2D_arr = np.array(x1_list[:len(x1_list)-1]).reshape((62, 2))
    # y1_2D_arr = np.array(y1_list[:len(y1_list)-1]).reshape((62, 2))
    # z1_2D_arr = np.array(z1_list[:len(z1_list)-1]).reshape((62, 2))

    xyz2_list = [[x,y,z] for x,y,z in zip(x2_list, y2_list, z2_list)]
    xyz2_list.sort(key=lambda xyz:(xyz[1], xyz[0]), reverse=False)
    x2_sorted_list = [xyz[0] for xyz in xyz2_list]
    y2_sorted_list = [xyz[1] for xyz in xyz2_list]
    z2_sorted_list = [xyz[2] for xyz in xyz2_list]
    x2_2D_arr = np.array(x2_sorted_list).reshape((63, 2))
    y2_2D_arr = np.array(y2_sorted_list).reshape((63, 2))
    z2_2D_arr = np.array(z2_sorted_list).reshape((63, 2))

    # x2_2D_arr = np.array(x2_list).reshape((63, 2))
    # y2_2D_arr = np.array(y2_list).reshape((63, 2))
    # z2_2D_arr = np.array(z2_list).reshape((63, 2))

    # 将z_min ~ z_max等分成6份，每个数值区间用一个颜色表示
    level_arr = np.array([10 ** i for i in range(-4, 3)])

    # pick the desired colormap, sensible levels, and define a normalization
    # instance which takes data values and translates those into levels.
    # cmap = plt.get_cmap('PiYG')
    cmap = plt.get_cmap('viridis')
    norm = BoundaryNorm(level_arr, ncolors=cmap.N, clip=True)

    fig, (ax1, ax2) = plt.subplots(nrows=2)
    fig.suptitle('Title of two subplots')

    # contours are *point* based plots, so convert our bound into point
    # centers
    # cf1 = ax1.contourf(x1_list, y1_list, z1_list, levels=level_arr, cmap=cmap)
    # setting the log locator tells contourf to use a log scale:

    # cf1 = ax1.contourf(x1_2D_arr, y1_2D_arr, z1_2D_arr, locator=ticker.LogLocator(), levels=level_arr, cmap=cmap)
    # fig.colorbar(cf1, ax=ax1)
    # ax1.set_title('1 Lai R(RC)_IS_lin-kk_res.txt')

    # im = ax1.pcolormesh(x, y, z, cmap=cmap, norm=norm)
    im = ax1.pcolormesh(x1_2D_arr, y1_2D_arr, z1_2D_arr, cmap=cmap, norm=norm)
    fig.colorbar(im, ax=ax1)
    ax1.set_title('pcolormesh with levels')

    # cf2 = ax2.contourf(x2_list, y2_list, z2_list, levels=level_arr, cmap=cmap)
    cf2 = ax2.contourf(x2_2D_arr, y2_2D_arr, z2_2D_arr, locator=ticker.LogLocator(), levels=level_arr, cmap=cmap)
    fig.colorbar(cf2, ax=ax2)
    # ax2.xlim(0,20) # AttributeError: 'AxesSubplot' object has no attribute 'xlim'
    ax2.set_title('2 GOA R(RC)_IS_lin-kk_res.txt')
    # ax1.plot(x1, y1, 'o-')
    # ax1.set_ylabel('Damped oscillation')

    # ax2.plot(x2, y2, '.-')
    # ax2.set_xlabel('time (s)')
    # ax2.set_ylabel('Undamped')

    plt.xlim(0, 17)
    plt.ylim(0, 2)
    plt.show()

# 1- Get Lai's manual fitting R(RC)_IS_lin-kk_res.txt and GOAs R(RC)_IS_lin-kk_res.txt
# 1.1- Get Lai's manual fitting R(RC)_IS_lin-kk_res.txt
lai_manual_fit_res_dict_list = read_lai_manual_fitting_res(ex_fp='../../../datasets/experiement_data/laiZhaoGui/eis/2020-07-22-阻抗类型整理2006.xlsx',\
                                                           sheet_name='statistic')
coor_dict_list = read_lai_test_coordinate(ex_fp='../../../datasets/experiement_data/laiZhaoGui/eis/坐标.xlsx',\
                                          sheet_name='Sheet1')
lai_manual_fit_res_dict_list = pack_lai_manual_fitting_res(lai_manual_fit_res_dict_list,\
                                                           coor_dict_list)
lai_x_list, lai_y_list, lai_z_list = wrap_lai_data_4_contour(lai_manual_fit_res_dict_list)

# 1.2- Get GOAs R(RC)_IS_lin-kk_res.txt
goa_fit_res_dict_list = get_GOAs_best_fitting_res(fp='../../playground/laiZhaoGui/goa/R(RC)_IS_lin-kk_res.txt/magNum=2_res')
goa_fit_res_dict_list = pack_GOAs_fit_res(goa_fit_res_dict_list, coor_dict_list)
goa_x_list, goa_y_list, goa_z_list = wrap_GOAs_data_4_contour(goa_fit_res_dict_list)

# 2- Plot contour
two_vertical_contour_0(x1_list=lai_x_list, y1_list=lai_y_list, z1_list=lai_z_list,\
                       x2_list=goa_x_list, y2_list=goa_y_list, z2_list=goa_z_list)
"""
我认为失败的原因
    1- python 的contourf 要求横纵坐标都是网格状的二维数组，但是我的实际数据（测试点的坐标）不是每行元素个数一致，创建二维数组就意味着，有些位置
    上的数字是瞎编的，没有意义，如果填写0，反而意味着拟合的效果非常好，这是错的
    2- 之前工作（三元合金）的contour就是用origin画的，应该是可行的
"""