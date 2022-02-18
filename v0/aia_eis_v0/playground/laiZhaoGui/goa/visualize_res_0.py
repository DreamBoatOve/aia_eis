from utils.file_utils.filename_utils import get_date_prefix
from playground.laiZhaoGui.goa.get_lai_manual_fitting_res import read_lai_manual_fitting_res, read_lai_test_coordinate, pack_lai_manual_fitting_res, wrap_lai_data_4_contour
from playground.laiZhaoGui.goa.get_GOAs_fitting_res import get_GOAs_best_fitting_res, pack_GOAs_fit_res, wrap_GOAs_data_4_contour

"""
模块功能：
    集合Lai-EIS 人工处理结果 与 GOA直接处理原始EIS结果 画图
    version
        0：visualize_res_0.py
"""

# 1- Get Lai's manual fitting R(RC)_IS_lin-kk_res.txt and GOAs R(RC)_IS_lin-kk_res.txt
# 1.1- Get Lai's manual fitting R(RC)_IS_lin-kk_res.txt
lai_manual_fit_res_dict_list = read_lai_manual_fitting_res(ex_fp='../../../../datasets/experiement_data/laiZhaoGui/eis/2020-07-22-阻抗类型整理2006.xlsx',\
                                                           sheet_name='statistic')
coor_dict_list = read_lai_test_coordinate(ex_fp='../../../../datasets/experiement_data/laiZhaoGui/eis/坐标.xlsx',\
                                          sheet_name='Sheet1')
lai_manual_fit_res_dict_list = pack_lai_manual_fitting_res(lai_manual_fit_res_dict_list,\
                                                           coor_dict_list)
lai_x_list, lai_y_list, lai_z_list = wrap_lai_data_4_contour(lai_manual_fit_res_dict_list)

# 1.2- Get GOAs R(RC)_IS_lin-kk_res.txt
goa_fit_res_dict_list = get_GOAs_best_fitting_res(fp='./R(RC)_IS_lin-kk_res.txt/2nd/magNum=2_res', order_str='2nd')
goa_fit_res_dict_list = pack_GOAs_fit_res(goa_fit_res_dict_list, coor_dict_list)
goa_x_list, goa_y_list, goa_z_list = wrap_GOAs_data_4_contour(goa_fit_res_dict_list)

# 2- Export coordinates of test points and their fitting results (Lai's and GOAs') to txt
def export_fit_res(x_list, y_list, z_list, fn):
    # sort data by y then by x
    xyz_list = [[x,y,z] for x,y,z in zip(x_list, y_list, z_list)]
    xyz_list.sort(key= lambda xyz : (xyz[1], xyz[0]), reverse=False)
    x_list = [xyz[0] for xyz in xyz_list]
    y_list = [xyz[1] for xyz in xyz_list]
    z_list = [xyz[2] for xyz in xyz_list]

    header = ','.join(['x_coor', 'y_coor', 'z_coor(Chi-Square)']) + '\n'
    with open(fn, 'a+') as file:
        file.write(header)
        for x, y, z in zip(x_list, y_list, z_list):
            line = ','.join(list(map(str, [x,y,z]))) + '\n'
            file.write(line)
# 2.1- Export coordinates of test points and their fitting results Lai's to txt
# export_fit_res(x_list=lai_x_list, y_list=lai_y_list, z_list=lai_z_list, fn=get_date_prefix()+'lai_fit_res.txt')
# 2.2- Export coordinates of test points and their fitting results GOAs' to txt
export_fit_res(x_list=goa_x_list, y_list=goa_y_list, z_list=goa_z_list, fn=get_date_prefix()+'GOA_fit_res.txt')

"""
3- Use Origin to plot Chi-Square-Contour
我用各种ECM上最优的前五种GOA拟合的Chi-Squared误差，与赖拟合的误差，分别画在上下两个等高图上
    3.1 Delete abnormal data in GOA 's fit R(RC)_IS_lin-kk_res.txt manually
        Lai
            Z: Min              Max
                0.0004402       0.04055
        GOA
            Z: Min                          Max(1st, too big and weird, delete it)         Max(2nd, better and normal)         Max(3rd)
                0.0033192209598534358       13891082844.471136                              54.41158700914487                   27.29493804319961
        1- Delete the huge abnormal data in z2_list (GOAs' R(RC)_IS_lin-kk_res.txt), (x=1.3,y=0.458,z=13891082844.471136,fn=2-3,ECM-Num=9)
        2- Set the value range of colorbar as 1e-4 ~ 1e2, 6 margins
    3.2 plot setting
        1- 各自 或 一起 拥有colorbar
        2- colorbar 上的刻度是对数分布的
        3- the range of x-axis is: -0.2 ~ 16.5 mm (Increment 2); the range of y-axis is: -0.1 ~ 2 mm (Increment 0.5).
            x Range     y Range     z Range
            min max     min max     min(laiZhaoGui)        max(laiZhaoGui)
            0   16.2    0   1.833   0.0004402       0.04055
                                    z Range
                                    min(laiZhaoGui)        max(laiZhaoGui)
                                    0.003           54.4
    fit-res_contour_plot_details
        origin导入txt文件 
            https://jingyan.baidu.com/article/870c6fc3324488b03fe4beca.html
        origin 画contour
            https://jingyan.baidu.com/article/72ee561a75b777e16038df68.html
        origin contour colorbar 设置成 对数
            6.9.2 Contour Plots and Color Mapping
            https://www.originlab.com/doc/Tutorials/Contour-Color-Map
        origin turn off 'speed mode'
            打开在header中的Graph，在下拉菜单的中下部可以找到'Speed Mode'
        origin 设置x，y轴的坐标范围
            https://jingyan.baidu.com/article/e4511cf3185eb56a855eaf05.html
        origin colorbar两端无用的色块删除掉
            在plot details - Colormap/Contours 中 点击欲删除的色块，把颜色设置成None
        origin colorbar value in log scale
            在plot details - Numeric Formats - Format 中选择Scientific
        origin 设置x，y轴的长宽比例 按照数值大小正常显示
            graph -》 layer -》size/speed 中手动调整
        origin contour smoothing
            graph -》 layer -》具体的数据book -》Contouring Info -》Smoothing
                有 total points increase factor 和 Smoothing Parameter两个参数可调，但是怎么调都感觉数据失真很严重，所以决定不用平滑功能
        Chi-squared Error的图不画成 Colormap Surface彩面图 （https://www.jianshu.com/p/bac08e1dc36a），减轻 GOA的Error 
        和 Lai的Error 差别的直观视觉体现，可在后续拟合的电路元件的分布上使用 Colormap Surface彩面图
"""
