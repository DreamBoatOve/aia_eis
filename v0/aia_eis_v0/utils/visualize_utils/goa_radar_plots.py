import matplotlib.pyplot as plt
import numpy as np

from utils.post_process.goa_score import goa_scorer

"""
Function
    Three criterions, Accuracy score, Running time Score and Stability score, of 20 GOAs fitting on one kind of ECM
    is displayed on one radar plot
    For 9 kinds of ECMs, there is going to be 9 radar plots
tools
    matplotlib
        key words: 3D plotting
        Bar chart on polar axis
        控制颜色
            颜色之间的对应关系为 
                b—blue, c—cyan, g—green, k—-black, m—magenta, r—red, w—white, y—-yellow
        控制线型
            符号和线型之间的对应关系 
                - 实线; –- 短线; -. 短点相间线; ：虚点线
        控制标记风格
            .  Point marker; ,  Pixel marker; o  Circle marker; 
            v  Triangle down marker; ^  Triangle up marker; <  Triangle left marker; >  Triangle right marker 
            1  Tripod down marker; 2  Tripod up marker; 3  Tripod left marker; 4  Tripod right marker
            s  Square marker; p  Pentagon marker; *  Star marker; h  Hexagon marker; H  Rotated hexagon D Diamond marker
            d  Thin diamond marker; | Vertical line (vlinesymbol) marker; _  Horizontal line (hline symbol) marker
            +  Plus marker; x  Cross (x) marker
refers:
    python的matplotlib---雷达图
        https://www.cnblogs.com/changfan/p/11799721.html
Unfinished:
    After plot the radar, Reset the figure as following:
        Accuracy, Stability, Time axis_font = {'fontname': 'Microsoft YaHei', 'weight':'bold', 'size': '30'}
            left;       0.02 
            bottom;     0.02
            right;      1
            top;        0.95
            PNG or SVG format
            
        Accuracy, Stability, Time axis_font = {'fontname': 'Microsoft YaHei', 'size': '20'}
            left;       0.02 
            bottom;     0.02
            right;      0.98
            top;        0.96
            PNG or SVG format
            
    Decide to put 20 GOAs in a plot (too much to be clear), or just the best five GOAs?
        radar 正文中只放性能前5的GOA，全部的GOA也画出来放在补充材料中，以免有人感兴趣
    ----------------------------------------------
    Done: Adjust the font size of 'Accuracy, Stability, Time'
            Liu Peng:
            参考来源'https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot/23572192#23572192'
            定义坐标轴字体格式，包括大小颜色等
"""

# 'sub' means sub class/category
goa_category_dict = {'evolution': { 'sub': ['DE', 'EDA', 'EP', 'ES', 'GA'],
                                    'color': 'b',
                                    'marker': ['o', 'v', 's', 'p', 'h'],
                                    'lineStyle': '-'},
                     'human': {'sub': ['GSO', 'HS', 'ICA', 'ISA', 'TLBO'],
                                'color': 'k',
                                'marker': ['o', 'v', 's', 'p', 'h'],
                                'lineStyle': '--'},
                     'physic': {'sub': ['BB_BC', 'BH', 'CSS', 'GSA', 'MVO'],
                               'color': 'r',
                               'marker': ['o', 'v', 's', 'p', 'h'],
                               'lineStyle': '-.'},
                     'swarm':{'sub': ['ACA', 'ABC', 'GWO', 'PSO', 'WOA'],
                              # 'color': 'y', y: yellow
                              'color': 'g', # g: green
                              'marker': ['o', 'v', 's', 'p', 'h'],
                              'lineStyle': ':'}
                     }

def GOA20_three_criterions_radar_plots(goa_scores_on_three_criterions_dict, ecm_num):
    """
    把20种GOA在某种ECM上的三个性能全绘制在一幅图上，每个radar图上有20个圈，看起来很混乱
    :param
        goa_scores_on_three_criterions_dict:
        ecm_num:
    :return:
    """
    global goa_category_dict

    # Liu Peng:
    # 参考来源'https://stackoverflow.com/questions/3899980/how-to-change-the-font-size-on-a-matplotlib-plot/23572192#23572192'
    # 定义坐标轴字体格式，包括大小颜色等
    # axis_font = {'fontname': 'Microsoft YaHei', 'size': '20'}
    # Font for : Accuracy, Stability, Time
    axis_font = {'fontname': 'Microsoft YaHei', 'weight': 'bold', 'size': '30'}

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 使用ggplot的风格绘图, [ggplot] makes the background of radar as grey, ugly
    # plt.style.use('ggplot')

    # 构造数据
    # values = [3.2, 2.1, 3.5, 2.8, 3, 4]
    # values_list =
    # values_1 = [2.4, 3.1, 4.1, 1.9, 3.5, 2.3]
    # feature = ['个人能力', 'QC知识', "解决问题能力", "服务质量意识", "团队精神", "IQ"]
    # feature = ['Accuracy', 'Stability', 'Time']

    # N = len(values)
    N = 3

    # 设置雷达图的角度，用于平分切开一个平面,
    # this angles arr starts from 0 degree, which means the accuracy score is displayed on x-axis
    # add 90 degrees to rotate the [accuracy score / original x-axis] to original y-axis' position
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2

    # 使雷达图封闭起来
    # values = np.concatenate((values, [values[0]]))
    angles_arr = np.concatenate((angles, [angles[0]]))

    # 绘图
    fig = plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)

    # ---------- mine data -------------------------------
    for goa_name, scores_dict in goa_scores_on_three_criterions_dict.items():
        goa_score_on_ecmNum_list = scores_dict[ecm_num]
        # 使雷达图封闭起来
        goa_score_on_ecmNum_arr = np.concatenate((goa_score_on_ecmNum_list, [goa_score_on_ecmNum_list[0]]))

        # -------------------- get plot configuration: color, marker, line style --------------------
        for goa_category_key, one_class_goa_category_dict in goa_category_dict.items():
            goa_category_list = one_class_goa_category_dict['sub']
            if goa_name in goa_category_list:
                goa_name_index = goa_category_list.index(goa_name)
                linestyle = one_class_goa_category_dict['lineStyle']
                color = one_class_goa_category_dict['color']
                marker = one_class_goa_category_dict['marker'][goa_name_index]
                fmt = color + marker + linestyle
                ax.plot(angles_arr, goa_score_on_ecmNum_arr, fmt, markersize=10, linewidth=5, label='')
                # fill triangle with color
                # ax.fill(angles_arr, goa_score_on_ecmNum_arr, color, alpha=0.2)
                break
        # -------------------- get plot configuration: color, marker, line style --------------------

        # ax.plot(angles_arr, goa_score_on_ecmNum_arr, 'o-', linewidth=2, label='')
    # ---------- mine data -------------------------------
    # 绘制折线图
    # ax.plot(angles, values, 'o-', linewidth=2, label='活动前')
    # ax.fill(angles, values, 'r', alpha=0.5)

    # 填充颜色
    # ax.plot(angles, values_1, 'o-', linewidth=2, label='活动后')
    # ax.fill(angles, values_1, 'b', alpha=0.5)

    # --------------- 添加每个特质的标签 ---------------
    # 去掉Acc、Stability、Time，这些自带的标题挡住了部分图像，之后手动加上去
    # ax.set_thetagrids(angles * 180 / np.pi, feature, **axis_font) # Liu Peng: 以kwargs对象形式使用定义的字体格式
    ax.set_thetagrids(angles * 180 / np.pi) # Liu Peng: 以kwargs对象形式使用定义的字体格式

    # not working
    # ax.set_thetagrids(angles * 180 / np.pi, feature, {'family':'Times New Roman', 'size': 56})
    # ax.set_thetagrids(angles = angles * 180 / np.pi, labels = feature, {'family':'Times New Roman', 'fontsize': 56})
    # --------------- 添加每个特质的标签 ---------------

    # 设置极轴范围
    # ax.set_ylim(0, 10.7)
    ax.set_ylim(0, 12)
    # ax.set_ylim(0, 10.7, fontsize=10) ==> TypeError: set_ylim() got an unexpected keyword argument 'fontsize'
    # plt.yticks(fontproperties = 'Times New Roman', size = 25)
    # Adjust the 2,4,6,8,10
    plt.yticks(fontproperties = 'Microsoft YaHei', size = 28)

    # 添加标题
    # plt.title('20 GOAs\' fitting scores on ECM-{0}'.format(ecm_num))
    # Remove title, 1- it is not necessary; 2- Without title, there are space for the main figure

    # plt.xlabel('xxxxxxxxxxx', fontdict={'family': 'Times New Roman', 'size': 56})
    # plt.ylabel('yyyyyyyyyyy', fontdict={'family': 'Times New Roman', 'size': 56})

    # 增加网格纸
    ax.grid(True)
    plt.show()
g_s = goa_scorer(excel_path='../../goa/goa_training_records.xlsx', sheet_name='goa_training_res_2')
goa_scores_on_three_criterions_dict, goa_weighted_score_dict = g_s.get_goa_scores()
GOA20_three_criterions_radar_plots(goa_scores_on_three_criterions_dict, ecm_num=1)

goa_ecm_match_dict = {
    # ecm_num : [1st_GOA, 2nd_GOA, 3rd_GOA, 4th_GOA, 5th_GOA]
    1:['DE', 'GSO', 'TLBO', 'MVO', 'WOA'],
    2:['GSO', 'GWO', 'WOA', 'DE', 'ABC'],
    3:['GWO', 'WOA', 'HS', 'ABC', 'TLBO'],
    4:['GSO', 'WOA', 'GWO', 'CSS', 'HS'],
    5:['WOA', 'ABC', 'EP', 'DE', 'GWO'],
    6:['WOA', 'GWO', 'ABC', 'DE', 'HS'],
    7:['WOA', 'DE', 'ABC', 'GWO', 'HS'],
    8:['GWO', 'WOA', 'HS', 'ABC', 'CSS'],
    9:['DE', 'WOA', 'ABC', 'ICA', 'BH']
}
def GOA5_three_criterions_radar_plots(goa_scores_on_three_criterions_dict, ecm_num):
    """
    把每种ECM上综合性能前5的三个性能全绘制在一幅图上，一幅图上只有5个圈，这样图会清晰许多
    :param
        goa_scores_on_three_criterions_dict:
        ecm_num:
    :return:
    """
    global goa_ecm_match_dict
    global goa_category_dict

    # 定义坐标轴字体格式，包括大小颜色等
    # axis_font = {'fontname': 'Microsoft YaHei', 'size': '20'}
    axis_font = {'fontname': 'Microsoft YaHei', 'weight':'bold', 'size': '30'}

    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 使用ggplot的风格绘图
    plt.style.use('ggplot')

    feature = ['Accuracy', 'Stability', 'Time']
    N = len(feature)

    # 设置雷达图的角度，用于平分切开一个平面,
    # this angles arr starts from 0 degree, which means the accuracy score is displayed on x-axis
    # add 90 degrees to rotate the [accuracy score / original x-axis] to original y-axis' position
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False) + np.pi / 2
    # 使雷达图封闭起来
    angles_arr = np.concatenate((angles, [angles[0]]))

    # 绘图
    fig = plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)

    # ---------- mine data -------------------------------
    best5_GOAs_list = goa_ecm_match_dict[ecm_num]
    for goa_name in best5_GOAs_list:
        three_scores_list = goa_scores_on_three_criterions_dict[goa_name][ecm_num]
        # 使雷达图封闭起来
        three_scores_arr = np.concatenate((three_scores_list, [three_scores_list[0]]))

        # -------------------- get plot configuration: color, marker, line style --------------------
        for goa_category_key, one_class_goa_category_dict in goa_category_dict.items():
            goa_category_list = one_class_goa_category_dict['sub']
            if goa_name in goa_category_list:
                goa_name_index = goa_category_list.index(goa_name)
                linestyle = one_class_goa_category_dict['lineStyle']
                color = one_class_goa_category_dict['color']
                marker = one_class_goa_category_dict['marker'][goa_name_index]
                fmt = color + marker + linestyle
                ax.plot(angles_arr, three_scores_arr, fmt, markersize=10, linewidth=4, label='')
                break
        # -------------------- get plot configuration: color, marker, line style --------------------
    # ---------- mine data -------------------------------

    # 添加每个特质的标签
    ax.set_thetagrids(angles * 180 / np.pi, feature, **axis_font)  # Liu Peng: 以kwargs对象形式使用定义的字体格式
    # 设置极轴范围
    ax.set_ylim(0, 10.7)
    # Adjust the font of '2,4,6,8,10'
    plt.yticks(fontproperties = 'Microsoft YaHei', size = 25)

    # 增加网格纸
    ax.grid(True)
    plt.show()
# g_s = goa_scorer(excel_path='../../goa/goa_training_records.xlsx', sheet_name='goa_training_res')
# goa_scores_on_three_criterions_dict, goa_weighted_score_dict = g_s.get_goa_scores()
# GOA5_three_criterions_radar_plots(goa_scores_on_three_criterions_dict, ecm_num=9)