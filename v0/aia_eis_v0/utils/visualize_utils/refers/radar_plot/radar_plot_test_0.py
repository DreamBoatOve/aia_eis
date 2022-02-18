import matplotlib.pyplot as plt
import numpy as np

"""
python的matplotlib---雷达图
    Simple and clear
    https://www.cnblogs.com/changfan/p/11799721.html
"""

def one_point_in_radar():
    """
    20:以20作为半径
    ylim(0,100):设置极轴的范围
    lw=2:表示极坐标图案的宽度
    ro:绘制的极坐标图形为红色圆点
    """

    plt.polar(0.25 * np.pi, 20, "ro", lw=2)
    plt.ylim(0, 100)
    plt.show()
# one_point_in_radar()

def polygon_in_radar():
    """
    绘制多个点，并且第一个点与最后一个点相同，使其成为闭合图案
    """

    theta = np.array([0.25, 0.75, 1, 1.5, 0.25])
    r = [20, 60, 40, 80, 20]

    plt.polar(theta * np.pi, r, "r-", lw=2)
    plt.ylim(0, 100)
    plt.show()
# polygon_in_radar()

def color_polygon_in_radar():
    # 使用ggplot的绘图风格
    plt.style.use('ggplot')

    # 构建角度与值
    theta = np.array([0.25, 0.75, 1, 1.5, 0.25])
    r = [20, 60, 40, 80, 20]

    plt.polar(theta * np.pi, r, "r-", lw=1)

    # 设置填充颜色，并且透明度为0.75
    plt.fill(theta * np.pi, r, 'r', alpha=0.75)
    plt.ylim(0, 100)

    # 显示网格线
    plt.grid(True)
    plt.show()
color_polygon_in_radar()

def multi_polygon_in_radar():
    # 中文和负号的正常显示
    plt.rcParams['font.sans-serif'] = 'Microsoft YaHei'
    plt.rcParams['axes.unicode_minus'] = False

    # 使用ggplot的风格绘图
    plt.style.use('ggplot')

    # 构造数据
    values = [3.2, 2.1, 3.5, 2.8, 3, 4]
    values_1 = [2.4, 3.1, 4.1, 1.9, 3.5, 2.3]
    feature = ['个人能力', 'QC知识', "解决问题能力", "服务质量意识", "团队精神", "IQ"]

    N = len(values)

    # 设置雷达图的角度，用于平分切开一个平面
    angles = np.linspace(0, 2 * np.pi, N, endpoint=False)

    # 使雷达图封闭起来
    values = np.concatenate((values, [values[0]]))
    angles = np.concatenate((angles, [angles[0]]))

    values_1 = np.concatenate((values_1, [values_1[0]]))
    # 绘图
    fig = plt.figure()
    # 设置为极坐标格式
    ax = fig.add_subplot(111, polar=True)
    # 绘制折线图
    ax.plot(angles, values, 'o-', linewidth=2, label='活动前')
    ax.fill(angles, values, 'r', alpha=0.5)

    # 填充颜色
    ax.plot(angles, values_1, 'o-', linewidth=2, label='活动后')
    ax.fill(angles, values_1, 'b', alpha=0.5)

    # 添加每个特质的标签
    ax.set_thetagrids(angles * 180 / np.pi, feature)
    # 设置极轴范围
    ax.set_ylim(0, 5)
    # 添加标题
    plt.title('活动前后员工状态')
    # 增加网格纸
    ax.grid(True)
    plt.show()
# multi_polygon_in_radar()