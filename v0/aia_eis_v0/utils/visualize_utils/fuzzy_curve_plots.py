import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib.ticker import LinearLocator, FormatStrFormatter

"""
function
    画EIS高中低三个频段Fuzzy-Curve在二维数据上的三维图
    
requirements:
    1- 三张图（高、中、低频段）的视角一致
    2- 纵轴Z的坐标范围一致，方便对比
    3- 找出每张图的Z的最大、小值（print出来），并标明其位置（绘图后，手动标注）
    4- 在图上标出 差值 = 最大 - 最小
    
Used Techs:
    Matplotlib 3D plot
        plot code refers from: surface3d.py, because it has colorbar to show the maximum and minimum
        
Unfinished Modifications:
    1- More precise Z-range
    2- Beautify background
    3- Adjust the distance between the figure and colorbar
    
"""
def fuzzy_curve_3D_plot(x_list, y_list, z_list):
    z_max, z_min = max(z_list), min(z_list)
    z_max_index, z_min_index = z_list.index(z_max), z_list.index(z_min)
    z_max_coor = (x_list[z_max_index], y_list[z_max_index], z_max)
    z_min_coor = (x_list[z_min_index], y_list[z_min_index], z_min)
    print('Max point coordinate:', z_max_coor)
    print('Min point coordinate:', z_min_coor)

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    # Plot the surface.
    """
    plot_surface
        X, Y, Z : 2d arrays
        have to change 1D-list to 2D np.array
    """
    # 1- convert list to np.array
    x_1D_arr = np.array(x_list)
    y_1D_arr = np.array(y_list)
    z_1D_arr = np.array(z_list)

    # 2- modify the shape of arr from 1D to 2D, I already now the list's length is 100 * 100
    x_2D_arr = x_1D_arr.reshape((100, 100))
    y_2D_arr = y_1D_arr.reshape((100, 100))
    z_2D_arr = z_1D_arr.reshape((100, 100))

    surf = ax.plot_surface(x_2D_arr, y_2D_arr, z_2D_arr, cmap=cm.coolwarm, linewidth=0, antialiased=False)

    # Customize the display of each axis (x, y, z).
    ax.set_xlim(-0.01, 1.01)
    ax.set_ylim(-0.01, 1.01)
    ax.set_zlim(5, 6.5)
    ax.zaxis.set_major_locator(LinearLocator(10))
    ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()