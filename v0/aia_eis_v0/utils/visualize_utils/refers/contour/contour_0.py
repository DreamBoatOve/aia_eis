# Implementation of matplotlib function
import matplotlib.pyplot as plt
import numpy as np

"""
Contour
    画出来的等高图只有等高线
Contourf
    画出来的等高平面被相同的颜色填充
"""

def plot_contour_by_contour():
    feature_x = np.arange(0, 50, 2)
    feature_y = np.arange(0, 50, 3)

    # Creating 2-D grid of features
    [X, Y] = np.meshgrid(feature_x, feature_y)

    fig, ax = plt.subplots(1, 1)

    Z = np.cos(X / 2) + np.sin(Y / 4)

    # plots contour lines
    ax.contour(X, Y, Z)

    ax.set_title('Contour Plot')
    ax.set_xlabel('feature_x')
    ax.set_ylabel('feature_y')

    plt.show()
# plot_contour_by_contour()

def plot_contour_by_contourF():
    feature_x = np.linspace(-5.0, 3.0, 70)
    feature_y = np.linspace(-5.0, 3.0, 70)

    # Creating 2-D grid of features
    [X, Y] = np.meshgrid(feature_x, feature_y)

    fig, ax = plt.subplots(1, 1)

    Z = X ** 2 + Y ** 2

    # plots filled contour plot
    ax.contourf(X, Y, Z)

    ax.set_title('Filled Contour Plot')
    ax.set_xlabel('feature_x')
    ax.set_ylabel('feature_y')

    plt.show()
# plot_contour_by_contourF()