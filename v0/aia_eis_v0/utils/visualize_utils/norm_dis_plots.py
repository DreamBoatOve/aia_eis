import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
from mpl_toolkits.mplot3d import Axes3D

"""
Draw 1D, 2D Normal Distribution to explain my modification in Fuzzy Curve
"""
def Gaussian_Distribution(N=2, M=1000, m=0, sigma=1):
    '''
    Function
        Generate N dimensions Normal distribution data
    ----------
    Parameters
    ----------
        N 维度
        M 样本数
        m 样本均值
        sigma: 样本方差
    Returns
    -------
        data  shape(M, N), M 个 N 维服从高斯分布的样本
        Gaussian  高斯分布概率密度函数
    '''
    mean = np.zeros(N) + m  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)
    return data, Gaussian

def generate_1D_data():
    #-------------------- 生成两个正态分布的数据 --------------------
    # 生成mean=0.3， sigma=0.1的数据，x=【0~1，step=0.001】
    mean0, sigma0 = 0.3, 0.1
    mean1, sigma1 = 0.7, 0.2
    x = np.linspace(0, 1, 1000)

    data0, Gaussian0 = Gaussian_Distribution(N=1, M=1000, m=mean0, sigma=sigma0)
    y1 = Gaussian0.pdf(x)
    data1, Gaussian1 = Gaussian_Distribution(N=1, M=1000, m=mean1, sigma=sigma1)
    y2 = Gaussian1.pdf(x)

    with open('1D_data.txt', 'a') as f:
        line = ''
        for a,b,c in zip(x, y1, y2):
            line += ','.join([str(s) for s in [a,b,c]]) + '\n'
        f.write(line)
    #-------------------- 生成两个正态分布的数据 --------------------
# generate_1D_data()

def generate_2D_data1():
    x, y = np.mgrid[-1:1:.01, -1:1:.01]
    pos = np.empty(x.shape + (2,))  # 从x.shape=(200,200)变为(200,200,2)
    pos[:, :, 0] = x
    pos[:, :, 1] = y
    # mean=[0.5, -0.2],cov=[[2.0, 0.3], [0.3, 0.5]]，声明一个带着指定mean和cov的rv对象
    rv0 = multivariate_normal([0.5, -0.2], [[2.0, 0.3], [0.3, 0.5]])
    z0 = rv0.pdf(pos)

    rv1 = multivariate_normal([0.3, -0.7], [[2.0, 0.3], [0.3, 0.5]])
    z1 = rv1.pdf(pos)

    with open('2D_data.txt', 'a') as f:
        line = ''
        for i in range(100):
            for j in range(100):
                a, b, c, d = x[i, j], y[i, j], z0[i, j], z1[i, j]
                line += ','.join([str(s) for s in [a, b, c, d]]) + '\n'
        f.write(line)
generate_2D_data1()

def norm1D_plot(mean=0, sigma=0.1):
    data, Gaussian = Gaussian_Distribution(N=1, M=1000, m=mean, sigma=sigma)
    x = np.linspace(-1, 1, 1000)
    # 计算一维高斯概率
    y = Gaussian.pdf(x)
    plt.plot(x, y)
    plt.show()
# norm1D_plot()

def norm2D_plot(mean=0, sigma=0.1):
    M = 1000
    data, Gaussian = Gaussian_Distribution(N=2, M=mean, sigma=sigma)
    # 生成二维网格平面
    X, Y = np.meshgrid(np.linspace(-1, 1, M), np.linspace(-1, 1, M))
    # 二维坐标数据
    d = np.dstack([X, Y])
    # 计算二维联合高斯概率
    Z = Gaussian.pdf(d).reshape(M, M)

    '''二元高斯概率分布图'''
    fig = plt.figure(figsize=(12, 8))
    ax = Axes3D(fig)
    ax.plot_surface(X, Y, Z, rstride=1, cstride=1, cmap='seismic', alpha=0.8)
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_zlabel('Z')
    plt.show()
# norm2D_plot(mean=0, sigma=0.1)