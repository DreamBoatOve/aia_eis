"""
Python 算例实现Levenberg-Marquardt算法
    https://blog.csdn.net/Huangyuliang1992/article/details/79033142#commentBox
用的是 python-2
"""

# -*- coding:utf-8 -*-
# autor:HuangYuliang
import numpy as np
from numpy import matrix as mat
from matplotlib import pyplot as plt
import random

n = 100
a1, b1, c1 = 1, 3, 2  # 这个是需要拟合的函数y(x) 的真实参数
h = np.linspace(0, 1, n)  # 产生包含噪声的数据
y = [np.exp(a1 * i ** 2 + b1 * i + c1) + random.gauss(0, 4) for i in h]
y = mat(y)  # 转变为矩阵形式


def Func(abc, iput):  # 需要拟合的函数，abc是包含三个参数的一个矩阵[[a],[b],[c]]
    a = abc[0, 0]
    b = abc[1, 0]
    c = abc[2, 0]
    return np.exp(a * iput ** 2 + b * iput + c)


def Deriv(abc, iput, n):  # 对函数求偏导
    x1 = abc.copy()
    x2 = abc.copy()
    x1[n, 0] -= 0.000001
    x2[n, 0] += 0.000001
    p1 = Func(x1, iput)
    p2 = Func(x2, iput)
    d = (p2 - p1) * 1.0 / (0.000002)
    return d


J = mat(np.zeros((n, 3)))  # 雅克比矩阵
fx = mat(np.zeros((n, 1)))  # f(x)  100*1  误差
fx_tmp = mat(np.zeros((n, 1)))
xk = mat([[0.8], [2.7], [1.5]])  # 参数初始化
lase_mse = 0
step = 0
u, v = 1, 2
conve = 100

while (conve):

    mse, mse_tmp = 0, 0
    step += 1
    for i in range(n):
        fx[i] = Func(xk, h[i]) - y[0, i]  # 注意不能写成  y - Func  ,否则发散
        mse += fx[i, 0] ** 2

        for j in range(3):
            J[i, j] = Deriv(xk, h[i], j)  # 数值求导
    mse /= n  # 范围约束

    H = J.T * J + u * np.eye(3)  # 3*3
    dx = -H.I * J.T * fx  # 注意这里有一个负号，和fx = Func - y的符号要对应
    xk_tmp = xk.copy()
    xk_tmp += dx

    for j in range(n):
        fx_tmp[i] = Func(xk_tmp, h[i]) - y[0, i]
        mse_tmp += fx_tmp[i, 0] ** 2
    mse_tmp /= n

    q = (mse - mse_tmp) / ((0.5 * dx.T * (u * dx - J.T * fx))[0, 0])

    if q > 0:
        s = 1.0 / 3.0
        v = 2
        mse = mse_tmp
        xk = xk_tmp
        temp = 1 - pow(2 * q - 1, 3)

        if s > temp:
            u = u * s
        else:
            u = u * temp
    else:
        u = u * v
        v = 2 * v
        xk = xk_tmp

    print
    "step = %d,abs(mse-lase_mse) = %.8f" % (step, abs(mse - lase_mse))
    if abs(mse - lase_mse) < 0.000001:
        break

    lase_mse = mse  # 记录上一个 mse 的位置
    conve -= 1
print
xk

z = [Func(xk, i) for i in h]  # 用拟合好的参数画图

plt.figure(0)
plt.scatter(h, y, s=4)
plt.plot(h,z,'r')
plt.show()