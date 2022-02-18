import numpy as np
import matplotlib.pyplot as plt
from scipy.optimize import leastsq

"""
主要参考 
    在python中利用最小二乘拟合三次抛物线函数的方法
    https://www.jb51.net/article/153711.htm
"""

# 三次函数的标准形式 f(x) = A * x * x * x + B * x * x + C * x + D
def func(params, x):
    a, b, c, d = params
    return a * x * x * x + b * x * x + c * x + d

# 误差函数，即拟合曲线所求的值与实际值的差
def error(params, x, y):
    return func(params, x) - y

# 对参数求解
def slovePara(x_arr, y_arr):
    p0 = [1, 1, 1, 1]
    Para = leastsq(error, p0, args=(x_arr, y_arr))
    return Para

def cub_fun_fit(x_list, y_list):
    para = slovePara(np.array(x_list), np.array(y_list))
    return para

def cubic_test():
    # Target fun = 2 * x**3 + 2 * x**2 + 3 * x + 1
    x_list = [1,  2,    3,  4]
    y_list = [8,  31,   82, 170]
    plt.figure(figsize=(8, 6))
    plt.scatter(x_list, y_list, color="green", label="sample data", linewidth=2)

    paras = cub_fun_fit(x_list, y_list)
    print(paras)

    x_plot = [6 * i / 1000.0 for i in range(1000)]
    y_plot = [func(paras[0], x) for x in x_plot]

    plt.plot(x_plot, y_plot, color="red", label="solution line", linewidth=2)
    plt.legend()  # 绘制图例
    plt.show()

# if __name__ == '__main__':
#     cubic_test()