# -*-coding:utf-8 -*-

"""
Function
    将EIS插值到80个点
Routine
    80/20           4       每两个点之间3=57   77  80-77  第二阶段方法
    80/(21-26)      3       每两个点之间2
    80/(27-40)      2       每两个点之间1
    80/(41-60)      1       第二阶段方法
version:
    0
        金阳写的太杂乱
        First author: Zhao JinYang
        Second author: Zhao ZhaoYang
"""

import os
import xlwt
import numpy as np
from scipy import interpolate
import pylab as pl

# 定位到文件所在位置
a = os.getcwd()  # 获取当前目录
print(a)  # 打印当前目录
os.chdir('D:\Interpolation')  # 定位到新的目录Interpolation
a = os.getcwd()  # 获取定位之后的目录

x = []
y = []
z = []
x_new = []
x_nn = []
with open('Interpolation' + '.txt', 'r', encoding='UTF-8') as fd:
    for text in fd.readlines():
        a = text.split(',')[0]
        b = text.split(',')[1]
        x.append(float(a))
        y.append(float(b))

import copy
x0 = copy.deepcopy(x)

# print(x)
# print(y)
print(len(x))

f = interpolate.interp1d(x, y, kind="quadratic")
i = 0
j = int(len(x))
# mid = int(j / 2)
z = int(80/len(x))
if z == 2:
    for i in range(len(x)-1):
        u = (x[i]+x[i+1])/2
        x.insert((2 * i + 1), float(u))
        i = i+1
        x_new.append(float(u))
    i = 0
    x_new = []
    for i in range(80-len(x)):
        v = (x[i]+x[i+1])/2
        x.insert((2 * i + 1), float(v))
        i = i+1
        x_new.append(float(v))
elif z == 1:
    for i in range(80-len(x)):
        u = (x[i] + x[i + 1]) / 2
        x.insert((2 * i + 1), float(u))
        i = i + 1
        x_new.append(float(u))
elif z == 3:
    for i in range(len(x)-1):
        u = (x[i]+x[i+1])/3
        v = 2*(x[i] + x[i + 1])/3
        x.insert((2 * i + 1), float(u))
        x.insert((2 * i + 2), float(v))
        i = i+2
        x_new.append(float(u))
    i = 0
    x_new = []
    for i in range(80-len(x)):
        v = (x[i]+x[i+1])/2
        x.insert((2 * i + 1), float(v))
        i = i+1
        x_new.append(float(v))
elif z == 4:
    for i in range(len(x) - 1):
        u = (x[i] + x[i + 1]) / 4
        v = 2 * (x[i] + x[i + 1]) / 3
        w = 3 * (x[i] + x[i + 1]) / 3
        x.insert((2 * i + 1), float(u))
        x.insert((2 * i + 2), float(v))
        x.insert((2 * i + 3), float(w))
        i = i + 3
        x_new.append(float(u))
    i = 0
    x_new = []
    for i in range(80 - len(x)):
        v = (x[i] + x[i + 1]) / 2
        x.insert((2 * i + 1), float(v))
        i = i + 1
        x_new.append(float(v))

print(x_new)
print(x)
print(len(x))
z = f(x)
print(z)
print(len(z))

# x = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
# y = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# pl.plot(x0, y, label=str("Raw Data"))
# pl.scatter(x0, y, label=str("Raw Data"))
# pl.legend()
# pl.show()
# xnew = np.linspace(0, 10, 101)
# for kind in ["nearest","zero","slinear","quadratic","cubic"]:#插值方式
#     #"nearest","zero"为阶梯插值
#     #slinear 线性插值
#     #"quadratic","cubic" 为2阶、3阶B样条曲线插值
#     f =interpolate.interp1d(x,y,kind=kind)
#     # ‘slinear’, ‘quadratic’ and ‘cubic’ refer to a spline interpolation of first, second or third order)
#     ynew=f(xnew)
#     pl.plot(xnew,ynew,label=str(kind))


# x_new = np.linspace(0, 10, 101)
# f = interpolate.interp1d(x, y, kind="cubic")
# y_new = f(x_new)
# pl.plot(x_new, y_new, label=str("cubic"))

print(z)
pl.scatter(x0, y, marker='v',label=str("Raw Data"))
pl.legend()
pl.scatter(x, z, marker='o', label=str("cubic"))
pl.legend(loc="lower right")
pl.show()