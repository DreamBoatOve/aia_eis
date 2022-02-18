import copy
from scipy import interpolate

def eis_interpolater(z_real_list, z_imag_list, interpolate_kind='quadratic', total_points=80):
    """
    Function
        将EIS插值到80个点
    Routine
        80/20           4       每两个点之间3=57   77  80-77  第二阶段方法
        80/(21-26)      3       每两个点之间2
        80/(27-40)      2       每两个点之间1
        80/(41-60)      1       第二阶段方法
    :param
        z_real_list
            [float]
        z_imag_list
            [float]
        interpolate_kind
            插值函数的种类
        z_list: [(z_r, z_img(带原始数据的负号))]
    :return:

    version:
        1
            对version-0进行整理和函数封装
            author: Zhao ZhaoYang
        0
            金阳写的太杂乱
            First author: Zhao JinYang
            Second author: Zhao ZhaoYang
    """
    x, y = z_real_list, z_imag_list
    interpolate_fun = interpolate.interp1d(x, y, kind=interpolate_kind)

    # 这段代码是在原始的x中插入新增的若干个xi，增加到80个x
    x_new = []
    z = int(total_points / len(x))

    if z == 1:
        x1 = copy.deepcopy(x)
        x2 = []
        for i in range(total_points - len(x)):
            x2.append((x1[i] + x1[i+1]) / 2)
        for a, b in zip(x1[: len(x2)], x2):
            x_new.extend([a,b])
        x_new.extend(x1[len(x2):])

    elif z == 2:
        x1 = copy.deepcopy(x)
        x2 = []
        for i in range(len(x1) - 1):
            x2.append((x1[i] + x1[i+1]) / 2)
        for a, b in zip(x1[:-1], x2):
            x_new.extend([a, b])

        x3 = []
        for i in range(total_points - len(x_new)):
            x3.append((x_new[i] + x_new[i+1]) / 2)
        x4 = []
        for c,d in zip(x_new[: len(x3)], x3):
            x4.extend([c,d])
        x_new = x4 + x_new[len(x3) : ]

    elif z == 3:
        x1 = []
        for i in range(len(x)-1):
            a = x[i] + (x[i+1] - x[i]) * 1 / 3
            b = x[i] + (x[i+1] - x[i]) * 2 / 3
            x1.append([a,b])

        x2 = []
        for c, d in zip(x, x1):
            x2.extend([c, d[0], d[1]])

        x3 = []
        for i in range(total_points - len(x2)):
            x3.append((x2[i] + x2[i+1]) / 2)

        x4 = []
        for q, w in zip(x2[: len(x3)], x3):
            x4.extend([q,w])
        x_new = x4 + x2[len(x3) :]

    y_new = interpolate_fun(x_new)
    return x_new, y_new