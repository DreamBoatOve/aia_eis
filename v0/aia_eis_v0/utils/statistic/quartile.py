import math
import numpy as np

def cal_quartile(arr):
    # 计算四分位的各个数值
    sortedArr = np.sort(arr)  # small --> big
    num = arr.size

    Q1_index = (num + 1) / 4
    if (num + 1) % 4 == 0:
        Q1 = sortedArr[int(Q1_index)]
    else:
        decimal_part = Q1_index - math.floor(Q1_index)
        if decimal_part < 0.5:
            Q1 = 0.75 * sortedArr[int(Q1_index)] + 0.25 * sortedArr[int(Q1_index) + 1]
        else:
            Q1 = 0.25 * sortedArr[int(Q1_index)] + 0.75 * sortedArr[int(Q1_index) + 1]

    Q2_index = (num + 1) / 2
    if (num + 1) % 2 == 0:
        Q2 = sortedArr[int(Q2_index)]
    else:
        decimal_part = Q2_index - math.floor(Q2_index)
        if decimal_part < 0.5:
            Q2 = 0.75 * sortedArr[int(Q2_index)] + 0.25 * sortedArr[int(Q2_index) + 1]
        else:
            Q2 = 0.25 * sortedArr[int(Q2_index)] + 0.75 * sortedArr[int(Q2_index) + 1]

    Q3_index = (num + 1) * 3 / 4
    if (num + 1) * 3 % 4 == 0:
        Q3 = sortedArr[int(Q3_index)]
    else:
        decimal_part = Q3_index - math.floor(Q3_index)
        if decimal_part < 0.5:
            Q3 = 0.75 * sortedArr[int(Q3_index)] + 0.25 * sortedArr[int(Q3_index) + 1]
        else:
            Q3 = 0.25 * sortedArr[int(Q3_index)] + 0.75 * sortedArr[int(Q3_index) + 1]

    # IQR: (Inter-Quartile Range) 在统计中叫内距.内距又称为四分位差
    IQR = Q3 - Q1

    up_boundary = Q3 + 1.5 * IQR
    low_boundary = Q1 - 1.5 * IQR
    return [low_boundary, Q1, Q2, Q3, up_boundary]