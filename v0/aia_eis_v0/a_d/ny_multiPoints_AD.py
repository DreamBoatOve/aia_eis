import copy

from IS.IS import IS_0
from ny_onePoint_AD import load_EIS, getMaxAndIndex, detect

"""
Multi Outlier Detection
    Two Outlier Detection
        Routine of 双点异常实验
            计算原始EIS的 Vogit残差
            1- SM先拟合一条，计算SM残差
            2- 按照SM残差从大到小的顺序 依次尝试，找出第一个最可能的异常点
                2.1- 按照Index每次去掉一个点，计算去掉一个点后的EIS的Vogit残差
        
            计算删掉一个点后的EIS的Vogit残差
            再次进行SM，计算SM残差
            2- 按照SM残差从大到小的顺序 依次尝试，找出第一个最可能的异常点
            2.1- 按照Index每次去掉一个点，计算去掉一个点后的EIS的Vogit残差
        
        Note
            位置组合
            Outlier index:
                High=5/middle=25/low=55
                Fre Index = 5 + 25
                    disturb degree
                        High + High
                        High + Mid
                        Mid + High
                        Mid + Mid
                5 + 55
                    High + High
                    High + Mid
                    Mid + High
                    Mid + Mid
                25 + 55
                    High + High
                    High + Mid
                    Mid + High
                    Mid + Mid
        
            扰动程度组合；
                高/中，由于【低程度扰动】近似噪声，在单点异常实验中基本检测不出来，所以不考虑
    Three Outlier Detection
        Routine of 3点异常实验
            1- SM先拟合一条，计算残差，找出最可能的异常点
            2- 按照残差从大到小的顺序 依次尝试
                2.1- 每次去掉一个点
        
        Note
            位置组合
            Outlier index:
                High=5/middle=25/low=55
                5 + 25 + 55
                    High + High + High
                    High + High + Mid
                    High + Mid + High
                    High + Mid + Mid
        
                    Mid + High + High
                    Mid + High + Mid
                    Mid + Mid + High
                    Mid + Mid + Mid
        
            扰动程度组合；
                高/中，由于【低程度扰动】近似噪声，在单点异常实验中基本检测不出来，所以不考虑
"""


def deletedPointIndexSorter(deletedPointIndex_list, index):
    """
    Function
        将index与deletedPointIndex_list中的元素e逐个进行对比，
        例如deletedPointIndex_list = 【4，5，6，7】
            if index = 3
                3<4 --> [3,4,5,6,7]
            if index = 5
                5>4, index += 1 ==> 6
                6>5, index += 1 ==> 7
                7>6, index += 1 ==> 8 --> [4,5,6,7,8]
    :param deletedPointIndex_list:
    :param index:
    :return:
    """
    if len(deletedPointIndex_list) == 0:
        deletedPointIndex_list.append(index)
        return deletedPointIndex_list

    if index > deletedPointIndex_list[-1]:
        deletedPointIndex_list.append(index)
        return deletedPointIndex_list

    # for i in sorted(deletedPointIndex_list):
    for i, pointIndex in enumerate(deletedPointIndex_list):
        if (i == 0) and (index < pointIndex):
            deletedPointIndex_list.insert(0, index)
            return deletedPointIndex_list

        elif index >= pointIndex:
            index += 1
            if i == len(deletedPointIndex_list) - 1:
                deletedPointIndex_list.append(index)
                return deletedPointIndex_list
            elif index < deletedPointIndex_list[i + 1]:
                deletedPointIndex_list.insert(i, index)
                return deletedPointIndex_list


# res = deletedPointIndexSorter(deletedPointIndex_list=[4,5,6,7,10],
#                               # index=3)
#                                index=7)
# print(res)

def detectMultiPoints(eis_source, criteria_type='absRealAndAbsImag', qua_flag=False, pointNum=3, Q_flag='Q3'):
    if isinstance(eis_source, str):
        if pointNum == 2:
            eis_source = load_EIS(fn=eis_source,
                                  fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)(RW)_pkl/twoOutliers/')
        elif pointNum == 3:
            eis_source = load_EIS(fn=eis_source,
                                  fp='../plugins_test/jupyter_code/rbp_files/0/R(RC)(RW)_pkl/threeOutliers/')
    elif isinstance(eis_source, IS_0):
        eis_source = eis_source
    else:
        import sys
        print('detect中ecm_fn的参数给错了，查查')
        sys.exit()

    count = 0
    chiSquare = 1.0
    deletedPointIndex_list = []
    # global deletedPointIndex_list
    tmp_eis_source = copy.deepcopy(eis_source)
    while (count < pointNum) or (chiSquare > 1e-5):
        deletedPointIndex, chiSquare = detect(tmp_eis_source, criteria_type, qua_flag, Q_flag)
        print('--------------\nThe index of the {0} outlier is: {1}'.format(count, deletedPointIndex))
        deletedPointIndex_list = deletedPointIndexSorter(deletedPointIndex_list, deletedPointIndex)

        # deletedPointIndex_list.append(deletedPointIndex)
        # sortedDeletedPointIndex_list = sorted(deletedPointIndex_list)

        tmp_eis_source = copy.deepcopy(eis_source)
        for dpi in sorted(deletedPointIndex_list, reverse=True):  # dpi == deletedPointIndex
            # for dpi in sortedDeletedPointIndex_list: # dpi == deletedPointIndex
            tmp_eis_source.removeZByIndex(index=dpi)

        # eis_source.removeZByIndex(index=deletedPointIndex)

        count += 1


# Two Outlier Detection
# detectMultiPoints(eis_source='2021_10_13_R(RC)(RW)_MLF_HM_ecm.pkl',
                  # criteria_type='absRe',
                  # criteria_type='absIm',
                  # criteria_type='absRealAndAbsImag',
                  # criteria_type='absResidual',
                  # -------------
                  # qua_flag=False, pointNum=2)
                  # qua_flag=True, pointNum=2, Q_flag='Q3')
                  # qua_flag=True, pointNum=2, Q_flag='Q2')

# Three Outlier Detection
detectMultiPoints(eis_source='2021_10_13_R(RC)(RW)_HMH_ecm.pkl',
                    # criteria_type='absRe',
                    # criteria_type='absIm',
                    # criteria_type='absRealAndAbsImag',
                    criteria_type='absResidual',
                    # -------------
                    # qua_flag=False, pointNum=3)
                    # qua_flag=True, pointNum=3, Q_flag='Q3')
                    qua_flag=True, pointNum=3, Q_flag='Q2')
