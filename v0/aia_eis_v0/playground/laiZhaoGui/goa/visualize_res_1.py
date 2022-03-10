"""
模块功能：
    集合 Lai-EIS 人工处理结果 画图

    version
        1：
        0：visualize_res_0.py
"""

# 从20种GOA为ECM2 or ECM9 的拟合挑选出 最合适的 4 or 5个GOA，画图看看每个GOA对同一个 实验EIS的拟合结果
# 由于有三种异常点检测的指标：Ny-Im，Bd-|Z|，Bd-Phase