import numpy as np

def savitzkyGolay(z_arr, loop_time, convPoints):
    """
    Function
        对Z_arr进行简单的平滑得到Z_smooth_arr
    :param
        z_arr:
        loop_time:
        convPoints:
    :return:
        Z_smooth_arr
    refer
        paper0：Smoothing and Differentiation of Data by Simplified Least Squares Procedures
    """
    # ComplexWarning: Casting complex values to real discards the imaginary part, 这样arr中的dtype只能是z的实部，虚部被舍弃
    # z_smoothed_arr = np.zeros(shape=(z_arr.size,))
    z_smoothed_arr = np.empty(shape=(z_arr.size,), dtype=complex)
    for t in range(loop_time):
        if convPoints == 5:
            # 5--(-3,12,17,12,-3)/35
            z_smoothed_arr[:2] = z_arr[:2]
            start_i = 2
            for i in range(start_i, z_arr.size-2):
                z = -3 * z_arr[i-2] + 12 * z_arr[i-1] + 17 * z_arr[i] + 12 * z_arr[i+1] - 3 * z_arr[i+2]
                z_smoothed_arr[i] = z/35
            z_smoothed_arr[-2:] = z_arr[-2:]

        elif convPoints == 7:
            # 7--(-2,3,6,7,6,3,-2)/21
            z_smoothed_arr[:3] = z_arr[:3]
            start_i = 3
            for i in range(start_i, z_arr.size-3):
                z = -2 * z_arr[i-3] + 3 * z_arr[i-2] + 6 * z_arr[i-1] + 7 * z_arr[i] + 6 * z_arr[i+1] + 3 * z_arr[i+2] - 2 * z_arr[i+3]
                z_smoothed_arr[i] = z/21
            z_smoothed_arr[-3:] = z_arr[-3:]

        elif convPoints == 9:
            # 9--(-21,14,39,54,59,54,39,14,-21)/231
            z_smoothed_arr[:4] = z_arr[:4]
            start_i = 4
            for i in range(start_i, z_arr.size-4):
                z = -21*z_arr[i-4] + 14*z_arr[i-3] + 39 * z_arr[i-2] + 54 * z_arr[i-1] + 59 * z_arr[i] + 54 * z_arr[i+1] + 39 * z_arr[i+2] + 14 * z_arr[i+3] - 21 * z_arr[i+4]
                z_smoothed_arr[i] = z/231

            z_smoothed_arr[-4:] = z_arr[-4:]

        elif convPoints == 11:
            # 11--(-36,9,44,69,84,89,84,69,44,9,-36)/429
            z_smoothed_arr[:5] = z_arr[:5]
            start_i = 5
            for i in range(start_i, z_arr.size-5):
                z = -36 * z_arr[i-5] + 9 * z_arr[i-4] + 44 * z_arr[i-3] + 69* z_arr[i-2] + 84 * z_arr[i-1] + 89 * z_arr[i] + 84 * z_arr[i+1] + 69 * z_arr[i+2] + 44*z_arr[i+3] + 9*z_arr[i+4] -36*z_arr[i+5]
                z_smoothed_arr[i] = z/429
            z_smoothed_arr[-5:] = z_arr[-5:]

    return z_smoothed_arr