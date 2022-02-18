import numpy as np
import math
import matplotlib.pyplot as plt

def calPhase(z):
    """
    :param
        z: 3+4j
    :return:
        phase: 45°
    """
    phase = np.arctan2(z.imag, z.real) * 180 / np.pi
    return phase

def bode_two_plot(fre_arr, z_arr, z1_arr,
                  fig_title='Bode-',
                  ax1_title_list=['Bode-','1'],
                  ax2_title_list=['Bode-','2']):
    """
    同时画俩个阻抗的数据，放在一起，方便比较
    Draw Impedance Modulus |Z| and Phase in two(up and down) plot
        |z| in up plot
        Phase in down plot
    """
    fre_log10_arr = np.log10(fre_arr)

    z_norm_arr = np.array([np.sqrt(z.real ** 2 + z.imag**2) for z in z_arr])
    z_norm_log10_arr = np.log10(z_norm_arr)

    z1_norm_arr = np.array([np.sqrt(z1.real ** 2 + z1.imag**2) for z1 in z1_arr])
    z1_norm_log10_arr = np.log10(z1_norm_arr)

    # tan_arr = np.zeros(z_arr.size)
    # tan1_arr = np.zeros(z1_arr.size)
    # for i, z in enumerate(z_arr):
    #     if z.real == 0.0:
    #         tan_arr[i] = float('inf')
    #     else:
    #         tan_arr[i] = z.imag / z.real
    # for i, z1 in enumerate(z1_arr):
    #     if z1.real == 0.0:
    #         tan1_arr[i] = float('inf')
    #     else:
    #         tan1_arr[i] = z1.imag / z1.real
    """
    注意np.arctan和np.arctan2的区别
        np.arctan
            返回的角度在【-90，90】
        **np.arctan2
            返回的角度在【-180，180】
            
        Examples
        --------
        Consider four points in different quadrants:
        >>> x = np.array([-1, +1, +1, -1])
        >>> y = np.array([-1, -1, +1, +1])
        >>> np.arctan2(y, x) * 180 / np.pi
        array([-135.,  -45.,   45.,  135.])
    """
    phase_arr = np.zeros(z_arr.size)
    phase1_arr = np.zeros(z_arr.size)

    for i, z in enumerate(z_arr):
        if z.real == 0.0:
            if z.imag > 0:
                phase_arr[i] = 90
            elif z.imag == 0:
                phase_arr[i] = 0
            elif z.imag < 0:
                phase_arr[i] = -90
        else:
            phase_arr[i] = np.arctan2(z.imag, z.real) * 180 / np.pi
    for i, z1 in enumerate(z1_arr):
        if z1.real == 0.0:
            if z1.imag > 0:
                phase1_arr[i] = 90
            elif z1.imag == 0:
                phase1_arr[i] = 0
            elif z1.imag < 0:
                phase1_arr[i] = -90
        else:
            phase1_arr[i] = np.arctan2(z1.imag, z1.real) * 180 / np.pi

    fig, (ax1, ax2) = plt.subplots(2, sharex=True) # 上方的图没有X轴，和下图共享一个X轴
    # fig, (ax1, ax2) = plt.subplots(2) # 上下两张图都有各自的X轴
    fig.suptitle(fig_title)

    ax1.plot(fre_log10_arr, z_norm_log10_arr, 'o--', label=ax1_title_list[0] + '|Z| - Fre(log10 mode)')
    ax1.plot(fre_log10_arr, z1_norm_log10_arr, '*--', label=ax1_title_list[1] + '|Z| - Fre(log10 mode)')
    # ax1.plot(fre_log10_arr, z_norm_log10_arr, 'o--', label='Impedance - Frequency (log10 mode)')

    # ax1.set_xlabel(xlabel='Log$_{10}$(Frequency) [Hz]')
    ax1.set_ylabel(ylabel='Log$_{10}$(Z) [$\Omega$]')
    ax1.legend()

    ax2.plot(fre_log10_arr, phase_arr, 'o--', label=ax2_title_list[0] + 'Phase - Fre(log10 mode)')
    ax2.plot(fre_log10_arr, phase1_arr, '*--', label=ax2_title_list[1] + 'Phase - Fre(log10 mode)')

    ax2.set_xlabel(xlabel='Log$_{10}$(Frequency) [Hz]')
    ax2.set_ylabel(ylabel='Phase (deg)')
    ax2.legend()

    plt.show()

def bode_one_plot(fre_list, z_list, fig_title='Bode-'):
    """
    只能画一个阻抗的数据
    Draw Impedance Modulus |Z| and Phase in two(up and down) plot
        |z| in up plot
        Phase in down plot

    参考教程
        Creating multiple subplots using plt.subplots
        https://matplotlib.org/devdocs/gallery/subplots_axes_and_figures/subplots_demo.html
    """
    fre_log10_list = [math.log(f, 10) for f in fre_list]

    z_norm_list = [math.sqrt(z.real ** 2 + z.imag ** 2) for z in z_list]
    z_norm_log10_list = [math.log(z, 10) for z in z_norm_list]

    # tan_list = [z.imag / z.real for z in z_list] # z.real might be zero
    tan_list = []
    for z in z_list:
        if z.real == 0:
            tan_list.append(float('inf'))
        else:
            tan_list.append(z.imag / z.real)
    t_list = [math.atan(tan) * 180 / math.pi for tan in tan_list]

    fig, (ax1, ax2) = plt.subplots(2, sharex=True)
    # fig, (ax1, ax2) = plt.subplots(2)
    fig.suptitle(fig_title)

    ax1.plot(fre_log10_list, z_norm_log10_list, 'o--', label='Impedance - Frequency (log10 mode)')

    # ax1.set_xlabel(xlabel='Log$_{10}$(Frequency) [Hz]')
    ax1.set_ylabel(ylabel='Log$_{10}$(Z) [$\Omega$]')
    ax1.legend()

    ax2.plot(fre_log10_list, t_list, 'o--', label='Phase - Frequency')

    ax2.set_xlabel(xlabel='Log$_{10}$(Frequency) [Hz]')
    ax2.set_ylabel(ylabel='Phase (deg)')
    ax2.legend()

    plt.show()

def bode_absZ(fre_arr, z_arr_list, plot_type='log10', fig_title='',label_list=[]):
    """
    在同一幅Bode-|Z|上同时绘制 len(z_arr_list) 条 曲线
    :param
        fre_arr:
        z_arr_list:
        plot_type:
            'normal'
            'log10'
            'dB'
                20 * log10(K)
                K = |Z|
        fig_title:
        label_list:
    :return:
    """
    fre_log10_arr = np.log10(fre_arr)

    if plot_type == 'log10':
        z_norm_log10_arr_list = []
        for z_arr in z_arr_list:
            z_norm_arr = np.array([np.sqrt(z.real ** 2 + z.imag ** 2) for z in z_arr])
            z_norm_log10_arr = np.log10(z_norm_arr)
            z_norm_log10_arr_list.append(z_norm_log10_arr)
    elif plot_type == 'dB':
        z_norm_dB_arr_list = []
        for z_arr in z_arr_list:
            z_norm_arr = np.array([np.sqrt(z.real ** 2 + z.imag ** 2) for z in z_arr])
            z_norm_dB_arr = 20 * np.log10(z_norm_arr)
            z_norm_dB_arr_list.append(z_norm_dB_arr)

    fmt_list = ['o--', '*--', '^--']
    for i in range(len(z_arr_list)):
        if plot_type == 'normal':
            plt.plot(fre_log10_arr, z_arr_list[i], fmt_list[i], label=label_list[i])
        elif plot_type == 'log10':
            plt.plot(fre_log10_arr, z_norm_log10_arr_list[i], fmt_list[i], label=label_list[i])
        elif plot_type == 'dB':
            plt.plot(fre_log10_arr, z_norm_dB_arr_list[i], fmt_list[i], label=label_list[i])

    plt.xlabel('Log$_{10}$(Frequency) [Hz]')
    if plot_type == 'normal':
        plt.ylabel('|Z|')
    elif plot_type == 'log10':
        plt.ylabel('Log$_{10}$(Z) [$\Omega$]')
    elif plot_type == 'dB':
        plt.ylabel('dB')

    plt.title(fig_title)
    plt.legend()
    plt.show()

def bode_Phase(fre_arr, z_arr_list, fig_title='',label_list=[]):
    """
    在同一幅Bode-|Z|上同时绘制 len(z_arr_list) 条 曲线
    :param
        fre_arr:
        z_arr_list:
        plot_type:
            'normal'
            'log10'
            'dB'
                20 * log10(K)
                K = |Z|
        fig_title:
        label_list:
    :return:
    """
    fre_log10_arr = np.log10(fre_arr)

    phase_arr_list = []
    for i in range(len(z_arr_list)):
        phase_arr = np.array([np.arctan2(z.imag, z.real) * 180 / np.pi for z in z_arr_list[i]])
        phase_arr_list.append(phase_arr)

    fmt_list = ['o--', '*--', '^--']
    for i in range(len(z_arr_list)):
        plt.plot(fre_log10_arr, phase_arr_list[i], fmt_list[i], label=label_list[i])

    plt.xlabel('Log$_{10}$(Frequency) [Hz]')
    plt.ylabel('Phase(°)')

    plt.title(fig_title)
    plt.legend()
    plt.show()