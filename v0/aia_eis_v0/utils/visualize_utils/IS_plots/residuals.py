import math
import numpy as np
import matplotlib.pyplot as plt

def imag_residual_plot(f, res_imag_arr, fmt='.-', y_limits=(-5, 5)):
    plt.plot(f, res_imag_arr * 100, fmt, label=r'$\Delta_{\,\mathrm{Re}}$')

    # Make x axis log scale
    plt.xscale('log')

    # Set the labels to delta vs f
    plt.xlabel('$f$ [Hz]', fontsize=14)
    plt.ylabel('$\\Delta$ $(\\%)$', fontsize=14)

    plt.legend()
    plt.xlim(min(f), max(f))
    plt.ylim(y_limits)
    plt.show()

def real_residual_plot(f, res_real_arr, fmt='.-', y_limits=(-5, 5)):
    plt.plot(f, res_real_arr * 100, fmt, label=r'$\Delta_{\,\mathrm{Re}}$')

    # Make x axis log scale
    plt.xscale('log')

    # Set the labels to delta vs f
    plt.xlabel('$f$ [Hz]', fontsize=14)
    plt.ylabel('$\\Delta$ $(\\%)$', fontsize=14)

    plt.legend()
    plt.xlim(min(f), max(f))
    plt.ylim(y_limits)
    plt.show()

def residuals_plot(f, residual_arr=None, residual_real_arr=None, residual_imag_arr=None, fmt='.-', y_limits=None):
    """
    refer: Impedance-->visualization.py-->plot_residuals(ax, f, res_real, res_imag, fmt='.-', y_limits=(-5, 5), **kwargs):
    :return:
    """
    if residual_arr is not None:
        plt.plot(f, residual_arr.real * 100, fmt, label=r'$\Delta_{\,\mathrm{Re}}$')
        plt.plot(f, residual_arr.imag * 100, fmt, label=r'$\Delta_{\,\mathrm{Im}}$')

        y_abs_max = max(max(np.abs(residual_arr.real * 100)), max(np.abs(residual_arr.imag * 100)))
    elif (residual_real_arr is not None) and (residual_imag_arr is not None):
        plt.plot(f, residual_real_arr * 100, fmt, label=r'$\Delta_{\,\mathrm{Re}}$')
        plt.plot(f, residual_imag_arr * 100, fmt, label=r'$\Delta_{\,\mathrm{Im}}$')

        y_abs_max = max(max(np.abs(residual_real_arr * 100)), max(np.abs(residual_imag_arr * 100)))

    # Make x axis log scale
    plt.xscale('log')

    # Set the labels to delta vs f
    plt.xlabel('$f$ [Hz]', fontsize=14)
    plt.ylabel('$\\Delta$ $(\\%)$', fontsize=14)

    plt.legend()
    plt.xlim(min(f) / 2, max(f) * 2)

    if y_abs_max < 5:
        y_limits = [-5, 5]
    else:
        y_limits = [-1.5 * y_abs_max, 1.5 * y_abs_max]
    plt.ylim(y_limits)

    # if y_limits is not None:
    #     # y_limits = [-5, 5]
    #     plt.ylim(y_limits)
    # else:
    #     if y_abs_max < 5:
    #         y_limits = [-5, 5]
    #     else:
    #         y_limits = [-1.5 * y_abs_max, 1.5 * y_abs_max]
    #     plt.ylim(y_limits)
    plt.show()
