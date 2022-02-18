"""
f5 is taken from paper <Grey wolf Optimizer>-Unimodal benchmark functions-f5
"""
def f5(x_list):
    """
    :param
        x_list:
            list(float, x0, x1, x2, ...)
            -30 <= xi <= 30
    :return:
        f
            float
    """
    f = 0.0
    for i in range(len(x_list) - 1):
        x1 = x_list[i]
        x2 = x_list[i+1]
        f += 100 * ((x2 - x1 ** 2) ** 2) + (x1 - 1) ** 2
    return f