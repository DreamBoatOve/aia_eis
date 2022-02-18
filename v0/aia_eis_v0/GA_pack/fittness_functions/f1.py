# X = [x0, x1, x2, ..., xn], -100 <= xi <= 100
# f1(X) = SUM(x0^2 + x1^2 + ... + xn^2)
def f1(x_list):
    x_sum = 0.0
    for x in x_list:
        x_sum += pow(x, 2)
    return x_sum