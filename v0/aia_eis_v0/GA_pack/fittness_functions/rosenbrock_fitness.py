def rosenbrock(x1, x2):
    return 100 * ((x1 ** 2 - x2) ** 2) + (1 - x1) ** 2

class rosenbrock():
    """
    rosenbrock:
        function: f(x1, x2) = 100*(x1^2 - x2)^2 + (1-x1)^2
        target value: maximum
        variable range: -2.048 <= x1, x2 <= 2.048
        maximum points:
            f(2.048, -2.048) = 3897.7342
            **f(-2.048, -2.048) = 3905.9262
        Find the MAXIMUM value of rosenbrock function
    """
    def __init__(self, x1, x2):
        self.x1 = x1
        self.x2 = x2
    def get_rosenbrock_fitness(self):
        return 100*((self.x1**2 - self.x2)**2) + (1 - self.x1)**2