import sys
import math

"""
Construct all the circuit elements used in the project
    Capacitor   C
    Inductance  L
    Warburg     War 以便和频率w区分
    Constant-phase element CPE (Q)
"""

def ele_C(w, C):
    """
    :param
        w: Angular frequency [1/s], (s:second)
        C: F
    :return:
    """
    return 1 / (1j * w * C)

def ele_L(w, L):
    """
    :param
        w: Angular frequency [1/s], (s:second)
        L: Inductance [ohm * s]
    :return:
    """
    return 1j * w * L

def ele_Warburg(w, sigma):
    """
    :param
        w: Angular frequency [1/s], (s:second)
        The first expression:
            Warburg = σ * (w^(-0.5)) * (1 - 1j)
            sigma: warburg coefficient, no unit

        The second expression:
            refer:
                PAPER:
                    100-173-053-Preparation of carbon-coated lithium iron phosphate/titanium nitride for a lithium-ion
                    supercapacitor, Eq-3
            Warburg = W_R * coth[(1j * W_T * w)^W_P] / (1j * W_T * w)^W_P
            if we take z = (1j * W_T * w)^W_P:
            then, the equation can be simplified as Warburg = W_R * coth[z] / z

            W_R: Diffusion resistance (Warburg diffusion resistance)
            W_T: Diffusion time constant, is equal to (L^2)/D
                L and D are the effective diffusion length and diffusion coefficient, respectively
            W_P: is a fractional exponent between 0 and 1.
            coth: hyperbolic cotangent
                webs:
                    coth: https://www.mathworks.com/help/matlab/ref/coth.html
                    Hyperbolic Cotangent: https://mathworld.wolfram.com/HyperbolicCotangent.html
                coth(z) = (e^z + e^(-z)) / (e^z - e^(-z)) = (e^2z + 1) / (e^2z - 1)
    :return:
        Zwar: Warburg impedance
    """
    def coth(z):
        """
        coth 双曲余切函数
        问题：math.e ** (2 * z) ==> OverflowError: complex exponentiation
            当z较大时，会出现计算上溢，经简单实验，math.e ** (500 + 500 j)就会出现上溢

            解决方案1：Negative
                coth(x) = (e ** x + e ** (-x)) / (e ** x - e ** (-x))
                        = (e ** 2x + 1) / (e ** 2x - 1)
                when x --> +Inf, e ** 2x --> +Inf:
                    coth(x) = (Inf + 1) / (Inf - 1)
                            = Inf / Inf
                            = 1
                when x --> -Inf, e ** 2x --> 0
                    coth(x) = (0 + 1) / (0 - 1)
                            = +1 / -1
                            = -1
            解决方案2:
                Use Mpmath package, it is a Python library for arbitrary-precision floating-point arithmetic
                import mpmath as mp
                mp.coth(x), x can be float, int, complex
        """
        # c = (math.e ** (2 * z) + 1) / (math.e ** (2 * z) - 1)
        import mpmath as mp
        c = mp.coth(z)
        return c

    if (type(sigma) == int) or (type(sigma) == float):
        Zwar = sigma * (w ** (-0.5)) * (1 - 1j)
    elif len(sigma) == 3:
        Wr, Wt, Wp = sigma
        # z = (1j * W_T * w)^W_P
        z = (1j * Wt * w) ** Wp
        Zwar = Wr * coth(z) / z
    else:
        print("There is something wrong in your Warburg parameters, check it")
        sys.exit(0)
    return Zwar

def ele_Q(w, q, n):
    """
    :param
        w: Angular frequency [1/s], (s:second)
        q: CPE coefficient, Constant phase element [s^n/ohm]
            or named as CPE_T
        n: Constant phase elelment exponent [-]
            or name as CPE_P
    :return:
        Zcpe: Impedance of a Constant phase element
    """
    z = 1 / (q * ((1j * w) ** n))
    return z

def ele_O(w):
    pass

def ele_T(w):
    pass

def ele_S(w):
    pass