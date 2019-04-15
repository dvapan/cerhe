from functools import lru_cache, reduce
from operator import mul

import scipy as sc

class Polynom:
    def __init__(self, count_var, degree):
        self.count_var = count_var
        self.degree = degree
        self.coeff_size = len(pow_indeces(count_var, degree))
        self.coeffs = sc.zeros(self.coeff_size)
        self.var_coeffs = None

    def fx(self, x, deriv=None):
        if deriv is None:
            deriv = sc.zeros_like(x)
        self.var_coeffs = get_deriv_poly_var_coeffs(self.count_var, self.degree, x, deriv)
        return sc.dot(self.var_coeffs, self.coeffs)

class PolynomExpression:
    pass

@lru_cache()
def pow_indeces(vars_count, degree):
    """ Function returns power indeces for variables in polynom.
    """
    max_num = degree * 10**(vars_count - 1)
    nums = filter(lambda x: nsum(x) <= degree, range(max_num + 1))
    fmt_str = '{:0>' + str(vars_count) + '}'

    def row_powers(x):
        return [int(el) for el in fmt_str.format(x)]
    return sc.array(list(map(row_powers, nums)))


def nsum(x):
    return sum(map(int, str(x)))


def construct_element(x, pow_i, num_deriv):
    dpow_i = pow_i - num_deriv
    if dpow_i < 0:
        return 0
    coeff = fact_div(pow_i, dpow_i)
    return coeff * x**dpow_i

@lru_cache()
def fact_div(a, b):
    return reduce(mul, range(b+1, a+1), 1)

def get_deriv_poly_var_coeffs(count_var, degree, x, deriv):
    pow_i = pow_indeces(count_var, degree)

    def prod(pows):
        return reduce(mul, map(construct_element, x, pows, deriv), 1)
    return sc.fromiter(map(prod, pow_i), float)


def poly_fx(count_var, degree, x, coeffs, deriv=None, shift=0):
    if deriv is None:
        deriv = sc.zeros_like(x)
    var_coeffs = get_deriv_poly_var_coeffs(count_var, degree, x, deriv)
    return var_coeffs, sc.dot(var_coeffs, coeffs)


