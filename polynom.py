from functools import lru_cache
import scipy as sc


class Context:
    def __init__(self):
        self.count = 0
        self.variables = dict()
        self.expr = lambda x, y: 0

    def assign(self, variable):
        self.variables[variable] = self.count
        variable.owner = self
        self.count += 1

    def index_of(self, var):
        return self.variables.get(var)

    def eval(self, x, deriv=None):
        return self.expr(x, deriv)


class Polynom:
    def __init__(self, count_var, degree):
        self.count_var = count_var
        self.degree = degree
        self.coeff_size = len(pow_indices(count_var, degree))
        self.coeffs = sc.zeros(self.coeff_size)
        self.var_coeffs = None
        self.owner = None

    def fx(self, x, deriv=None):
        if deriv is None:
            deriv = sc.zeros_like(x[0])
        self.var_coeffs = get_deriv_poly_var_coeffs(self.count_var,
                                                    self.degree, x, deriv)
        return sc.dot(self.var_coeffs, self.coeffs)

    def __call__(self, x, deriv=None):
        return self.fx_var(x, deriv)

    def fx_var(self, x, deriv=None):
        val = self.fx(x, deriv).reshape(-1, 1)
        if self.owner is None:
            return sc.hstack([val, self.var_coeffs])
        else:
            count_exprs = len(val)
            size = self.coeff_size*self.owner.count
            shift = self.coeff_size*self.owner.index_of(self)
            lzeros = sc.zeros((count_exprs, shift))
            rzeros = sc.zeros((count_exprs, size - self.coeff_size - shift))
            return sc.hstack([val, lzeros, self.var_coeffs, rzeros])


@lru_cache()
def pow_indices(vars_count, degree):
    """ Function returns power indices for variables in polynom.
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
    filtered = dpow_i
    dpow_i = sc.piecewise(dpow_i,
                          [dpow_i >= 0, dpow_i < 0],
                          [lambda x: x, 0])
    coeff = vfact_div(pow_i, filtered)
    return coeff * x**dpow_i


@lru_cache()
def fact_div(a, b):
    if b >= 0:
        return sc.multiply.reduce(sc.arange(b+1, a+1))
    else:
        return 0


vfact_div = sc.vectorize(fact_div)


def get_deriv_poly_var_coeffs(count_var, degree, x, deriv):
    pow_i = pow_indices(count_var, degree)
    return sc.array(
        [sc.multiply.reduce(construct_element(xi, pow_i, deriv).transpose())
         for xi in x])
