from functools import lru_cache
import scipy as sc


class Context:
    def __init__(self):
        self.count = 0
        self.dvariables = dict()
        self.lvariables = list()
        self.expr = lambda x, y: 0

    def assign(self, variable):
        self.dvariables[variable] = self.count
        self.lvariables.append(variable)
        variable.owner = self
        self.count += 1

    def index_of(self, var):
        return self.dvariables.get(var)

    def at(self, var_ind):
        return self.lvariables[var_ind]

    def eval(self, x, deriv=None):
        return self.expr(x, deriv)


class Polynom:
    def __init__(self, count_var, degree, val=0):
        self.__name__ = "polynom"
        self.count_var = count_var
        self.degree = degree
        self.coeff_size = len(pow_indices(count_var, degree))
        self.coeffs = sc.zeros(self.coeff_size)
        self.coeffs[0] = val
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
        val = self.fx(x, deriv)
        if self.owner is None:
            return sc.hstack([val, self.var_coeffs])
        else:
            cff_cnt = [v.coeff_size for v in self.owner.lvariables]
            size = sum(cff_cnt)
            shift_ind = self.owner.index_of(self)
            lzeros = sum((cff_cnt[i] for i in range(shift_ind)))
            rzeros = sum((cff_cnt[i] for i in range(shift_ind+1,self.owner.count)))
            lzeros = sc.zeros(lzeros)
            rzeros = sc.zeros(rzeros)
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
    from numpy import power
    dpow_i = pow_i - num_deriv
    filtered = dpow_i
    dpow_i = sc.piecewise(dpow_i,
                          [dpow_i >= 0, dpow_i < 0],
                          [lambda x: x, 0])
    coeff = vfact_div(pow_i, filtered)
    return coeff * power(x, dpow_i)


@lru_cache()
def fact_div(a, b):
    if b >= 0:
        return sc.multiply.reduce(sc.arange(b+1, a+1))
    else:
        return 0


vfact_div = sc.vectorize(fact_div)


def get_deriv_poly_var_coeffs(count_var, degree, x, deriv):
    pow_i = pow_indices(count_var, degree)
    return sc.multiply.reduce(construct_element(x, pow_i, deriv), axis=1).T
