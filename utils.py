import scipy as sc
# noinspection PyUnresolvedReferences
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from polynom import Polynom, Context

def left_boundary_coords(x):
    lx = sc.full_like(x[1], x[0][0])
    return sc.vstack((lx, x[1])).transpose()


def right_boundary_coords(x):
    rx = sc.full_like(x[1], x[0][-1])
    return sc.vstack((rx, x[1])).transpose()


def top_boundary_coords(x):
    ut = sc.full_like(x[0], x[1][0])
    return sc.vstack((x[0], ut)).transpose()


def bottom_boundary_coords(x):
    bt = sc.full_like(x[0], x[1][-1])
    return sc.vstack((x[0], bt)).transpose()


def boundary_coords(x):
    coords = {
        'l': left_boundary_coords(x),
        'r': right_boundary_coords(x),
        't': top_boundary_coords(x),
        'b': bottom_boundary_coords(x)
    }
    return coords


def make_gas_cer_pair(count_var, degree, gas_coeffs=None, cer_coeffs=None):
    cer = Polynom(count_var, degree)
    gas = Polynom(count_var, degree)
    if gas_coeffs is not None:
        gas.coeffs = gas_coeffs
    if cer_coeffs is not None:
        cer.coeffs = cer_coeffs
    context_test = Context()
    context_test.assign(gas)
    context_test.assign(cer)
    return gas, cer

splitter = (0, 17, 33, 50)

def slice(j,i):
    i_part0 = splitter[i]
    i_part1 = splitter[i + 1]
    j_part0 = splitter[j]
    j_part1 = splitter[j + 1]
    return i_part0, i_part1, j_part0, j_part1
