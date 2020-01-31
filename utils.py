import scipy as sc
from itertools import *
# noinspection PyUnresolvedReferences
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from polynom import Polynom, Context

from constants import *

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

def make_gas_cer_quad(count_var, degree, gas_coeffs=None, cer_coeffs=None, gasr_coeffs=None, cerr_coeffs=None):
    cer = Polynom(count_var, degree)
    gas = Polynom(count_var, degree)
    cerr = Polynom(count_var, degree)
    gasr = Polynom(count_var, degree)

    if gas_coeffs is not None:
        gas.coeffs = gas_coeffs
    if cer_coeffs is not None:
        cer.coeffs = cer_coeffs
    if gasr_coeffs is not None:
        gasr.coeffs = gasr_coeffs
    if cerr_coeffs is not None:
        cerr.coeffs = cerr_coeffs
    context_test = Context()
    context_test.assign(gas)
    context_test.assign(cer)
    context_test.assign(gasr)
    context_test.assign(cerr)
    return gas, cer, gasr, cerr


splitter = (0, 17, 33, 50)

def slice(j,i):
    i_part0 = splitter[i]
    i_part1 = splitter[i + 1]
    j_part0 = splitter[j]
    j_part1 = splitter[j + 1]
    return i_part0, i_part1, j_part0, j_part1


def split(name, reg1, reg2):
    return zip(repeat(name), product(range(reg1), range(reg2)))


def split_slice1(name, reg1, reg2):
    return zip(repeat(name), product(range(1, reg1), range(reg2)))

def split_slice2(name, reg1, reg2):
    return zip(repeat(name), product(range(0, reg1), range(1,reg2)))


def split_fix1(name, reg1, reg2):
    return zip(repeat(name), product(range(reg1 - 1, reg1), range(reg2)))


def split_fix2(name, reg1, reg2):
    return zip(repeat(name), product(range(reg1), range(reg2 - 1, reg2)))

def cast_type(seq,type):
    return zip(repeat(type), seq)

def balance_constraints(eqs, pol1, pol2):
    return product(eqs,
                   zip(
                       cast_type(split(pol1, xreg, treg), "i"),
                       cast_type(split(pol2, xreg, treg), "i")))


def start_constraints(eqs, pol1, base, regid, type):
    return product(eqs,
                   zip(
                       cast_type(split_fix1(pol1, regid, treg), type),
                       cast_type(repeat((base, (0, 0))), "c")))


def intereg_constraints(eqs, pol1):
    return chain(product(eqs,
                         zip(
                             cast_type(split(pol1, xreg, treg),"r"),
                             cast_type(split_slice1(pol1, xreg, treg),"l")),
                 product(eqs,
                         zip(
                             cast_type(split(pol1, xreg, treg),"r"),
                             cast_type(split_slice1(pol1, xreg, treg),"l")))))


def intemod_constraints(eqs, pol1, pol2):
    return chain(product(eqs,
                         zip(
                             cast_type(split_fix2(pol1, xreg, 1),"t"),
                             cast_type(split_fix2(pol2, xreg, treg), "b"))),
                 product(eqs,
                         zip(
                            cast_type(split_fix2(pol1, xreg, treg), "b"),
                            cast_type(split_fix2(pol2, xreg, 1), "t"))))

def construct_mode(beqs, base, base_id,  type, pols):
    return chain(
        balance_constraints(beqs,
                            pols[0], pols[1]),
        start_constraints(['gas'], pols[0], base, base_id, type),
        product(["gas"],
                zip(
                    cast_type(split(pols[0], xreg-1, treg), "r"),
                    cast_type(split_slice1(pols[0], xreg, treg), "l"))),
        product(["gas"],
                zip(
                    cast_type(split(pols[0], xreg, treg-1), "b"),
                    cast_type(split_slice2(pols[0], xreg, treg), "t"))),
        product(["cer"],
                zip(
                    cast_type(split(pols[1], xreg-1, treg), "r"),
                    cast_type(split_slice1(pols[1], xreg, treg), "l"))),
        product(["cer"],
                zip(
                    cast_type(split(pols[1], xreg, treg-1), "b"),
                    cast_type(split_slice2(pols[1], xreg, treg), "t"))))

