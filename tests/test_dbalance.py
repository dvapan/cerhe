import unittest
import scipy as sc
import numpy.testing as sct
from polynom import Polynom
from polynom import Context
from utils import boundary_coords, approximate
import dbalance as db
import utils as ut


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


class TestDBalance(unittest.TestCase):
    def setUp(self):
        self.gas, self.cer = make_gas_cer_pair(2, 3)
        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))
        self.xt_part = [(x, t) for x in X_part[0] for t in T_part[0]]

    def test_g2c(self):
        x = self.xt_part
        r = db.g2c(x, self.cer, self.gas)
        testr = sc.loadtxt("tests/g2c.dat")
        self.assertEqual(self.cer.coeff_size+self.gas.coeff_size+1, len(r[0]))
        self.assertEqual(len(x)*2, len(r))
        sct.assert_equal(testr, r)

    def test_c2a(self):
        x = self.xt_part
        r = db.c2a(x, self.cer, self.gas)
        testr = sc.loadtxt("tests/c2a.dat")
        self.assertEqual(self.cer.coeff_size+self.gas.coeff_size+1, len(r[0]))
        self.assertEqual(len(x)*2, len(r))
        sct.assert_equal(testr, r)


class TestDiscrepancyCount(unittest.TestCase):

    def setUp(self):
        xreg = 3

        self.gas, self.cer = make_gas_cer_pair(2, 3)

        T = sc.linspace(0, 1, 50)
        T_part = sc.split(T, (17, 33))
        vert_bounds = sc.linspace(0, 1, xreg+1)
        lx = sc.full_like(T_part[0], vert_bounds[0])
        self.lb = sc.vstack((lx, T_part[0])).transpose()

    def test_diff_poly2val(self):
        r = ut.delta_polynom_val(self.lb, self.gas, 1)
        testr = sc.loadtxt("tests/lbound.dat")
        sct.assert_equal(testr, r)




class TestApproximate(unittest.TestCase):
    def test_boundary_vals_creation(self):
        self.gas, self.cer = make_gas_cer_pair(2, 3)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        at = at.reshape((3, 3, 2, 10))
        atr = atr.reshape((3, 3, 2, 10))
        tgas, tcer = make_gas_cer_pair(2, 3, at[0][0][0], at[0][0][1])
        i = 0
        j = 0

        coords = boundary_coords((X_part[i], T_part[j]))

        r = tgas(coords)[:, 0]
        testr = sc.loadtxt("tests/gas_boundary_vals.dat")
        sct.assert_equal(testr, r)
        r = tcer(coords)[:, 0]
        testr = sc.loadtxt("tests/cer_boundary_vals.dat")
        sct.assert_equal(testr, r)

    def approximation_onereg(self):
        xreg, treg = 3, 3
        xdop = 1
        self.gas, self.cer = make_gas_cer_pair(2, 3)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        at = at.reshape((3, 3, 2, 10))
        atr = atr.reshape((3, 3, 2, 10))
        tgas, tcer = make_gas_cer_pair(2, 3, at[0][0][0], at[0][0][1])

        i = 0
        j = 0

        coords = boundary_coords((X_part[i], T_part[j]))
        gas = tgas(coords)[:, 0]
        dxgas = tgas(coords, [1, 0])[:, 0]
        dtgas = tgas(coords, [0, 1])[:, 0]
        cer = tcer(coords)[:, 0]
        dtcer = tcer(coords, [0, 1])[:, 0]
        bound_vals = [[gas, dxgas, dtgas], [cer, dtcer]]
        derivs = [[[0, 0], [1, 0], [0, 1]],
                  [[0, 0], [0, 1]]]
        x, xd = approximate((X_part[i], T_part[j]),
                            db.g2c,
                            (self.gas, self.cer),
                            coords, bound_vals, derivs,
                            xdop)
        print(x, xd)
        x, xd = approximate((X_part[i], T_part[j]),
                            db.c2a,
                            (self.gas, self.cer),
                            coords, bound_vals, derivs,
                            xdop)
        print(x, xd)

    def test_boundary_approximation(self):
        xdop = 1

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        at = at.reshape((3, 3, 2, 10))
        atr = atr.reshape((3, 3, 2, 10))
        tgas1, tcer1 = make_gas_cer_pair(2, 3, at[0][0][0], at[0][0][1])
        tgas2, tcer2 = make_gas_cer_pair(2, 3, at[1][0][0], at[1][0][1])

        rx = sc.full_like(T_part[0], X_part[0][-1])
        first_reg_rb = sc.vstack((rx, T_part[0])).transpose()
        lx = sc.full_like(T_part[0], X_part[1][0])
        secnd_reg_lb = sc.vstack((lx, T_part[0])).transpose()

        avg_vals = (tgas1(first_reg_rb)[:, 0] + tgas2(secnd_reg_lb)[:, 0])/2

        bnd = Polynom(1, 3)
        lp_dim = bnd.coeff_size + 1
        coords = sc.arange(len(avg_vals))
        prb_chain = []
        poly_discr = ut.delta_polynom_val(
            coords,
            bnd,
            avg_vals)
        prb_chain.append(poly_discr)
        prb_chain.append(-poly_discr)

        prb = sc.vstack(prb_chain)
        x, xd = ut.solve_linear(prb, lp_dim, xdop)

        bnd.coeffs = x[:-1]
        sct.assert_almost_equal(bnd(coords)[:, 0], avg_vals)


TGZ = 1800
TBZ = 778.17


def fromTET(TET, tgaz=TGZ, tair=TBZ):
    return tair + TET*(tgaz-tair)
