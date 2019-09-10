import unittest
import scipy as sc
import numpy.testing as sct
from polynom import Polynom
from polynom import Context
from utils import boundary_coords, approximate_equation_polynom
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
    def setUp(self):
        self.gas, self.cer = make_gas_cer_pair(2, 3)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        self.X_part = sc.split(X, (17, 33))
        self.T_part = sc.split(T, (17, 33))

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        self.at = at.reshape((3, 3, 2, 10))
        self.atr = atr.reshape((3, 3, 2, 10))

    def test_boundary_vals_creation(self):

        tgas, tcer = make_gas_cer_pair(2, 3, self.at[0][0][0], self.at[0][0][1])
        i = 0
        j = 0

        coords = boundary_coords((self.X_part[i], self.T_part[j]))

        r = tgas(coords)[:, 0]
        testr = sc.loadtxt("tests/gas_boundary_vals.dat")
        sct.assert_equal(testr, r)
        r = tcer(coords)[:, 0]
        testr = sc.loadtxt("tests/cer_boundary_vals.dat")
        sct.assert_equal(testr, r)

    def test_approximation_onereg(self):
        xdop = 1

        tgas, tcer = make_gas_cer_pair(2, 3, self.at[0][0][0], self.at[0][0][1])

        i = 0
        j = 0

        coords = boundary_coords((self.X_part[i], self.T_part[j]))
        gas = tgas(coords)[:, 0]
        dxgas = tgas(coords, [1, 0])[:, 0]
        dtgas = tgas(coords, [0, 1])[:, 0]
        cer = tcer(coords)[:, 0]
        dtcer = tcer(coords, [0, 1])[:, 0]
        bound_vals = [[gas, dxgas, dtgas], [cer, dtcer]]
        derivs = [[[0, 0], [1, 0], [0, 1]],
                  [[0, 0], [0, 1]]]
        x, xd, unparsed = approximate_equation_polynom((self.X_part[i], self.T_part[j]),
                                             db.g2c,
                                             (self.gas, self.cer),
                                             coords, bound_vals, derivs,
                                             xdop)
        dual_sol_bnds = dict()
        dual_sol_bnds['gas'] = ut.parse_bounds((self.X_part[i], self.T_part[j]), xd[0][0])
        dual_sol_bnds['dxgas'] = xd[0][1]
        dual_sol_bnds['dtgas'] = xd[0][2]
        dual_sol_bnds['cer'] = ut.parse_bounds((self.X_part[i], self.T_part[j]), xd[1][0])
        dual_sol_bnds['dtcer'] = xd[1][1]
        sc.set_printoptions(precision=3, linewidth=110)
        import pprint
        pprint.pprint(dual_sol_bnds['gas'])
        pprint.pprint(dual_sol_bnds['cer'])

        xt_part = [(x, t) for x in self.X_part[i] for t in self.T_part[j]]
        rgas, _ = make_gas_cer_pair(2, 3, x[0:10], x[10:20])
        sct.assert_almost_equal(tgas(xt_part)[:, 0], rgas(xt_part)[:, 0],4)

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

        first_reg_rb = ut.right_boundary_coords((X_part[0],T_part[0]))
        secnd_reg_lb = ut.left_boundary_coords((X_part[1],T_part[0]))

        avg_vals = (tgas1(first_reg_rb)[:, 0] + tgas2(secnd_reg_lb)[:, 0])/2

        bnd = Polynom(1, 3)
        x, xd = ut.approximate_bound_polynom(bnd,avg_vals,xdop)
        bnd.coeffs = x[:-1]
        coords = sc.arange(len(avg_vals))
        sct.assert_almost_equal(bnd(coords)[:, 0], avg_vals)


TGZ = 1800
TBZ = 778.17


def fromTET(TET, tgaz=TGZ, tair=TBZ):
    return tair + TET*(tgaz-tair)
