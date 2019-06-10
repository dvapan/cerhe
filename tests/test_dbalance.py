import unittest
import scipy as sc
import numpy.testing as sct
from polynom import Polynom
from polynom import Context
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import dbalance as db

class TestDBalance(unittest.TestCase):
    def setUp(self):
        self.cer = Polynom(2,3)
        self.gas = Polynom(2,3)

        context = Context()
        context.assign(self.gas)
        context.assign(self.cer)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))
        self.xt_part = [(x, t) for x in X_part[0] for t in T_part[0]]

    def test_g2c(self):
        x = self.xt_part
        r = db.g2c(x,self.cer,self.gas)
        testr = sc.loadtxt("tests/g2c.dat")
        self.assertEqual(self.cer.coeff_size+self.gas.coeff_size+1, len(r[0]))
        self.assertEqual(len(x)*2,len(r))
        sct.assert_equal(testr, r)

    def test_c2a(self):
        x = self.xt_part
        r = db.c2a(x,self.cer,self.gas)
        testr = sc.loadtxt("tests/c2a.dat")
        self.assertEqual(self.cer.coeff_size+self.gas.coeff_size+1, len(r[0]))
        self.assertEqual(len(x)*2,len(r))
        sct.assert_equal(testr, r)

class TestDiscrepancyCount(unittest.TestCase):

    def setUp(self):
        xreg, treg = 3, 3
        self.cer = Polynom(2,3)
        self.gas = Polynom(2,3)

        context = Context()
        context.assign(self.gas)
        context.assign(self.cer)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))
        vert_bounds = sc.linspace(0, 1, xreg+1)
        hori_bounds = sc.linspace(0, 1, treg+1)

        xt_part = [(x, t) for x in X_part[0] for t in T_part[0]]
        lx = sc.full_like(T_part[0], vert_bounds[0])
        self.lb = sc.vstack((lx, T_part[0])).transpose()

    def test_diff_poly2val(self):
        r = db.delta_polynom_val(self.lb, self.gas,1)
        testr = sc.loadtxt("tests/lbound.dat")
        sct.assert_equal(testr, r)


def solve_linear(prb, lp_dim, xdop):
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    xdop_ar = sc.zeros(lp_dim)
    xdop_ar[0] = xdop
    prb = xdop_ar + prb
    A = prb[:, 1:]
    A = sc.hstack((A, sc.ones((len(A), 1))))
    A = sc.matrix(A)
    b = prb[:, 0]
    b = xdop - b
    b = CyLPArray(b)
    s += A*x >= b 
    s.objective = x[-1] 
    s.dual()
    outx = s.primalVariableSolution['x']
    outx_dual = s.dualConstraintSolution
    sc.savetxt("xd.dat",s.dualConstraintSolution['R_4'])
    return outx, outx_dual

def approximate(X, polynoms, bound_coords, bound_vals, derivs, xdop):
    prb_chain = []
    count_var = polynoms[0].count_var
    poly_cnt = len(polynoms)
    xt_part = [(x, t) for x in X[0] for t in X[1]]
    res = db.g2c(xt_part, polynoms[1], polynoms[0])
    prb_chain.append(res)
    prb_chain.append(-res)
    # bound_coords = polynoms_bounds[:, :count_var]
    # bound_vals = polynoms_bounds[:, count_var:]
    for poly_idx in range(len(polynoms)):
        for val_idx in range(len(bound_vals[poly_idx])):
            poly_discr = db.delta_polynom_val(
                bound_coords,
                polynoms[poly_idx],
                bound_vals[poly_idx][val_idx],
                derivs[poly_idx][val_idx])
            prb_chain.append(poly_discr)
            prb_chain.append(-poly_discr)
    lp_dim = sum([x.coeff_size for x in polynoms]) + 1
    # xdop_ar = sc.zeros(lp_dim)
    # xdop_ar[0] = xdop
    # prb1 = sc.vstack(prb_chain)
    # prb = xdop_ar + prb1
    prb = sc.vstack(prb_chain)
    sc.savetxt("out",prb)

    x,xd = solve_linear(prb,lp_dim, xdop)

    print(x)
    print(xd)

def boundary_coords(x):
    coords_chain = []
    lx = sc.full_like(x[1], x[0][0])
    coords_chain.append(sc.vstack((lx, x[1])).transpose())
    rx = sc.full_like(x[1], x[0][-1])
    coords_chain.append(sc.vstack((rx, x[1])).transpose())
    ut = sc.full_like(x[0], x[1][0])
    coords_chain.append(sc.vstack((x[0], ut)).transpose())
    bt = sc.full_like(x[0], x[1][-1])
    coords_chain.append(sc.vstack((x[0], bt)).transpose())

    return sc.vstack(coords_chain)


class TestApproximate(unittest.TestCase):
    def test_boundary_vals_creation(self):
        xreg, treg = 3, 3
        xdop = 1
        self.cer = Polynom(2, 3)
        self.gas = Polynom(2, 3)

        context = Context()
        context.assign(self.gas)
        context.assign(self.cer)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        at = at.reshape((3, 3, 2, 10))
        atr = atr.reshape((3, 3, 2, 10))
        tcer = Polynom(2, 3)
        tgas = Polynom(2, 3)
        tgas.coeffs = at[0][0][0]
        tcer.coeffs = at[0][0][1]
        context_test = Context()
        context_test.assign(tgas)
        context_test.assign(tcer)

        i = 0
        j = 0

        coords = boundary_coords((X_part[i], T_part[j]))

        r = tgas(coords)[:, 0]
        testr = sc.loadtxt("tests/gas_boundary_vals.dat")
        sct.assert_equal(testr, r)
        r = tcer(coords)[:, 0]
        testr = sc.loadtxt("tests/cer_boundary_vals.dat")
        sct.assert_equal(testr, r)


    def test_approximation(self):
        xreg, treg = 3, 3
        xdop = 1
        self.cer = Polynom(2, 3)
        self.gas = Polynom(2, 3)

        context = Context()
        context.assign(self.gas)
        context.assign(self.cer)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        at = at.reshape((3, 3, 2, 10))
        atr = atr.reshape((3, 3, 2, 10))
        tcer = Polynom(2, 3)
        tgas = Polynom(2, 3)
        tgas.coeffs = at[0][0][0]
        tcer.coeffs = at[0][0][1]
        context_test = Context()
        context_test.assign(tgas)
        context_test.assign(tcer)

        i = 0
        j = 0

        coords = boundary_coords((X_part[i], T_part[j]))
        gas = tgas(coords)[:,0]
        dxgas = tgas(coords,[1,0])[:,0]
        dtgas = tgas(coords,[0,1])[:,0]
        cer = tcer(coords)[:,0]
        dtcer = tcer(coords,[0,1])[:,0]
        bound_vals = [[gas, dxgas, dtgas],[cer,dtcer]]

        derivs = [[[0,0],[1,0],[0,1]],[[0,0],[0,1]]]
        approximate((X_part[i],T_part[j]),
                    (self.gas, self.cer),
                    coords, bound_vals, derivs,
                    xdop) 
