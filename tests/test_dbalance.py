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


class TestApproximate(unittest.TestCase):
    def test(self):
        xreg, treg = 3, 3
        xdop = 5
        self.cer = Polynom(2, 3)
        self.gas = Polynom(2, 3)

        context = Context()
        context.assign(self.gas)
        context.assign(self.cer)

        X = sc.linspace(0, 1, 50)
        T = sc.linspace(0, 1, 50)
        X_part = sc.split(X, (17, 33))
        T_part = sc.split(T, (17, 33))
        vert_bounds = sc.linspace(0, 1, xreg+1)
        hori_bounds = sc.linspace(0, 1, treg+1)

        at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
        at = at.reshape((3, 3, 2, 10))
        atr = atr.reshape((3, 3, 2, 10))
        self.tcer = Polynom(2, 3)
        self.tgas = Polynom(2, 3)
        self.tgas.coeffs = at[0][0][0]
        self.tcer.coeffs = at[0][0][1]
        context_test = Context()
        context_test.assign(self.tgas)
        context_test.assign(self.tcer)

        i = 0
        j = 0

        prb_chain = []
        xt_part = [(x, t) for x in X_part[i] for t in T_part[j]]
        res = db.g2c(xt_part, self.cer, self.gas)

        lx = sc.full_like(T_part[j], vert_bounds[i])
        lb = sc.vstack((lx, T_part[j])).transpose()
        lb_vals = sc.zeros_like(T_part[j])
        tgl = db.delta_polynom_val(lb, self.gas, lb_vals)
        lb_vals = self.tcer(lb)[:, 0]
        tcl = db.delta_polynom_val(lb, self.cer, lb_vals)

        rx = sc.full_like(T_part[j], vert_bounds[i + 1])
        rb = sc.vstack((rx, T_part[j])).transpose()
        rb_vals = self.tgas(rb)[:, 0]
        tgr = db.delta_polynom_val(rb, self.gas, rb_vals)
        rb_vals = self.tcer(rb)[:, 0]
        tcr = db.delta_polynom_val(rb, self.cer, rb_vals)

        bt = sc.full_like(X_part[i], hori_bounds[j + 1])
        bb = sc.vstack((X_part[i], bt)).transpose()
        bb_vals = self.tgas(bb)[:, 0]
        tgb = db.delta_polynom_val(bb, self.gas, bb_vals)
        bb_vals = self.tcer(bb)[:, 0]
        tcb = db.delta_polynom_val(rb, self.cer, bb_vals)
        prb_chain.append(tgl)
        prb_chain.append(tcl)
        prb_chain.append(tgr)
        prb_chain.append(tcr)
        # prb_chain.append(tgu)
        # prb_chain.append(tcu)
        prb_chain.append(tgb)
        prb_chain.append(tcb)
        prb_chain.append(-tgl)
        prb_chain.append(-tcl)
        prb_chain.append(-tgr)
        prb_chain.append(-tcr)
        # prb_chain.append(-tgu)
        # prb_chain.append(-tcu)
        prb_chain.append(-tgb)
        prb_chain.append(-tcb)

        s = CyClpSimplex()
        lp_dim = self.cer.coeff_size + self.gas.coeff_size + 1
        x = s.addVariable('x', lp_dim)
        xdop_ar = sc.zeros(lp_dim)
        xdop_ar[0] = xdop
        prb1 = sc.vstack(prb_chain)
        prb = xdop_ar + prb1

        A = prb[:, 1:]
        A = sc.hstack((A, sc.ones((len(A), 1))))
        A = sc.matrix(A)
        b = prb[:, 0]
        b = xdop - b
        b = CyLPArray(b)

        s += A*x-b >= 0
        s.objective = x[-1]
        s.dual()
