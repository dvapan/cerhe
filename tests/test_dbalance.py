import unittest
import scipy as sc
import numpy.testing as sct
from polynom import Polynom
from polynom import Context

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
    def test_diff_val(self):
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
        tgl = db.delta_polynom_val(self.lb, self.gas,1)
        testr = sc.loadtxt("tests/lbound.dat")
        sct.assert_equal(testr, tgl)
