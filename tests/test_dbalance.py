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
        self.assertEqual(self.cer.coeff_size+self.gas.coeff_size+1, len(r[0]))
        self.assertEqual(len(x)*2,len(r))

    def test_a2c(self):
        x = self.xt_part
        r = db.g2c(x,self.cer,self.gas)
        self.assertEqual(self.cer.coeff_size+self.gas.coeff_size+1, len(r[0]))
        self.assertEqual(len(x)*2,len(r))
