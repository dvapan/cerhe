import unittest
import scipy as sc
import numpy.testing as sct
from polynom import Polynom
from polynom import Context

class TestContextPolynoms(unittest.TestCase):
    def setUp(self):
        self.poly1 = Polynom(2,2)
        self.poly2 = Polynom(2,2)
        self.poly1.coeffs = sc.arange(1,self.poly1.coeff_size + 1)
        self.poly2.coeffs = sc.arange(1,self.poly2.coeff_size + 1)

        self.context = Context()
        self.context.assign(self.poly1)
        self.context.assign(self.poly2)

    def test_creation(self):
        p = Polynom(2,2)
        self.assertIsNone(p.owner)
        self.assertIs(self.poly1.owner,self.context)
        self.assertIs(self.poly2.owner,self.context)

    def test_slice(self):
        test_data1 = dict()
        test_data1[0,0] = sc.array([1, 1,0,0,0,0,0,0,0,0,0,0,0])
        test_data1[1,1] = sc.array([21,1,1,1,1,1,1,0,0,0,0,0,0])
        test_data2 = dict()
        test_data2[0,0] = sc.array([1, 0,0,0,0,0,0,1,0,0,0,0,0])
        test_data2[1,1] = sc.array([21,0,0,0,0,0,0,1,1,1,1,1,1])

        for x,fx in test_data1.items():
            sct.assert_equal(fx, self.poly1(x))
        for x,fx in test_data2.items():
            sct.assert_equal(fx, self.poly2(x))

    def test_addition(self):
        test_data = dict()
        test_data[0,0] = sc.array([1, 1,0,0,0,0,0,0,0,0,0,0,0])
        test_data[1,1] = sc.array([21,1,1,1,1,1,1,0,0,0,0,0,0])
        test_data[0,0] += sc.array([1, 0,0,0,0,0,0,1,0,0,0,0,0])
        test_data[1,1] += sc.array([21,0,0,0,0,0,0,1,1,1,1,1,1])
        for x,fx in test_data.items():
            sct.assert_equal(fx, self.poly1(x)+self.poly2(x))

    def test_expr(self):
        def expr(x,deriv=None):
            return self.poly1(x, deriv) + self.poly2(x, deriv)
        self.context.expr = expr

        sct.assert_equal(expr([0,0],[0,0]), self.context.eval([0,0],[0,0]))

        self.poly1.coeffs = sc.arange(3,self.poly1.coeff_size + 3)
        self.poly2.coeffs = sc.arange(1,self.poly2.coeff_size + 1)

        sct.assert_equal(expr([0,0],[0,0]), self.context.eval([0,0],[0,0]))

