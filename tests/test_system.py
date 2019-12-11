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

        self.x1 = sc.array([0,0])
        self.x2 = sc.array([1,1])
        self.f1x1 = sc.array([1, 1,0,0,0,0,0,0,0,0,0,0,0])
        self.f1x2 = sc.array([21,1,1,1,1,1,1,0,0,0,0,0,0])
        self.f2x1 = sc.array([1, 0,0,0,0,0,0,1,0,0,0,0,0])
        self.f2x2 = sc.array([21,0,0,0,0,0,0,1,1,1,1,1,1])

    def test_creation(self):
        p = Polynom(2,2)
        self.assertIsNone(p.owner)
        self.assertIs(self.poly1.owner,self.context)
        self.assertIs(self.poly2.owner,self.context)

    def test_slice(self):
        sct.assert_equal(self.f1x1, self.poly1(self.x1))
        sct.assert_equal(self.f1x2, self.poly1(self.x2))
        sct.assert_equal(self.f2x1, self.poly2(self.x1))
        sct.assert_equal(self.f2x2, self.poly2(self.x2))

    def test_addition(self):
        sct.assert_equal(self.f1x1 + self.f2x1, self.poly1(self.x1) + self.poly2(self.x1))
        sct.assert_equal(self.f1x2 + self.f2x2, self.poly1(self.x2) + self.poly2(self.x2))


    def test_expr(self):
        def expr(x,deriv=None):
            return self.poly1(x, deriv) + self.poly2(x, deriv)
        self.context.expr = expr

        sct.assert_equal(expr([0,0],[0,0]), self.context.eval([0,0],[0,0]))

        self.poly1.coeffs = sc.arange(3,self.poly1.coeff_size + 3)
        self.poly2.coeffs = sc.arange(1,self.poly2.coeff_size + 1)

        sct.assert_equal(expr([0,0],[0,0]), self.context.eval([0,0],[0,0]))

class TestContextPolynomsDifferentVarCountTwoPols(unittest.TestCase):
    def setUp(self):
        self.poly1 = Polynom(2,2)
        self.poly2 = Polynom(3,2)
        self.poly1.coeffs = sc.arange(1,self.poly1.coeff_size + 1)
        self.poly2.coeffs = sc.arange(1,self.poly2.coeff_size + 1)
        self.context = Context()
        self.context.assign(self.poly1)
        self.context.assign(self.poly2)

        self.x21 = sc.array([0,0])
        self.x22 = sc.array([1,1])
        self.x31 = sc.array([0,0,0])
        self.x32 = sc.array([1,1,1])

        self.f1x1 = sc.array([1, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.f1x2 = sc.array([21,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
        self.f2x1 = sc.array([1, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        self.f2x2 = sc.array([55,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

    def test_creation(self):
        p = Polynom(2,2)
        self.assertIsNone(p.owner)
        self.assertIs(self.poly1.owner,self.context)
        self.assertIs(self.poly2.owner,self.context)

    def test_slice(self):
        sct.assert_equal(self.f1x1, self.poly1(self.x21))
        sct.assert_equal(self.f1x2, self.poly1(self.x22))
        sct.assert_equal(self.f2x1, self.poly2(self.x31))
        sct.assert_equal(self.f2x2, self.poly2(self.x32))

    def test_addition(self):
        sct.assert_equal(self.f1x1 + self.f2x1, self.poly1(self.x21) + self.poly2(self.x31))
        sct.assert_equal(self.f1x2 + self.f2x2, self.poly1(self.x22) + self.poly2(self.x32))


class TestContextPolynomsDifferentVarCountFourPols(unittest.TestCase):
    def setUp(self):
        self.poly1 = Polynom(2,2)
        self.poly2 = Polynom(3,2)
        self.poly3 = Polynom(2,2)
        self.poly4 = Polynom(3,2)
        self.poly1.coeffs = sc.arange(1,self.poly1.coeff_size + 1)
        self.poly2.coeffs = sc.arange(1,self.poly2.coeff_size + 1)
        self.poly3.coeffs = sc.arange(1,self.poly3.coeff_size + 1)
        self.poly4.coeffs = sc.arange(1,self.poly4.coeff_size + 1)
        self.context = Context()
        self.context.assign(self.poly1)
        self.context.assign(self.poly2)
        self.context.assign(self.poly3)
        self.context.assign(self.poly4)

        self.x21 = sc.array([0,0])
        self.x22 = sc.array([1,1])
        self.x31 = sc.array([0,0,0])
        self.x32 = sc.array([1,1,1])

        self.f1x1 = sc.array([1, 1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.f1x2 = sc.array([21,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.f2x1 = sc.array([1, 0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.f2x2 = sc.array([55,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.f3x1 = sc.array([1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0])
        self.f3x2 = sc.array([21,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,0,0,0,0,0,0,0,0,0,0])
        self.f4x1 = sc.array([1, 0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,0,0,0,0,0,0,0,0,0])
        self.f4x2 = sc.array([55,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,0,1,1,1,1,1,1,1,1,1,1])

    def test_creation(self):
        p = Polynom(2,2)
        self.assertIsNone(p.owner)
        self.assertIs(self.poly1.owner,self.context)
        self.assertIs(self.poly2.owner,self.context)

    def test_slice(self):
        sct.assert_equal(self.f1x1, self.poly1(self.x21))
        sct.assert_equal(self.f1x2, self.poly1(self.x22))
        sct.assert_equal(self.f2x1, self.poly2(self.x31))
        sct.assert_equal(self.f2x2, self.poly2(self.x32))
        sct.assert_equal(self.f3x1, self.poly3(self.x21))
        sct.assert_equal(self.f3x2, self.poly3(self.x22))
        sct.assert_equal(self.f4x1, self.poly4(self.x31))
        sct.assert_equal(self.f4x2, self.poly4(self.x32))

    def test_addition(self):
        sct.assert_equal(self.f1x1 + self.f2x1, self.poly1(self.x21) + self.poly2(self.x31))
        sct.assert_equal(self.f1x2 + self.f2x2, self.poly1(self.x22) + self.poly2(self.x32))

