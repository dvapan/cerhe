import unittest
import scipy as sc
import numpy.testing as sct
from polynom import Polynom

class TestPolynom(unittest.TestCase):
    def setUp(self):
        self.poly = Polynom(2,2)
        self.poly.coeffs = sc.arange(1,self.poly.coeff_size + 1)

    def test_fx_2degree(self):
        test_data = {}
        x = sc.array([[0,0],
                      [0,1],
                      [1,0],
                      [1,1]])
        ans = sc.array([1,6,11,21])
        sct.assert_equal(ans, self.poly.fx(x))

    def test_fx_var_coeffs(self):
        x = sc.array([[0,0],
                      [1,1]])
        fx = sc.array([[1,1,0,0,0,0,0],
                       [21,1,1,1,1,1,1]])
        sct.assert_equal(fx, self.poly.fx_var(x))

    def test_fx_deriv(self):
        test_data = {}
        test_data[0,0] = sc.array([4,0,0,0,1,0,0]), [1,0]
        test_data[1,1] = sc.array([21, 0,0,0,1,1,2]), [1,0]
        x = sc.array([[0,0],
                      [1,1]])
        fx = sc.array([[4,0,0,0,1,0,0],
                      [21, 0,0,0,1,1,2]])
        deriv = [1,0]
        sct.assert_equal(fx, self.poly.fx_var(x, deriv))
        deriv = [0,1]
        x = sc.array([[0,0],
                      [1,1]])
        fx = sc.array([[2,0,1,0,0,0,0],
                       [13,0,1,2,0,1,0]])
        sct.assert_equal(fx, self.poly.fx_var(x, deriv))

    def test_scipy_array_equals(self):
        self.assertTrue(sc.any(sc.arange(10) == sc.arange(10)))

#TODO: make higher degree tests
