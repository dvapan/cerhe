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
        test_data[0,0] = 1
        test_data[1,0] = 11
        test_data[0,1] = 6
        test_data[1,1] = 21
        for x,fx in test_data.items():
            self.assertEqual(fx,self.poly.fx(x))

    def test_fx_var_coeffs(self):
        test_data = {}
        vc_zeros = sc.zeros(6)
        vc_zeros[0] = 1
        test_data[0,0] = 1,  sc.array([1,0,0,0,0,0])
        test_data[1,1] = 21, sc.array([1,1,1,1,1,1])
        for x,fx in test_data.items():
            self.assertEqual(fx[0], self.poly.fx(x))
            sct.assert_equal(fx[1], self.poly.var_coeffs)

    def test_fx_deriv(self):
        test_data = {}
        test_data[0,0] = 4,  sc.array([0,0,0,1,0,0]), [1,0]
        test_data[1,1] = 21, sc.array([0,0,0,1,1,2]), [1,0] 
        for x,fx in test_data.items():
            self.assertEqual(fx[0], self.poly.fx(x, fx[2]))
            sct.assert_equal(fx[1], self.poly.var_coeffs)
        test_data[0,0] = 2,  sc.array([0,1,0,0,0,0]), [0,1]
        test_data[1,1] = 13, sc.array([0,1,2,0,1,0]), [0,1] 
        for x,fx in test_data.items():
            self.assertEqual(fx[0], self.poly.fx(x, fx[2]))
            sct.assert_equal(fx[1], self.poly.var_coeffs)

    def test_scipy_array_equals(self):
        self.assertTrue(sc.any(sc.arange(10) == sc.arange(10)))

#TODO: make higher degree tests
