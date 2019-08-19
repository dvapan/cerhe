import unittest
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import numpy.testing as sct


class TestCylp(unittest.TestCase):
    def test_connection(self):
        s = CyClpSimplex()
        x = s.addVariable('x', 2)

        # Create coefficients and bounds
        A = sc.matrix([[-2., 1.],
                       [-2., -1.0],
                       [3., 2.]])
        b = CyLPArray([4, -8, 6])

        # Add constraints
        s += A * x >= b
        s += x >= 0
        # Set the objective function
        c = CyLPArray([3., 1.])
        s.objective = c * x

        # Solve using primal Simplex
        s.dual()
        sct.assert_equal(sc.array([0., 4.]),
                         s.primalVariableSolution['x'])
        sct.assert_equal(sc.array([1., 0., 0.]),
                         s.dualConstraintSolution['R_1'])
