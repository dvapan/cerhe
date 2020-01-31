import unittest

import scipy as sc
import numpy as np
from cylp.cy import CyClpSimplex,CyCoinModel
from cylp.py.modeling.CyLPModel import CyLPArray
import numpy.testing as sct



class TestCylp(unittest.TestCase):
    def connection(self):
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
        print(s.primalVariableSolution)
        print(s.objectiveValue)
        print(s.dualConstraintSolution)

        sct.assert_equal(sc.array([0., 4.]),
                         s.primalVariableSolution['x'])
        sct.assert_equal(sc.array([1., 0., 0.]),
                         s.dualConstraintSolution['R_1'])

    def test_CyCoinModel(self):
        s = CyClpSimplex()
        s.resize(0,2)
        s.CLP_addConstraint(2,np.array([0,1],np.int32),
                        np.array([-2, 1],np.float64),4,np.inf)
        s.CLP_addConstraint(2,np.array([0,1],np.int32),
                        np.array([-2, -1],np.float64),-8,np.inf)
        s.CLP_addConstraint(2,np.array([0,1],np.int32),
                        np.array([3, 2],np.float64),6,np.inf)

        # Create coefficients and bounds
        s.setObjectiveArray(np.array([3,1], np.float64))


        # Solve using primal Simplex
        s.primal()
        print(s.primalVariableSolution)
        print(s.objectiveValue)
        print(s.dualConstraintSolution)
        sct.assert_equal(sc.array([0., 4.]),
                         s.primalVariableSolution)
        sct.assert_equal(sc.array([1., 0., 0.]),
                         s.dualConstraintSolution)
