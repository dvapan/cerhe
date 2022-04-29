import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import logging

def solve(A, rhs, ct=None, logLevel=0):
    s = CyClpSimplex()
    s.logLevel = logLevel
    lp_dim = A.shape[1]

    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)
    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    s.objective = x[lp_dim - 1]
    nnz = np.count_nonzero(A)
    logging.debug(f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.primal()
    k = list(s.primalConstraintSolution.keys())[0]
    k2 =list(s.dualConstraintSolution.keys())[0]
    q = s.dualConstraintSolution[k2]
    logging.debug(f"{s.getStatusString()} objective: {s.objectiveValue}")
    logging.debug(f"nonzeros rhs: {np.count_nonzero(s.primalConstraintSolution[k])}")
    logging.debug(f"nonzeros dual: {np.count_nonzero(s.dualConstraintSolution[k2])}")

    basis = s.getBasisStatus()
    print("Basis[0]")
    print(basis[0])
    print("basis[1]")
    print(basis[1])
    np.savetxt("out",basis[1])

    print("#"*100)

    s = CyClpSimplex()
    s.logLevel = logLevel
    lp_dim = A.shape[1]

    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)
    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    s.objective = x[lp_dim - 1]
    nnz = np.count_nonzero(A)
    logging.debug(f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.setBasisStatus(*basis)
    s.primal()

    return s.primalVariableSolution['x']
