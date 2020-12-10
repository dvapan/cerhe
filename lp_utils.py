from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import scipy as sc
import numpy as np

VBND = 10**3

def slvlprd(prb, lp_dim, xdop):
    """Solve linear problem with one residual by cylp"""
    print("LOAD TO LP")
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    A = prb[:, 1:]
    nnz = np.count_nonzero(A)
    aec = len(A)*len(A[0])
    print(nnz, aec, nnz/aec)
    A = sc.matrix(A)
    b = prb[:, 0]
    b = CyLPArray(b)
    s += A*x >= b
    s += x[lp_dim-1] >= 0
    s += x[lp_dim-1] <= xdop
    s.objective = x[lp_dim-1]
    print ("task size: {} {}".format(*A.shape))
    sc.savetxt("tttt",A.shape)
    print ("START")
    s.primal()
    outx = s.primalVariableSolution['x']
    return outx, A[:,:-1].dot(outx[:-1])-b, s.dualConstraintSolution

