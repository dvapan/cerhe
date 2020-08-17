from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import scipy as sc

VBND = 10**3

def slvlprd(prb, lp_dim, xdop):
    """Solve linear problem with one residual by cylp"""
    print("LOAD TO LP")
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    A = prb[:, 1:]
    A = sc.matrix(A)
    b = prb[:, 0]
    b = CyLPArray(b)
    s += A*x >= b
    s += x[lp_dim-1] >= 0
    s += x[lp_dim-1] <= xdop
    s.objective = x[lp_dim-1]
    print ("START")
    s.primal()
    outx = s.primalVariableSolution['x']
    return outx, A.dot(outx) - b, s.dualConstraintSolution

