from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import scipy as sc

VBND = 10**3

def slvlprd(prb, lp_dim, xdop, flag=False):
    """Solve linear problem with one residual by cylp"""
    print("LOAD TO LP")
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    xdop_ar = sc.zeros(lp_dim + 1)
    A = prb[:, 1:]
    A = sc.matrix(A)
    b = prb[:, 0]
    b = CyLPArray(b)
    s += A*x >= b
    s += x[lp_dim-1] >= 0
    s += x[lp_dim-1] <= xdop
    s.objective = x[lp_dim-1]
    print ("START")
    s.dual()
    outx = s.primalVariableSolution['x']
    if flag:
        sc.savetxt("dat",A,fmt="%+16.5f")
        sc.savetxt("datb",b,fmt="%+16.5f")
        tt = A.dot(outx)
        sc.savetxt("otkl",tt.T, fmt="%16.5f")
    return outx, A.dot(outx) - b, s.dualConstraintSolution["R_1"]

