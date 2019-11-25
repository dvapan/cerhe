from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import scipy as sc

TGZ = 1800
TBZ = 778.17

VBND = 10**3

def slvlprd(prb, lp_dim, xdop, flag=False):
    """Solve linear problem with one residual by cylp"""
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    xdop_ar = sc.zeros(lp_dim)
    xdop_ar[0] = xdop
    prb = xdop_ar + prb
    A = prb[:, 1:]
    ons=sc.ones((len(A), 1))
    ons[:227]*=200
    ons[227:]*=10
    A = sc.hstack((A, ons))
    A = sc.matrix(A)
    b = prb[:, 0]

    b = xdop - b
    b = CyLPArray(b)
    s += A*x >= b
    for i in range(lp_dim-1):
        s += x[i] >= -VBND
        s += x[i] <= VBND
    s += x[lp_dim-1] >= 0
    s += x[lp_dim-1] <= xdop
    s.objective = x[-1]
    s.dual()
    outx = s.primalVariableSolution['x']
    if flag:
        sc.savetxt("dat",A,fmt="%+16.5f")
        sc.savetxt("datb",b,fmt="%+16.5f")
        tt = A.dot(outx)
        print(tt.max())
        sc.savetxt("otkl",tt.T, fmt="%16.5f")
    return outx, A.dot(outx)

def slvlprdd(prb, lp_dim, xdop):
    """Solve linear problem with one residual by cylp"""
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    xdop_ar = sc.zeros(lp_dim+1)
    xdop_ar[0] = xdop
    prb = xdop_ar + prb
    A = prb[:, 1:]
    #A = sc.hstack((A, sc.ones((len(A), 1))))
    A = sc.matrix(A)
    b = prb[:, 0]

    b = xdop - b
    b = CyLPArray(b)
    s += A*x >= b
    s += x[lp_dim-1] >= 0
    s += x[lp_dim-1] <= xdop
    s.objective = x[-1]
    s.dual()
    outx = s.primalVariableSolution['x']
    return outx, abs(A.dot(outx) - b)


def slvlprdn(prb, lp_dim, ind, xdop, bnd):
    """Solve linear problem with n residual, by cylp"""
    ind_max = max(ind) + 1
    def make_prb(i):
        x = sc.zeros(ind_max)
        x[i] = 1
        return x

    vmake_prb = sc.vectorize(make_prb,signature='(m)->(k)')

    ind_prb = vmake_prb(ind.reshape(-1, 1))
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim+ind_max)

    xdop_ar = sc.zeros(lp_dim+1)
    xdop_ar[0] = xdop
    prb = xdop_ar + prb
    A = prb[:, 1:]
    A = sc.hstack((A, ind_prb))
    A = sc.matrix(A)
    b = prb[:, 0]
    b = xdop - b
    b = CyLPArray(b)
    s += A*x >= b
    for i in range(lp_dim,lp_dim+ind_max):
        s += x[i] >= 0
        s += x[i] <= bnd
    obj = 0
    for var_ind in range(ind_max):
        obj += x[lp_dim + var_ind]
    s.objective = obj
    s.dual()
    outx = s.primalVariableSolution['x']
    outx_dual = s.dualConstraintSolution
    return outx, outx_dual, s.primalConstraintSolution
