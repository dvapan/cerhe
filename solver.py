import scipy as sc
from polynom import Polynom
from polynom import Context
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


import dbalance as db

TGZ = 1800
TBZ = 778.17


def fromTET(TET, tgaz=TGZ, tair=TBZ):
    return tair + TET*(tgaz-tair)


def solve_linear(prb, lp_dim, xdop):
    s = CyClpSimplex()
    x = s.addVariable('x', lp_dim)
    xdop_ar = sc.zeros(lp_dim)
    xdop_ar[0] = xdop
    prb = xdop_ar + prb
    A = prb[:, 1:]
    A = sc.hstack((A, sc.ones((len(A), 1))))
    A = sc.matrix(A)
    b = prb[:, 0]
    b = xdop - b
    b = CyLPArray(b)
    s += A*x >= b
    s.objective = x[-1]
    s.dual()
    outx = s.primalVariableSolution['x']
    outx_dual = s.dualConstraintSolution
    return outx, outx_dual


def approximate(X, equation, polynoms, bound_coords, bound_vals, derivs, xdop):
    prb_chain = []
    xt_part = [(x, t) for x in X[0] for t in X[1]]
    res = equation(xt_part, polynoms[1], polynoms[0])
    prb_chain.append(res)
    prb_chain.append(-res)
    for poly_idx in range(len(polynoms)):
        for val_idx in range(len(bound_vals[poly_idx])):
            poly_discr = db.delta_polynom_val(
                bound_coords,
                polynoms[poly_idx],
                bound_vals[poly_idx][val_idx],
                derivs[poly_idx][val_idx])
            prb_chain.append(poly_discr)
            prb_chain.append(-poly_discr)
    lp_dim = sum([x.coeff_size for x in polynoms]) + 1
    prb = sc.vstack(prb_chain)
    x, xd = solve_linear(prb, lp_dim, xdop)
    return x, list(xd.values())[0]


def boundary_coords(x):
    coords_chain = []
    lx = sc.full_like(x[1], x[0][0])
    coords_chain.append(sc.vstack((lx, x[1])).transpose())
    rx = sc.full_like(x[1], x[0][-1])
    coords_chain.append(sc.vstack((rx, x[1])).transpose())
    ut = sc.full_like(x[0], x[1][0])
    coords_chain.append(sc.vstack((x[0], ut)).transpose())
    bt = sc.full_like(x[0], x[1][-1])
    coords_chain.append(sc.vstack((x[0], bt)).transpose())

    return sc.vstack(coords_chain)


if __name__ == '__main__':
    xdop = 5
    xreg, treg = 3, 3
    cnt_var = 2
    degree = 3
    X = sc.linspace(0, 1, 50)
    T = sc.linspace(0, 1, 50)
    xt = [(x,t) for x in X for t in T]
    print(boundary_coords((X, T)))
