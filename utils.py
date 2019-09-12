import scipy as sc
# noinspection PyUnresolvedReferences
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

LEFT,RIGHT,TOP,BOTTOM = range(4)

def solve_linear_test(prb, lp_dim, xdop, x, xd):
    xdop_ar = sc.zeros(lp_dim)
    xdop_ar[0] = xdop
    prb = xdop_ar + prb
    A = prb[:, 1:]
    A = sc.hstack((A, sc.ones((len(A), 1))))
    b = prb[:, 0]
    b = xdop - b
    print ("TEST_DUAL_SOLUTION: dual",sum(xd*b),"primal:",x[-1],"delta:",x[-1]-sum(xd*b))
    # sc.savetxt('outx',sc.vstack([A.dot(x)-b, xd]).transpose())


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
    # sc.savetxt('test1',outx_dual['R_1'])
    return outx, outx_dual


def delta_polynom_val(x, polynom, vals, deriv=None):
    t = polynom(x, deriv)
    t[:, 0] = t[:, 0] - vals
    return t

def approximate_bound_polynom(polynom, vals,xdop):
    lp_dim = polynom.coeff_size + 1
    coords = sc.arange(len(vals))
    prb_chain = []
    poly_discr = delta_polynom_val(
        coords,
        polynom,
        vals)
    prb_chain.append(poly_discr)
    prb_chain.append(-poly_discr)

    prb = sc.vstack(prb_chain)
    x, xd = solve_linear(prb, lp_dim, xdop)
    return x, list(xd.values())[0]


def approximate_equation_polynom(X, equation, polynoms, bound_coords, bound_vals, derivs, xdop):
    prb_chain = []
    xt_part = [(x, t) for x in X[0] for t in X[1]]
    res = equation(xt_part, polynoms[1], polynoms[0])
    prb_chain.append(res)
    prb_chain.append(-res)
    count_boundares = 0
    for poly_idx in range(len(polynoms)):
        for val_idx in range(len(bound_vals[poly_idx])):
            poly_discr = delta_polynom_val(
                bound_coords,
                polynoms[poly_idx],
                bound_vals[poly_idx][val_idx],
                derivs[poly_idx][val_idx])
            prb_chain.append(poly_discr)
            prb_chain.append(-poly_discr)
            count_boundares += 1
    lp_dim = sum([x.coeff_size for x in polynoms]) + 1
    prb = sc.vstack(prb_chain)
    x, xd = solve_linear(prb, lp_dim, xdop)
    xd = sc.array(list(xd.values())[0])
    solve_linear_test(prb,lp_dim,xdop,x,xd)
    # xd = solve_linear_test(prb,lp_dim,xdop,x)
    xd = xd[len(res)*2:]
    unparsed = xd
    out = list(range(len(polynoms)))
    h = int(len(xd)/count_boundares)
    i = 0
    for poly_idx in range(len(polynoms)):
        out[poly_idx] = list(range(len(bound_vals[poly_idx])))
        for val_idx in range(len(bound_vals[poly_idx])):
            out[poly_idx][val_idx] = xd[i*h:(i+1)*h]
    return x, out, unparsed


def left_boundary_coords(x):
    lx = sc.full_like(x[1], x[0][0])
    return sc.vstack((lx, x[1])).transpose()


def right_boundary_coords(x):
    rx = sc.full_like(x[1], x[0][-1])
    return sc.vstack((rx, x[1])).transpose()


def top_boundary_coords(x):
    ut = sc.full_like(x[0], x[1][0])
    return sc.vstack((x[0], ut)).transpose()


def bottom_boundary_coords(x):
    bt = sc.full_like(x[0], x[1][-1])
    return sc.vstack((x[0], bt)).transpose()


def boundary_coords(x):
    coords_chain = [
        left_boundary_coords(x),
        right_boundary_coords(x),
        top_boundary_coords(x),
        bottom_boundary_coords(x)
    ]
    return sc.vstack(coords_chain)

def parse_bounds(x,dual_sol):
    full_size = int(len(dual_sol)/2)
    left_bound = len(x[0])
    right_bound = left_bound+len(x[0])
    top_bound = right_bound + len(x[1])
    bottom_bound = top_bound + len(x[1])
    out = dict()
    out['left_pos'] = dual_sol[:left_bound]
    out['left_neg'] = dual_sol[full_size:full_size+left_bound]
    out['right_pos'] = dual_sol[left_bound:right_bound]
    out['right_neg'] = dual_sol[full_size+left_bound:full_size+right_bound]
    out['top_pos'] = dual_sol[right_bound:top_bound]
    out['top_neg'] = dual_sol[full_size+right_bound:full_size+top_bound]
    out['bottom_pos'] = dual_sol[top_bound:bottom_bound]
    out['bottom_neg'] = dual_sol[full_size+top_bound:full_size+bottom_bound]
    return out
