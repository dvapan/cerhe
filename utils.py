import scipy as sc
# noinspection PyUnresolvedReferences
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from polynom import Polynom, Context

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
    slice = list()
    xt_part = [(x, t) for x in X[0] for t in X[1]]
    res = equation(xt_part, polynoms[1], polynoms[0])
    prb_chain.append(res)
    slice.append(len(slice))
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
    solve_linear_test(prb, lp_dim, xdop, x, xd)
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
    coords = {
        'l': left_boundary_coords(x),
        'r': right_boundary_coords(x),
        't': top_boundary_coords(x),
        'b': bottom_boundary_coords(x)
    }
    return coords


def make_gas_cer_pair(count_var, degree, gas_coeffs=None, cer_coeffs=None):
    cer = Polynom(count_var, degree)
    gas = Polynom(count_var, degree)
    if gas_coeffs is not None:
        gas.coeffs = gas_coeffs
    if cer_coeffs is not None:
        cer.coeffs = cer_coeffs
    context_test = Context()
    context_test.assign(gas)
    context_test.assign(cer)
    return gas, cer

splitter = (0, 17, 33, 50)

def slice(i,j):
    i_part0 = splitter[i]
    i_part1 = splitter[i + 1]
    j_part0 = splitter[j]
    j_part1 = splitter[j + 1]
    return i_part0, i_part1, j_part0, j_part1
