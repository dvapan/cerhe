import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from polynom import Polynom
from polynom import Context
import utils as ut
import dbalance as db

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

if __name__ == '__main__':
    xdop = 5
    xreg, treg = 3, 3
    cnt_var = 2
    degree = 3
    X = sc.linspace(0, 1, 50)
    T = sc.linspace(0, 1, 50)
    gas, cer = make_gas_cer_pair(2, 3)
    X_part = sc.split(X, (17, 33))
    T_part = sc.split(T, (17, 33))
    xt = [(x, t) for x in X for t in T]

    xt_vals = sc.repeat(sc.linspace(1, 0, 50), 50).reshape(-1, 50)
    splitter = (0,17,33,50)
    i = 0
    j = 0
    i_part0 = splitter[i]
    i_part1 = splitter[i+1]
    j_part0 = splitter[j]
    j_part1 = splitter[j+1]
    sc.set_printoptions(precision=3, linewidth=110)
    reg = xt_vals[i_part0:i_part1, j_part0:j_part1]
    coords = ut.boundary_coords((X_part[i], T_part[j]))
    # vals = sc.hstack([reg[0, :], sc.full_like(), reg[:, 0], reg[:, -1]])
    vals = sc.hstack([reg[0, :], reg[-1, :], reg[:, 0], reg[:, -1]])
    vgas = vals
    vgas[0] = 0.5
    vdxgas = sc.zeros_like(vals)
    vdtgas = sc.zeros_like(vals)
    vcer = vals
    vdtcer = sc.zeros_like(vals)
    bound_vals = [[vgas, vdxgas, vdtgas], [vcer, vdtcer]]
    derivs = [[[0, 0], [1, 0], [0, 1]],
              [[0, 0], [0, 1]]]
    x, xd, inner = ut.approximate_equation_polynom((X_part[i], T_part[j]),
                                         db.g2c,
                                         (gas, cer),
                                         coords, bound_vals, derivs,
                                         xdop)

    dual_sol_bnds = dict()
    dual_sol_bnds['gas'] = ut.parse_bounds((X_part[i], T_part[j]), xd[0][0])
    dual_sol_bnds['dxgas'] = ut.parse_bounds((X_part[i], T_part[j]), xd[0][1])
    dual_sol_bnds['dtgas'] = ut.parse_bounds((X_part[i], T_part[j]), xd[0][2])
    dual_sol_bnds['cer'] = ut.parse_bounds((X_part[i], T_part[j]), xd[1][0])
    dual_sol_bnds['dtcer'] = ut.parse_bounds((X_part[i], T_part[j]), xd[1][1])
    sc.set_printoptions(precision=3, linewidth=110)
    import pprint

    print("gas")
    pprint.pprint(dual_sol_bnds['gas']['left_pos']-dual_sol_bnds['gas']['left_neg'])
    pprint.pprint(dual_sol_bnds['gas']['right_pos']-dual_sol_bnds['gas']['right_neg'])
    pprint.pprint(dual_sol_bnds['gas']['top_pos']-dual_sol_bnds['gas']['top_neg'])
    pprint.pprint(dual_sol_bnds['gas']['bottom_pos']-dual_sol_bnds['gas']['bottom_neg'])
    print("dxgas")
    pprint.pprint(dual_sol_bnds['dxgas']['left_pos']-dual_sol_bnds['gas']['left_neg'])
    pprint.pprint(dual_sol_bnds['dxgas']['right_pos']-dual_sol_bnds['gas']['right_neg'])
    pprint.pprint(dual_sol_bnds['dxgas']['top_pos']-dual_sol_bnds['gas']['top_neg'])
    pprint.pprint(dual_sol_bnds['dxgas']['bottom_pos']-dual_sol_bnds['gas']['bottom_neg'])
    print("dtgas")
    pprint.pprint(dual_sol_bnds['dtgas']['left_pos']-dual_sol_bnds['gas']['left_neg'])
    pprint.pprint(dual_sol_bnds['dtgas']['right_pos']-dual_sol_bnds['gas']['right_neg'])
    pprint.pprint(dual_sol_bnds['dtgas']['top_pos']-dual_sol_bnds['gas']['top_neg'])
    pprint.pprint(dual_sol_bnds['dtgas']['bottom_pos']-dual_sol_bnds['gas']['bottom_neg'])
    print("cer")
    pprint.pprint(dual_sol_bnds['cer']['left_pos']-dual_sol_bnds['cer']['left_neg'])
    pprint.pprint(dual_sol_bnds['cer']['right_pos']-dual_sol_bnds['cer']['right_neg'])
    pprint.pprint(dual_sol_bnds['cer']['top_pos']-dual_sol_bnds['cer']['top_neg'])
    pprint.pprint(dual_sol_bnds['cer']['bottom_pos']-dual_sol_bnds['cer']['bottom_neg'])
    gas.coeffs = x[:10]
    cer.coeffs = x[10:20]
    xt_part = [(x, t) for x in X_part[0] for t in T_part[0]]
    # print(gas(xt_part)[:,0])
    zz = dual_sol_bnds['gas']
    u_pos = sc.hstack([zz['left_pos'],
                       zz['right_pos'],
                       zz['top_pos'],
                       zz['bottom_pos']])
    u_neg = sc.hstack([zz['left_neg'],
                       zz['right_neg'],
                       zz['top_neg'],
                       zz['bottom_neg']])
    gas_vals = gas(coords)[:, 0]
    for i in range(len(coords)):
        print(coords[i], vals[i], gas_vals[i], u_pos[i], u_neg[i])
    i = 0


