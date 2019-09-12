import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import dbalance as db
import utils as ut
from polynom import Context, Polynom

splitter = (0, 17, 33, 50)

def print_lang_mult(deal_sol_bnds):
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

def slice(i,j):
    i_part0 = splitter[i]
    i_part1 = splitter[i + 1]
    j_part0 = splitter[j]
    j_part1 = splitter[j + 1]
    return i_part0, i_part1, j_part0, j_part1

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
    vbounds = sc.vstack([(xt_vals[i, :]+xt_vals[i+1, :])/2 for i in splitter[1:-1]])
    hbounds = sc.vstack([(xt_vals[:, i]+xt_vals[:, i+1])/2 for i in splitter[1:-1]])
    regs = list()
    for i in range(xreg):
        regs.append([])
        for j in range(treg):
            i_part0, i_part1, j_part0, j_part1 = slice(i, j)
            sc.set_printoptions(precision=3, linewidth=110)
            reg = xt_vals[i_part0:i_part1, j_part0:j_part1]
            coords = ut.left_boundary_coords((X_part[i], T_part[j]))

            reg = xt_vals[i_part0:i_part1, j_part0:j_part1]
            coords = ut.boundary_coords((X_part[i], T_part[j]))
            # if i == 0:
            #     lvals = xt_vals[0,j_part0:j_part1]
            # else:
            #     lvals = vbounds[i-1][j_part0:j_part1]
            # if i < xreg - 1:
            #     rvals = vbounds[i][j_part0][j_part1]

            vals = sc.hstack([reg[0, :], reg[-1, :], reg[:, 0], reg[:, -1]])
            vgas = vals
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
            gas.coeffs = x[:10]
            cer.coeffs = x[10:20]
            xt_part = [(x, t) for x in X_part[0] for t in T_part[0]]
            zz = dual_sol_bnds['gas']
            gas_vals = gas(coords)[:, 0]
            dxgas_vals = gas(coords, [1, 0])[:, 0]
            dtgas_vals = gas(coords, [0, 1])[:, 0]
            cer_vals = cer(coords)[:, 0]
            dtcer_vals = cer(coords, [0, 1])[:, 0]
            print(dxgas_vals)
            print(dtgas_vals)
            regs[i].append(dict())
            # regs[i].append(dict())
            regs[i][j]['xd'] = dual_sol_bnds
            regs[i][j]['opt'] = x[-1]
            # regs[i][j]['gas']=dict()
            # regs[i][j]['gas']['left'] = sc.vstack()
    # print(regs[0][0])
    # print (regs[0][0][:,3]+regs[0][1][:,3])
    # sc.savetxt('xd',sc.vstack([regs[0][0]['xd'],regs[0][1]['xd']]).transpose())
    # print(regs[0][0]['xd']['gas']['right_neg'])
    # from pprint import pprint
    # pprint(regs[0][1]['xd']['gas'])
    # pprint(regs[1][0]['xd']['gas'])
    for i in range(xreg):
        for j in range(treg):
            for el in regs[i][j]['xd'].keys():
                for pivot in regs[i][j]['xd'][el].keys():
                    if sum(abs(regs[i][j]['xd'][el][pivot]))>0:
                        print(i,j,el,pivot,regs[i][j]['xd'][el][pivot])
