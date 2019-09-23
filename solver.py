import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from numpy.lib import recfunctions as rfn
import dbalance as db
import utils as ut
from polynom import Context, Polynom

length = 1
time = 50

TGZ = 1800
TBZ = 778.17

coef = dict()
coef["alpha"] = 0.027 * time
coef["fyd"] = 2262.0 * length
coef["po"] = 1.0
coef["fgib"] = 5.0
coef["wg"] = 1.5 * time
coef["cg"] = 0.3
coef["lam"] = 0.0038 * time * length
coef["a"] = 2.3e-6 * time
coef["ck"] = 0.3
coef["qmacyd"] = 29405.0 * length

coef["k1"] = coef["alpha"] * coef["fyd"]
coef["k2"] = coef["po"] * coef["fgib"] * coef["cg"]
coef["k3"] = coef["ck"] * coef["qmacyd"]

tbltype = sc.dtype([('coord', sc.float64, 2),
                    ('sign', 'S1'),
                    ('ptype', 'S1'),
                    ('etype', 'S11'),
                    ('test_val', sc.float64),
                    ('coeff', sc.float64, 2 * 10)])


def add_constraints_block(chain, coords, sign, ptype, etype, test_val):
    task = sc.zeros(len(coords[ptype]), tbltype)
    task['coord'] = coords[ptype]
    task['sign'] = sign
    task['ptype'] = ptype
    task['etype'] = etype
    task['test_val'] = test_val
    chain.append(task)


def add_constraints_residual(chain, coords, ptype, etype, test_val = 0):
    add_constraints_block(chain, coords, '+', ptype, etype, test_val)
    add_constraints_block(chain, coords, '-', ptype, etype, test_val)


def add_constraints_internal(chain, coords, etype):
    add_constraints_residual(chain, coords, 'i', etype)


def add_constraints_boundary(chain, coords, etype, test_val):
    for ptype in 'lrtb':
        add_constraints_residual(chain, coords, ptype, etype, test_val[ptype])


tg, tc = ut.make_gas_cer_pair(2, 3)

test_val = {
    "l": 0,
    "r": 0,
    "t": 0,
    "b": 0
}


funcs = dict()
funcs['balance_eq1'] = sc.vectorize(
    lambda x: (tg(x) - tc(x)) * coef["k1"] + coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
    signature="(m)->(k)")
funcs['balance_eq2'] = sc.vectorize(
    lambda x: (tg(x) - tc(x)) * coef["k1"] - coef["k3"] * tc(x, [0, 1]),
    signature="(m)->(k)")
funcs['gas'] = sc.vectorize(
    lambda x: tg(x),
    signature="(m)->(k)")
funcs['dxgas'] = sc.vectorize(
    lambda x: tg(x, [1, 0]),
    signature="(m)->(k)")
funcs['dtgas'] = sc.vectorize(
    lambda x: tg(x, [0, 1]),
    signature="(m)->(k)")
funcs['cer'] = sc.vectorize(
    lambda x: tc(x),
    signature="(m)->(k)")
funcs['dtcer'] = sc.vectorize(
    lambda x: tc(x, [0, 1]),
    signature="(m)->(k)")

xdop = 5
xreg, treg = 3, 3
cnt_var = 2
degree = 3
X = sc.linspace(0, 1, 50)
T = sc.linspace(0, 1, 50)

X_part = sc.split(X, (17, 33))
T_part = sc.split(T, (17, 33))

xt_vals_gas_prim = sc.repeat(sc.linspace(1, 0, 50), 50).reshape(-1, 50)
xt_vals_gas_revr = sc.repeat(sc.linspace(0, 1, 50), 50).reshape(-1, 50)

xt_vals_ones = sc.ones_like(xt_vals_gas_revr)
xt_vals_zero = sc.zeros_like(xt_vals_gas_revr)

xt_vals_gas = sc.vstack([xt_vals_ones, xt_vals_gas_prim])


xv, tv = sc.meshgrid(X_part[0],T_part[0])
xv = xv.reshape(-1)
tv = tv.reshape(-1)

xt = ut.boundary_coords((X_part[0], T_part[0]))
xt['i'] = sc.vstack([xv, tv]).T

chain = list()
add_constraints_internal(chain, xt, 'balance_eq1')
add_constraints_internal(chain, xt, 'balance_eq2')
add_constraints_boundary(chain, xt, 'gas', test_val)
add_constraints_boundary(chain, xt, 'dxgas', test_val)
add_constraints_boundary(chain, xt, 'dtgas', test_val)
add_constraints_boundary(chain, xt, 'cer', test_val)
add_constraints_boundary(chain, xt, 'dtcer', test_val)


qq = rfn.stack_arrays(chain, usemask=False)

p = qq[qq['etype'] == b'balance_eq1']


p['coeff'] = funcs['balance_eq1'](p['coord'])[:,1:]

exit()

for i in range(xreg):
    for j in range(treg):
        xv, tv = sc.meshgrid(X_part[i], T_part[j])
        xv = xv.reshape(-1)
        tv = tv.reshape(-1)

        xt = ut.boundary_coords((X_part[i], T_part[j]))
        xt['i'] = sc.vstack([xv, tv]).T

        i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j)

        reg = xt_vals[i_part0:i_part1, j_part0:j_part1]
        coords = ut.left_boundary_coords((X_part[i], T_part[j]))
        bound_vals = dict({
            'l': reg[0, :],
            'r': reg[-1, :],
            't': reg[:, 0],
            'b': reg[:, -1]
        })
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
                if sum(abs(regs[i][j]['xd'][el][pivot])) > 0:
                    print(i, j, el, pivot, regs[i][j]['xd'][el][pivot])
