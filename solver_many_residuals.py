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
                    ('ind', sc.int64),
                    ('test_val', sc.float64),
                    ('val', sc.float64),
                    ('coeff', sc.float64, 2 * 10),
                    ('dual', sc.float64),
                    ('slack',sc.float64)])


def add_constraints_block(chain, coords, sign, ptype, etype, fnc, test_val):
    task = sc.zeros(len(coords[ptype]), tbltype)
    task['coord'] = coords[ptype]
    task['sign'] = sign
    if sign == '+':
        signv = 1
    elif sign == '-':
        signv = -1

    task['ptype'] = ptype
    task['etype'] = etype
    task['test_val'] = signv*test_val
    vals = signv*fnc[etype](coords[ptype])
    task['coeff'] = vals[:, 1:]
    task['val'] = vals[:, 1]
    chain.append(task)


def add_constraints_residual(chain, coords, ptype, etype, fnc, test_val = 0):
    add_constraints_block(chain, coords, '+', ptype, etype, fnc, test_val)
    add_constraints_block(chain, coords, '-', ptype, etype, fnc, test_val)


def add_constraints_internal(chain, coords, etype, fnc):
    add_constraints_residual(chain, coords, 'i', etype, fnc)


def add_one_constraints_boundary(chain, coords, etype, fnc, test_val,exist_directions):
    for ptype in exist_directions:
        add_constraints_residual(chain, coords, ptype, etype, fnc, test_val[ptype])
def add_constraints_boundary(chain, coords, etypes, fnc, test_val, exist_directions):
    for etype in etypes:
        add_one_constraints_boundary(chain, coords, etype, fnc, test_val, exist_directions)

tg, tc = ut.make_gas_cer_pair(2, 3)


funcs = dict()
funcs[b'balance_eq1'] = sc.vectorize(
    lambda x: (tg(x) - tc(x)) * coef["k1"] + coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
    signature="(m)->(k)")
funcs[b'balance_eq2'] = sc.vectorize(
    lambda x: (tg(x) - tc(x)) * coef["k1"] - coef["k3"] * tc(x, [0, 1]),
    signature="(m)->(k)")
funcs[b'gas'] = sc.vectorize(
    lambda x: tg(x),
    signature="(m)->(k)")
funcs[b'dxgas'] = sc.vectorize(
    lambda x: tg(x, [1, 0]),
    signature="(m)->(k)")
funcs[b'dtgas'] = sc.vectorize(
    lambda x: tg(x, [0, 1]),
    signature="(m)->(k)")
funcs[b'cer'] = sc.vectorize(
    lambda x: tc(x),
    signature="(m)->(k)")
funcs[b'dtcer'] = sc.vectorize(
    lambda x: tc(x, [0, 1]),
    signature="(m)->(k)")

xdop = 1
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

xt_vals_gas = sc.vstack([xt_vals_gas_prim, xt_vals_gas_revr])

# TEST fortran counted data
# at, atr = sc.array_split(sc.loadtxt("tests/rain33.dat"), 2)
# at = at.reshape((3, 3, 2, 10))
# atr = atr.reshape((3, 3, 2, 10))
# tgas1, tcer1 = ut.make_gas_cer_pair(2, 3, at[0][0][0], at[0][0][1])
# tgas2, tcer2 = ut.make_gas_cer_pair(2, 3, at[1][0][0], at[1][0][1])
#
# i, j = 0, 0
# xv, tv = sc.meshgrid(X_part[i], T_part[j])
# xv = xv.reshape(-1)
# tv = tv.reshape(-1)
#
# xt = ut.boundary_coords((X_part[i], T_part[j]))
# xt['i'] = sc.vstack([xv, tv]).T
# fnc = sc.vectorize(
#     lambda x: tgas1(x),
#     signature="(m)->(k)")
# print(fnc(xt['i']))
# exit()
# END TEST BLOCK
task = list()
from pprint import pprint
for i in range(1):
    task.append(list())
    for j in range(2):
        xv, tv = sc.meshgrid(X_part[i], T_part[j])
        xv = xv.reshape(-1)
        tv = tv.reshape(-1)

        xt = ut.boundary_coords((X_part[i], T_part[j]))
        xt['i'] = sc.vstack([xv, tv]).T

        i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j)

        reg = xt_vals_gas[i_part0:i_part1, j_part0:j_part1]
        bound_vals = dict({
            'l': reg[0, :],
            'r': reg[-1, :],
            't': reg[:, 0],
            'b': reg[:, -1]
        })

        if j == 0:
            exist_directions = "lrb"
        else:
            exist_directions = "lrtb"
        chain = list()
        add_constraints_internal(chain, xt, b'balance_eq1', funcs)
        add_constraints_internal(chain, xt, b'balance_eq2', funcs)
        add_constraints_boundary(chain, xt, [b'gas', b'cer'], #[b'gas', b'dxgas', b'dtgas', b'cer', b'dtcer'],
                                 funcs, bound_vals, exist_directions)
        qq = rfn.stack_arrays(chain, usemask=False)
        ln = len(qq[qq['sign'] == b'-'])
        qq[qq['sign'] == b'-']['ind'] = sc.arange(ln)
        qq[qq['sign'] == b'+']['ind'] = sc.arange(ln)
        task[i].append(qq)
        print("region", i, j)

q = task[0][0]
ind_part = 1 + sc.arange(len(q[q['sign'] == b'+']))

ind_full = sc.zeros_like(q['val'])


mask = q['sign'] == b'+'
ind_full[mask] = ind_part
mask = q['sign'] == b'-'
ind_full[mask] = ind_part

task[0][0]['ind'] = ind_full
# print(task[0][0][q['ptype'] != b'i'])

delta_val = task[0][0]['val'] - task[0][0]['test_val']

prb = sc.hstack([delta_val.reshape((-1, 1)), task[0][0]['coeff']])

# exit()

lp_dim = tc.coeff_size+tg.coeff_size
x, xd, xs = solve_linear(prb, lp_dim, task[0][0]['ind'], xdop)


xs = sc.array(list(xs.values())[0])
xd = sc.array(list(xd.values())[0])
task[0][0]['dual'] = xd
task[0][0]['slack'] = xs

q = task[0][0]
# q = q[abs(q['dual'])>1e-6]
out = sc.vstack([q['coord'][:, 0], q['coord'][:, 1], q['dual'], q['slack']])
sc.savetxt('out', out.T)

print(funcs[b'gas'](q['coord']))
print(x[lp_dim:])
print(max(x[lp_dim:]))

# constraints = q[q['ptype'] != b'i']
#
# pos_part = constraints[constraints['sign'] == b'+']
# neg_part = constraints[constraints['sign'] == b'-']
#
# out = sc.vstack([pos_part['dual'], neg_part['dual'], pos_part['test_val']])
# sc.savetxt('out',out.T)