import scipy as sc
from numpy.lib import recfunctions as rfn
import utils as ut
import lp_utils as lut

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
    task['val'] = vals[:, 0]
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

def slvrd(tsk):
    """Approximate solutioon with polynom by one residual"""
    delta_val = tsk['val']-tsk['test_val']
    prb = sc.hstack([delta_val.reshape((-1, 1)), tsk['coeff']])

    lp_dim = tc.coeff_size+tg.coeff_size+1
    return lut.slvlprd(prb, lp_dim, xdop)

def slvrdn(tsk, bnd):
    """Approximate solutioon with polynom by n residuals"""
    q = tsk
    ind_part = 1 + sc.arange(len(q[q['sign'] == b'+']))

    ind_full = sc.zeros_like(q['val'],sc.int64)

    mask = q['sign'] == b'+'
    ind_full[mask] = ind_part
    mask = q['sign'] == b'-'
    ind_full[mask] = ind_part

    delta_val = task[0][0]['val'] - task[0][0]['test_val']

    prb = sc.hstack([delta_val.reshape((-1, 1)), task[0][0]['coeff']])

    lp_dim = tc.coeff_size+tg.coeff_size
    return lut.slvlprdn(prb, lp_dim, ind_full, xdop, bnd)



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
        add_constraints_boundary(chain, xt, [b'gas', b'cer'],
                                 funcs, bound_vals, exist_directions)
        qq = rfn.stack_arrays(chain, usemask=False)
        task[i].append(qq)
        print("region",i,j)


x, xs, xd = slvrd(task[0][0])

xs = sc.array(list(xs.values())[0])
xd = sc.array(list(xd.values())[0])
task[0][0]['dual'] = xd
task[0][0]['slack'] = xs

q = task[0][0]

tg.coeffs=x[:10]

for coord in task[0][0]['coord']:
    print (tg(coord)[0])

bnd = x[-1]
# q = q[abs(q['dual'])>1e-6]
# out = sc.vstack([q['coord'][:, 0], q['coord'][:, 1], q['dual'], q['slack']])
# sc.savetxt('out', out.T)
#
# print(funcs[b'gas'](q['coord']))
# print(x)

# constraints = q[q['ptype'] != b'i']
#
# pos_part = constraints[constraints['sign'] == b'+']
# neg_part = constraints[constraints['sign'] == b'-']
# out = sc.vstack([pos_part['dual'], neg_part['dual'], pos_part['test_val']])
# sc.savetxt('out',out.T)

# print((q['val']-q['test_val']))
#
# print (sc.sum(x[:-1] * q['coeff'],axis=1))

resid = (q['val']-q['test_val']) + sc.sum(x[:-1] * q['coeff'],axis=1)
pos_part = resid[q['sign'] == b'+']
neg_part = resid[q['sign'] == b'-']
sc.savetxt('out',sc.vstack([pos_part, neg_part]).T)
resid = sc.ndarray.max(sc.vstack([pos_part, neg_part]).T,axis=1)

one_resid = x[-1], sum(resid)

x,xd,xs = slvrdn(task[0][0],bnd)
nresid = max(x[20:]), sum(x[20:])

print (sc.array([one_resid,nresid]))