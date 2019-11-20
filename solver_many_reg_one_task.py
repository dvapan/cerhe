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
                    ('sign', sc.int64),
                    ('ptype', 'S1'),
                    ('etype', 'S11'),
                    ('region_id', sc.int64),
                    ('constr_id', sc.int64),
])


def add_constraints_block(chain, coords, ptype, etype):
    task = sc.zeros(len(coords[ptype]), tbltype)
    task['coord'] = coords
    task['ptype'] = ptype
    task['etype'] = etype
    chain.append(task)


def add_constraints_internal(chain, coords, etype, fnc):
    add_constraints_block(chain, coords, 'i', etype)

def add_one_constraints_boundary(chain, coords, etype, fnc, test_val,exist_directions, bnd_idx):
    for ptype in exist_directions:
        add_constraints_residual(chain, coords, ptype, etype, fnc, test_val[ptype], bnd_idx[ptype])
def add_constraints_boundary(chain, coords, etypes, fnc, test_val, exist_directions, bnd_idx):
    for etype in etypes:
        add_one_constraints_boundary(chain, coords, etype, fnc, test_val[etype], exist_directions, bnd_idx[etype])

tg, tc = ut.make_gas_cer_pair(2, 3)


funcs = {
    b'balance_eq1': lambda x:(tg(x) - tc(x)) * coef["k1"] +
                             coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
    b'balance_eq3': lambda x:(tg(x) - tc(x)) * coef["k1"] -
                             coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
    b'balance_eq2': lambda x: (tg(x) - tc(x)) * coef["k1"] - coef["k3"] * tc(x, [0, 1]),
    b'gas': lambda x: tg(x),
    b'dxgas': lambda x: tg(x, [1, 0]),
    b'dtgas': lambda x: tg(x, [0, 1]),
    b'cer': lambda x: tc(x),
    b'dtcer': lambda x: tc(x, [0, 1]),
}
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

    ind_full = sc.zeros_like(q['val'], sc.int64)

    mask = q['sign'] == b'+'
    ind_full[mask] = ind_part
    mask = q['sign'] == b'-'
    ind_full[mask] = ind_part

    delta_val = task[0][0]['val'] - task[0][0]['test_val']

    prb = sc.hstack([delta_val.reshape((-1, 1)), task[0][0]['coeff']])

    lp_dim = tc.coeff_size+tg.coeff_size
    return lut.slvlprdn(prb, lp_dim, ind_full, xdop, bnd)

def slvrd_coord(tsk):
    pass

xdop = 1800
xreg, treg = 3, 3
cnt_var = 2
degree = 3
X = sc.linspace(0, 1, 50)
T = sc.linspace(0, 1, 50)

X_part = sc.split(X, (17, 33))
T_part = sc.split(T, (17, 33))

xt_vals_gas_prim = sc.repeat(sc.linspace(TGZ, TBZ, 50), 50).reshape(-1, 50)
xt_vals_gas_revr = sc.repeat(sc.linspace(TBZ, TGZ, 50), 50).reshape(-1, 50)

xt_vals_ones = sc.ones_like(xt_vals_gas_revr)
xt_vals_zero = sc.zeros_like(xt_vals_gas_revr)

xt_vals_gas = sc.vstack([xt_vals_gas_prim, xt_vals_gas_revr])

def lft_val(x):
    return x[0, :]
def rht_val(x):
    return x[-1, :]
def top_val(x):
    return x[:, 0]
def dwn_val(x):
    return x[:, -1]

########################################GAS2CERAMIC
for i in range(treg):
    i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j)
    reg = xt_vals_gas[i_part0:i_part1, j_part0:j_part1]
    bnd_val[i][0][b'gas']['l'] = TGZ




residual = 0
from pprint import pprint
task = list()
for i in range(treg):
    task.append(list())
    for j in range(xreg):
        print("prepare region", i, j, ":", exist_directions[i][j])
        xv, tv = sc.meshgrid(X_part[j], T_part[i])
        xv = xv.reshape(-1)
        tv = tv.reshape(-1)

        xt = ut.boundary_coords((X_part[j], T_part[i]))
        xt['i'] = sc.vstack([xv, tv]).T

        chain = list()
        add_constraints_internal(chain, xt, b'balance_eq1', funcs)
        add_constraints_internal(chain, xt, b'balance_eq2', funcs)
        add_constraints_boundary(chain, xt, [b'gas', b'cer'],
                                 funcs, bnd_val[i][j], exist_directions[i][j], bnd_idx[i][j])
        qq = rfn.stack_arrays(chain, usemask=False)
        task[i].append(qq)


#######################################CERAMIC2GAS
bnd_val = list()
exist_directions = list()
bnd_idx = list()
constr_id = 0
for i in range(treg):
    bnd_val.append(list())
    exist_directions.append(list())
    bnd_idx.append(list())
    for j in range(xreg):
        bnd_val[i].append({b'gas': dict(), b'cer': dict()})
        bnd_idx[i].append({b'gas': dict(), b"cer": dict()})
        exist_directions[i].append(list())

        i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j)

        reg = xt_vals_gas_revr[i_part0:i_part1, j_part0:j_part1]

for i in range(treg):
    bnd_val.append(list())
    exist_directions.append(list())
    bnd_idx.append(list())
    for j in range(xreg):
        bnd_val[i].append({b'gas': dict(), b'cer': dict()})
        bnd_idx[i].append({b'gas': dict(), b"cer": dict()})
        exist_directions[i].append(list())

        i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j)

        reg = xt_vals_gas_revr[i_part0:i_part1, j_part0:j_part1]

        if j > 0:
            i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j - 1)
            regl = xt_vals_gas_revr[i_part0:i_part1, j_part0:j_part1]
            bnd_val[i][j][b'gas']['l'] = (lft_val(reg) + rht_val(regl)) / 2
#            bnd_val[i][j][b'cer']['l'] = (lft_val(reg) + rht_val(regl)) / 2
            bnd_val[i][j][b'cer']['l'] = sc.zeros_like(lft_val(reg))
            exist_directions[i][j].append("l")
            bnd_idx[i][j][b'gas']['l'] = bnd_idx[i][j - 1][b'gas']['r']
            bnd_idx[i][j][b'cer']['l'] = bnd_idx[i][j - 1][b'cer']['r']
        else:
            bnd_val[i][j][b'gas']['l'] = lft_val(reg)
#            bnd_val[i][j][b'cer']['l'] = lft_val(reg)
            bnd_val[i][j][b'cer']['l'] = sc.zeros_like(lft_val(reg))
            exist_directions[i][j].append("l")
            bnd_idx[i][j][b'gas']['l'] = -1
            bnd_idx[i][j][b'cer']['l'] = -1


        if j < xreg - 1:
            i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j + 1)
            regr = xt_vals_gas_revr[i_part0:i_part1, j_part0:j_part1]
            bnd_val[i][j][b'gas']['r'] = (rht_val(reg) + lft_val(regr)) / 2
#            bnd_val[i][j][b'cer']['r'] = (rht_val(reg) + lft_val(regr)) / 2
            bnd_val[i][j][b'cer']['r'] = sc.zeros_like(rht_val(reg))
            exist_directions[i][j].append("r")
            bnd_idx[i][j][b'gas']['r'] = sc.arange(constr_id,constr_id+len(bnd_val[i][j][b'gas']['r']))
            constr_id += len(bnd_val[i][j][b'gas']['r'])
            bnd_idx[i][j][b'cer']['r'] = sc.arange(constr_id, constr_id + len(bnd_val[i][j][b'gas']['r']))
            constr_id += len(bnd_val[i][j][b'gas']['r'])


        if i > 0:
            i_part0, i_part1, j_part0, j_part1 = ut.slice(i - 1, j)
            regt = xt_vals_gas_revr[i_part0:i_part1, j_part0:j_part1]
            bnd_val[i][j][b'gas']['t'] = (top_val(reg) + dwn_val(regt)) / 2
#            bnd_val[i][j][b'cer']['t'] = (top_val(reg) + dwn_val(regt)) / 2
            bnd_val[i][j][b'cer']['t'] = sc.zeros_like(top_val(reg))
            exist_directions[i][j].append("t")
            bnd_idx[i][j][b'gas']['t'] = bnd_idx[i-1][j][b'gas']['b']
            bnd_idx[i][j][b'cer']['t'] = bnd_idx[i - 1][j][b'cer']['b']
        if i < treg - 1:
            i_part0, i_part1, j_part0, j_part1 = ut.slice(i + 1, j)
            regb = xt_vals_gas_revr[i_part0:i_part1, j_part0:j_part1]
            bnd_val[i][j][b'gas']['b'] = (dwn_val(reg) + top_val(regb)) / 2
#            bnd_val[i][j][b'cer']['b'] = (dwn_val(reg) + top_val(regb)) / 2
            bnd_val[i][j][b'cer']['b'] = sc.zeros_like(dwn_val(reg))
            exist_directions[i][j].append("b")
            bnd_idx[i][j][b'gas']['b'] = sc.arange(constr_id,constr_id+len(bnd_val[i][j][b'gas']['b']))
            constr_id += len(bnd_val[i][j][b'gas']['b'])
            bnd_idx[i][j][b'cer']['b'] = sc.arange(constr_id,constr_id+len(bnd_val[i][j][b'gas']['b']))
            constr_id += len(bnd_val[i][j][b'gas']['b'])

residual = 0
from pprint import pprint
taskr = list()
for i in range(treg):
    taskr.append(list())
    for j in range(xreg):
        print("prepare region", i, j, ":", exist_directions[i][j])
        xv, tv = sc.meshgrid(X_part[j], T_part[i])
        xv = xv.reshape(-1)
        tv = tv.reshape(-1)

        xt = ut.boundary_coords((X_part[j], T_part[i]))
        xt['i'] = sc.vstack([xv, tv]).T

        add_constraints_internal(chain, xt, b'balance_eq3', funcs)
        add_constraints_internal(chain, xt, b'balance_eq2', funcs)
        add_constraints_boundary(chain, xt, [b'gas', b'cer'],
                                 funcs, bnd_val[i][j], exist_directions[i][j], bnd_idx[i][j])
        qq = rfn.stack_arrays(chain, usemask=False)
        taskr[i].append(qq)


solve=True
iter = 0

data = [1]

while solve:
    q = task[0][0]
    q1 = q[q['ptype'] == 'r']['val']
    q = task[0][1]
    q2 = q[q['ptype'] == 'l']['val']

    iter += 1
    residual = 0
    for i in range(treg):
        for j in range(xreg):
            print("solve region",i,j)
            x, xval = slvrd(task[i][j])

            task[i][j]['val'] = xval

            print(x[-1])
            residual = max(residual, x[-1])

            tgt, tct = ut.make_gas_cer_pair(2, 3, x[:10], x[10:-1])
            fnc = sc.vectorize(
                lambda x: tgt(x),
                signature="(m)->(k)")
            fnc2 = sc.vectorize(
                lambda x: tct(x),
                signature="(m)->(k)")

            eq = sc.vectorize(
                lambda x: coef["k2"] * (tgt(x, [1, 0]) * coef["wg"] + tgt(x, [0, 1])),
                signature="(m)->(k)")
            eq2 = sc.vectorize(
                lambda x:  coef["k1"] - coef["k3"] * tct(x, [0, 1]),
                signature="(m)->(k)")



    for i in range(treg):
        for j in range(xreg):
            print("solve region",i,j)
            x, xval = slvrd(taskr[i][j])

            taskr[i][j]['val'] = xval

            print(x[-1])
            residual = max(residual, x[-1])

            tgt, tct = ut.make_gas_cer_pair(2, 3, x[:10], x[10:-1])
            fnc = sc.vectorize(
                lambda x: tgt(x),
                signature="(m)->(k)")
            fnc2 = sc.vectorize(
                lambda x: tct(x),
                signature="(m)->(k)")

            eq = sc.vectorize(
                lambda x: coef["k2"] * (tgt(x, [1, 0]) * coef["wg"] + tgt(x, [0, 1])),
                signature="(m)->(k)")
            eq2 = sc.vectorize(
                lambda x:  coef["k1"] - coef["k3"] * tct(x, [0, 1]),
                signature="(m)->(k)")




    print("{:*^200}".format("ITER {}".format(iter)))
    print("{:*^200}".format("RESIDUAL: {}".format(residual)))
    print("{:*^200}".format("DELTA: {}".format(residual - data[-1])))
    data.append(residual)

    if abs(residual) < 1e-5:
        break
    print ("{:*^200}".format("UPDATE VALS"))

    N = 25
    new_task = list()
    reg_id = 0
    cnstr_count = 0
    for i in range(treg):
        for j in range(xreg):
            task[i][j]['region_id'] = reg_id
            q = sc.sort(task[i][j], order='dual')
            niter = 0
            for eq in q:
                if niter >= N:
                    break
                niter += 1
                new_task.append(eq)
                new_task[-1]['constr_id'] = -1
            reg_id += 1


    for i in range(treg):
        for j in range(xreg):
            for eq in task[i][j]:
                if eq['constr_id'] != -1:
                    cnstr_count += 1
                    new_task.append(eq)

    new_task = sc.array(new_task)

    prb = list()
    constr_idx = 0
    ccc= 0
    for eq in new_task:
        shift = eq['region_id']
        psize = len(eq['coeff'])
        lzeros = sc.zeros(psize * shift)
        rzeros = sc.zeros((reg_id - shift-1) * psize)
        pars = sc.zeros(constr_id)
        if eq['constr_id'] == -1:
            line = sc.hstack([[eq['test_val']], lzeros, eq['coeff'], rzeros, pars])
        else:
            pars[eq['constr_id']] = -eq['sign']
            line = sc.hstack([[0], lzeros, eq['coeff'], rzeros, pars])
        prb.append(line)
    prb = sc.vstack(prb)
    lp_dim = psize * (reg_id) + len(pars) + 1
    x, xd = lut.slvlprd(prb, lp_dim, xdop, True)

    sc.savetxt("outxd", xd.T,fmt="%+16.5f")
    sc.savetxt("outx", x.T,fmt="%+16.5f")
    exit()
    print ("{:*^200}".format("REMAKE TASK"))

    task = list()
    for i in range(treg):
        task.append(list())
        for j in range(xreg):
            print("prepare region", i, j, ":", exist_directions[i][j])
            xv, tv = sc.meshgrid(X_part[j], T_part[i])
            xv = xv.reshape(-1)
            tv = tv.reshape(-1)

            xt = ut.boundary_coords((X_part[j], T_part[i]))
            xt['i'] = sc.vstack([xv, tv]).T

            i_part0, i_part1, j_part0, j_part1 = ut.slice(i, j)

            reg = xt_vals_gas[i_part0:i_part1, j_part0:j_part1]
            bound_vals = dict({
                'l': reg[0, :],
                'r': reg[-1, :],
                't': reg[:, 0],
                'b': reg[:, -1]
            })

            chain = list()
            add_constraints_internal(chain, xt, b'balance_eq1', funcs)
            add_constraints_internal(chain, xt, b'balance_eq2', funcs)
            add_constraints_boundary(chain, xt, [b'gas', b'cer'],
                                     funcs, bnd_val[i][j], exist_directions[i][j])
            qq = rfn.stack_arrays(chain, usemask=False)
            qq['test_val'] = new_vals[i][j]
            task[i].append(qq)
