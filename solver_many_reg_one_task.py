import scipy as sc
from itertools import *
#from numpy.lib import recfunctions as rfn

# import utils as ut
# import lp_utils as lut

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


def add_constraints_internal(chain, coords, etype):
    add_constraints_block(chain, coords, 'i', etype)

def add_one_constraints_boundary(chain, coords, etype,exist_directions):
    for ptype in exist_directions:
        add_constraints_block(chain, coords, ptype)



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
        
# tg, tc = ut.make_gas_cer_pair(2, 3)

# funcs = {
#     b'balance_eq1': lambda x:(tg(x) - tc(x)) * coef["k1"] +
#                              coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
#     b'balance_eq3': lambda x:(tg(x) - tc(x)) * coef["k1"] -
#                              coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
#     b'balance_eq2': lambda x: (tg(x) - tc(x)) * coef["k1"] - coef["k3"] * tc(x, [0, 1]),
#     b'gas': lambda x: tg(x),
#     b'dxgas': lambda x: tg(x, [1, 0]),
#     b'dtgas': lambda x: tg(x, [0, 1]),
#     b'cer': lambda x: tc(x),
#     b'dtcer': lambda x: tc(x, [0, 1]),
# }


xdop = 1800
xreg, treg = 3, 3
cnt_var = 2
degree = 3
X = sc.linspace(0, 1, 50)
T = sc.linspace(0, 1, 50)

X_part = sc.split(X, (17, 33))
T_part = sc.split(T, (17, 33))

poly_name = lambda *pars: "{}_{}{}{}".format(*pars)


from pprint import pprint

print("Балансовые уравнения прямой режим")
equations_primal = ['balance_eq1','balance_eq3']
polynoms_primal = ["gas_p","cer_p"]
pprint(list(product(equations_primal,
               zip(repeat(polynoms_primal),
                   product(range(xreg),range(treg))))))

print("Стартовые ограничения")
equations_primal = ['gas']
polynoms_primal = ["gas_p","base_1"]
pprint(list(product(equations_primal,
               zip(repeat(polynoms_primal),
                   product(range(1),range(treg))))))


print("Балансовые уравнения обратный режим")
equations_rever = ['balance_eq2','balance_eq3']
polynoms_rever = ["gas_r","cer_r"]
pprint(list(product(equations_rever,
               zip(repeat(polynoms_rever),
                   product(range(xreg),range(treg))))))

print("Стартовые ограничения")
equations_primal = ['gas']
polynoms_primal = ["gas_r","base_2"]
pprint(list(product(equations_primal,
               zip(repeat(polynoms_primal),
                   product(range(1),range(treg))))))

            
# balance = chain(zip(repeat('balance_eq1'),descr_poly("gas"),descr_poly("cer")),
#                 zip(repeat('balance_eq3'),descr_poly("gas"),descr_poly("cer")))

# gas_p_start = zip(repeat('gas'),

# from pprint import pprint
# print("\n".join((starmap(lambda *z: " ".join(z), balance))))





