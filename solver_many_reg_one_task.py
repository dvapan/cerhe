import scipy as sc
from itertools import *
from pprint import pprint
import utils as ut
import lp_utils as lut

def parse_reg(pr):
    if type(pr) is str:
        return pr
    else:
        return pr[0] + "_" + "".join(map(str, pr[1]))

def make_coords(ids,type):
    i, j = ids
    xv, tv = sc.meshgrid(X_part[j], T_part[i])
    xv = xv.reshape(-1)
    tv = tv.reshape(-1)

    if type in "lrtb":
        xt = ut.boundary_coords((X_part[j], T_part[i]))[type]
    elif type == "i":
        xt = sc.vstack([xv, tv]).T
    return xt

def parse(eq, regs):
    crds = (make_coords(rg[1][1],rg[0]) for rg in regs)
    print(list(crds))
    print(eq + " : " + " ".join(map(parse_reg, regs)))
    return


X = sc.linspace(0, 1, 50)
T = sc.linspace(0, 1, 50)
X_part = sc.split(X, (17, 33))
T_part = sc.split(T, (17, 33))

tg, tc = ut.make_gas_cer_pair(2, 3)

pprint(list(starmap(parse,
                    chain(
                        ut.construct_mode(['be1', 'be3'],
                                       'base_1', 1, "l",
                                       ['gas_p', 'cer_p']),
                        ut.construct_mode(['be2', 'be3'],
                                       'base_2', ut.xreg, "r",
                                       ['gas_r', 'cer_r']),
                        ut.intemod_constraints(['cer'], "cer_p", "cer_r")))))
