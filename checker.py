import scipy as sc
from itertools import *
from functools import *
from pprint import pprint
import utils as ut
import lp_utils as lut
from constants import *

pc = sc.loadtxt("poly_coeff")

def parse_reg(pr):
    if type(pr) is str:
        return pr
    else:
        return pr[0]+pr[1][0] + "_" + str(pr[1][1])


def make_coords(ids,type):
    i, j = ids
    print(i,j)
    xv, tv = sc.meshgrid(X_part[i], T_part[j])
    xv = xv.reshape(-1)
    tv = tv.reshape(-1)

    if type in "lr":
        xt = ut.boundary_coords((X_part[i], T_part[j]))[type]
    elif type in "tb":
        xt = ut.boundary_coords((X_part[i], T_part[j]))[type]        
    elif type == "i":
        xt = sc.vstack([xv, tv]).T
    elif type == "c":
        xt = None
    return xt


def make_id(x):
    return x[0]*treg + x[1]
    
def parse(eq, regs):
    rg = regs[0]
    rg1 = regs[1]

    if max_reg != 1:
        gc,cc,gcr,ccr = sc.split(pc[make_id(rg[1][1])], 4)
    else:
        gc,cc,gcr,ccr = sc.split(pc, 4)
    tg, tc, tgr, tcr = ut.make_gas_cer_quad(2, 3, gc,cc,gcr,ccr)
    
    if max_reg != 1:
        gc1,cc1,gcr1,ccr1 = sc.split(pc[make_id(rg1[1][1])], 4)
    else:
        gc1,cc1,gcr1,ccr1 = sc.split(pc, 4)
    tg1, tc1, tgr1, tcr1 = ut.make_gas_cer_quad(2, 3, gc1,cc1,gcr1,ccr1)

    
    funcs = dict({
        'be1' : lambda x: (tg(x) - tc(x)) * coef["k1"] + coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
        'be3' : lambda x: (tg(x) - tc(x)) * coef["k1"] - coef["k3"] * tc(x, [0, 1]),
        'gas' : lambda x: tg(x),
        'cer' : lambda x: tc(x),
    })
    funcsr = dict({
        'be2' : lambda x: (tgr(x) - tcr(x)) * coef["k1"] - coef["k2"] * (tgr(x, [1, 0]) * coef["wg"] + tgr(x, [0, 1])),
        'be3' : lambda x: (tgr(x) - tcr(x)) * coef["k1"] - coef["k3"] * tcr(x, [0, 1]),
        'gas' : lambda x: tgr(x),
        'cer' : lambda x: tcr(x),
    })

    funcs1 = dict({
        'gas' : lambda x: tg1(x),
        'cer' : lambda x: tc1(x),
    })
    funcsr1 = dict({
        'gas' : lambda x: tgr1(x),
        'cer' : lambda x: tcr1(x),
    })


    if eq not in ['be1','be2','be3']:
        print(eq + " : " + " ".join(map(parse_reg, regs)))
    if eq in ['be1','be2','be3']:
        pass
    elif regs[1][1][0].startswith('base'):
        if regs[0][1][0].endswith("_p"):
            T = TGZ
        else:
            T = TBZ

        ind = make_id(rg[1][1])
        crds = make_coords(rg[1][1],rg[0])
        if rg[1][0].endswith("_p"):
            fnc = funcs
        elif rg[1][0].endswith("_r"):
            fnc = funcsr
        g = lambda x:"{:7.3f} {:7.3f}".format(fnc[eq](x)[0],T)
        print("\n".join(list(map(g, crds))))      

    else:
        if regs[0][1][0].endswith("_p"):
            T = TGZ
        else:
            T = TBZ

        ind = make_id(rg[1][1])
        crds = make_coords(rg[1][1],rg[0])
        crds1 = make_coords(rg1[1][1],rg1[0])
        if rg[1][0].endswith("_p"):
            fnc = funcs
        elif rg[1][0].endswith("_r"):
            fnc = funcsr

        if rg1[1][0].endswith("_p"):
            fnc1 = funcs1
        elif rg1[1][0].endswith("_r"):
            fnc1 = funcsr1

        g = lambda x:"{:4.3f} {:4.3f} {:4.3f}".format(x[2],x[3],abs(x[2]-x[3]))
        g1 = lambda x:fnc[eq](x)[0]
        g2 = lambda x:fnc1[eq](x)[0]
        print("\n".join(list(map(g,zip(crds,crds1,map(g1,crds),map(g2,crds1))))))




def main():
    list(starmap(parse,
                 chain(
                     ut.construct_mode(['be1', 'be3'],
                                       'base_1', 1, "l",
                                               ['gas_p', 'cer_p']),
                     ut.construct_mode(['be2', 'be3'],
                                       'base_2', xreg, "r",
                                       ['gas_r', 'cer_r']),
                     ut.intemod_constraints(['cer'], "cer_p", "cer_r"))))


if __name__ == "__main__":
    main()


