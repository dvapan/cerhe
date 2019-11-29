import scipy as sc
from itertools import *
from functools import *
from pprint import pprint
import utils as ut
import lp_utils as lut

from constants import *


tg, tc, tgr, tcr = ut.make_gas_cer_quad(2, 3)

funcs = dict({
    'be1' : lambda x: (tg(x) - tc(x)) * coef["k1"] + coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
    'be3' : lambda x: (tg(x) - tc(x)) * coef["k1"] - coef["k3"] * tc(x, [0, 1]),
    'gas' : lambda x: tg(x),
    'dxgas' : lambda x: tg(x, [1, 0]),
    'dtgas' : lambda x: tg(x, [0, 1]),
    'cer' : lambda x: tc(x),
    'dtcer' : lambda x: tc(x, [0, 1]),
})
funcsr = dict({
    'be2' : lambda x: (tgr(x) - tcr(x)) * coef["k1"] - coef["k2"] * (tgr(x, [1, 0]) * coef["wg"] + tgr(x, [0, 1])),
    'be3' : lambda x: (tgr(x) - tcr(x)) * coef["k1"] - coef["k3"] * tcr(x, [0, 1]),
    'gas' : lambda x: tgr(x),
    'dxgas' : lambda x: tgr(x, [1, 0]),
    'dtgas' : lambda x: tgr(x, [0, 1]),
    'cer' : lambda x: tcr(x),
    'dtcer' : lambda x: tcr(x, [0, 1]),
})


balance_coeff = 20
temp_coeff = 10

eq_resid=dict({
    'be1': balance_coeff,
    'be2': balance_coeff,
    'be3': balance_coeff,
    'gas':temp_coeff,
    'dxgas':temp_coeff,
    'dtgas':temp_coeff,
    'cer':temp_coeff,
    'dtcer':temp_coeff,
})


def parse_reg(pr):
    if type(pr) is str:
        return pr
    else:
        return pr[0]+pr[1][0] + "_" + str(pr[1][1])

def make_coords(ids,type):
    i, j = ids
    xv, tv = sc.meshgrid(X_part[j], T_part[i])
    xv = xv.reshape(-1)
    tv = tv.reshape(-1)

    if type in "lr":
        xt = ut.boundary_coords((X_part[i], T_part[j]))[type]
    elif type in "tb":
        xt = ut.boundary_coords((X_part[j], T_part[i]))[type]        
    elif type == "i":
        xt = sc.vstack([xv, tv]).T
    elif type == "c":
        xt = None
    return xt

def shifted(cffs,shift):
    psize = len(cffs[1:])
    lzeros = sc.zeros(psize * shift)
    rzeros = sc.zeros((max_reg - shift-1) * psize)
    cffs = sc.hstack([cffs[0],lzeros,cffs[1:],rzeros])
    return cffs
    

def count_eq(eq,rg, val):
    ind = make_id(rg[1][1])
    crds = make_coords(rg[1][1],rg[0])
    if rg[1][0].endswith("_p"):
        fnc = funcs
    elif rg[1][0].endswith("_r"):
        fnc = funcsr
    g = lambda x:shifted(fnc[eq](x),ind)
    cffs = sc.vstack(list(map(g, crds)))
    r_cffs = sc.full((1,len(cffs[:,0])),1)
    cffs[:,0] =  val - cffs[:,0]
    cffs /= eq_resid[eq]
    cffs = sc.vstack([sc.hstack([ cffs,r_cffs.reshape((-1,1))]),
                      sc.hstack([-cffs,r_cffs.reshape((-1,1))])])        
    return cffs



def make_id(x):
    return x[0]*treg + x[1]
    
def parse(eq, regs):
    print(eq + " : " + " ".join(map(parse_reg, regs)))
    if eq in ['be1','be2','be3']:
        out = count_eq(eq,regs[0],0)
    elif regs[1][1][0].startswith('base'):
        if regs[0][1][0].endswith("_p"):
            T = TGZ
        else:
            T = TBZ
        out = count_eq(eq,regs[0],T)
    else:            
        x1 = count_eq(eq,regs[0],0)
        x2 = count_eq(eq,regs[1],0)
        out = x2 - x1
        out[:,-1] = x1[:,-1]
    return out



def main():

    print ("PREPARE_PROBLEM")
    q = sc.vstack(list(starmap(parse,
                         chain(
                             ut.construct_mode(['be1', 'be3'],
                                               'base_1', 1, "l",
                                               ['gas_p', 'cer_p']),
                             ut.construct_mode(['be2', 'be3'],
                                               'base_2', xreg, "r",
                                               ['gas_r', 'cer_r']),
                             ut.intemod_constraints(['cer'], "cer_p", "cer_r")))))

    x,dx,dz = lut.slvlprd(q, 40*max_reg+1, TGZ,False)

    pc = sc.split(x[:-1],max_reg)

    sc.savetxt("poly_coeff",pc)
    sc.savetxt("resd",dz.reshape(-1,1))

if __name__ == "__main__":
    main()

         
