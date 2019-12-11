import scipy as sc
from itertools import *
from functools import *
from pprint import pprint
import utils as ut
import lp_utils as lut

from constants import *
from polynom import Polynom, Context

regsize = 0
tgp = Polynom(2, 3)
tcp = Polynom(3, 3)
tgr = Polynom(2, 3)
tcr = Polynom(3, 3)

context = Context()
context.assign(tgp)
context.assign(tcp)
context.assign(tgr)
context.assign(tcr)

print([v.coeff_size for v in context.lvariables])
regsize = sum([v.coeff_size for v in context.lvariables])
print(regsize)

funcs = dict({
    'gas2gas' : lambda x: (tgp(x[:-1]) - tcp(x)) * coef["k1"] + coef["k2"] * (tgp(x[:-1], [1, 0]) * coef["wg"] + tgp(x[:-1], [0, 1])),
    'gas2cer' : lambda x: (tgp(x[:-1]) - tcp(x)) * coef["alpha"] - coef["lam"] * tcp(x, [0, 0, 1]),
    'cer3'    : lambda x: tcp(x,[0, 1, 0]) - coef["a"]*(tcp(x,[0,0,2]) + 2/R[3] * tcp(x,[0,0,1])),
    'cer2'    : lambda x: tcp(x,[0, 1, 0]) - coef["a"]*(tcp(x,[0,0,2]) + 2/R[2] * tcp(x,[0,0,1])),
    'cer1'    : lambda x: tcp(x,[0, 1, 0]) - coef["a"]*(tcp(x,[0,0,2]) + 2/R[1] * tcp(x,[0,0,1])),
    'cer0'    : lambda x: tcp(x,[0, 0, 1]),
    'gas'     : lambda x: tgp(x[:-1]),
    'cer'     : lambda x: tcp(x),
})

funcsr = dict({
    'gas2gas' : lambda x: (tgr(x[:-1]) - tcr(x)) * coef["k1"] - coef["k2"] * (tgr(x[:-1], [1, 0]) * coef["wg"] + tgr(x[:-1], [0, 1])),
    'gas2cer' : lambda x: (tgr(x[:-1]) - tcr(x)) * coef["alpha"] - coef["lam"] * tcr(x, [0, 0, 1]),
    'cer3'    : lambda x: tcr(x,[0, 1, 0]) - coef["a"]*(tcr(x,[0,0,2]) + 2/R[3] * tcr(x,[0,0,1])),
    'cer2'    : lambda x: tcr(x,[0, 1, 0]) - coef["a"]*(tcr(x,[0,0,2]) + 2/R[2] * tcr(x,[0,0,1])),
    'cer1'    : lambda x: tcr(x,[0, 1, 0]) - coef["a"]*(tcr(x,[0,0,2]) + 2/R[1] * tcr(x,[0,0,1])),
    'cer0'    : lambda x: tcr(x,[0, 0, 1]),
    'gas'     : lambda x: tgr(x[:-1]),
    'cer'     : lambda x: tcr(x),
})


balance_coeff = 20
temp_coeff = 10

eq_resid = dict({
    'gas2gas' : balance_coeff,
    'gas2cer' : balance_coeff,
    'cer3'    : balance_coeff,
    'cer2'    : balance_coeff,
    'cer1'    : balance_coeff,
    'cer0'    : balance_coeff,
    'gas'     : temp_coeff,
    'cer'     : temp_coeff,
})




def parse_reg(pr):
    if type(pr) is str:
        return pr
    else:
        return pr[0]+pr[1][0] + "_" + str(pr[1][1])

def make_coords(ids,type):
    i, j = ids
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

def shifted(cffs,shift):
    psize = len(cffs[1:])
    lzeros = sc.zeros(psize * shift)
    rzeros = sc.zeros((max_reg - shift-1) * psize)
    cffs = sc.hstack([cffs[0],lzeros,cffs[1:],rzeros])
    return cffs
    

def count_eq(eq,rg, val):
    ind = make_id(rg[1][1])
    crds = make_coords(rg[1][1],rg[0])
    if rg[1][0].startswith('cer0'):
        crds = sc.hstack([crds,sc.full(len(crds),R[0])])
    elif rg[1][0].startswith('cer1'):
        crds = sc.hstack([crds,sc.full(len(crds),R[1])])
    elif rg[1][0].startswith('cer2'):
        crds = sc.hstack([crds,sc.full(len(crds),R[2])])
    else:# rg[1][0].startswith('cer3'):
        crds = sc.hstack([crds,sc.full((len(crds),1),R[3])])

    if rg[1][0].endswith("_p"):
        fnc = funcs
    elif rg[1][0].endswith("_r"):
        fnc = funcsr
    g = lambda x:shifted(fnc[eq](x),ind)
    print(eq)
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
    if eq in ['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0']:
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

        crds = make_coords(regs[0][1][1],regs[0][0])
        crds1 = make_coords(regs[1][1][1],regs[1][0])

        g = lambda x:"({}) ({})".format("{:3.2f},{:3.2f}".format(*x[0]),"{:3.2f},{:3.2f}".format(*x[1]))
        g1 = lambda x:fnc[eq](x)[0]
        g2 = lambda x:fnc1[eq](x)[0]
        print("\n".join(list(map(g,zip(crds,crds1)))))

    return out



def main():

    print ("PREPARE_PROBLEM")
    q = sc.vstack(list(starmap(parse,
                         chain(
                             ut.construct_mode(['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0'],
                                               'base_1', 1, "l",
                                               ['gas_p', 'cer_p']),
                             ut.construct_mode(['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0'],
                                               'base_2', xreg, "r",
                                               ['gas_r', 'cer_r']),
                             ut.intemod_constraints(['cer'], "cer_p", "cer_r")))))

    x,dx,dz = lut.slvlprd(q, regsize*max_reg+1, TGZ,False)

    pc = sc.split(x[:-1],max_reg)

    sc.savetxt("poly_coeff",pc)
    sc.savetxt("resd",dz.reshape(-1,1))

if __name__ == "__main__":
    main()

         
