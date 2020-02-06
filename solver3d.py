import numpy as np
from itertools import *
from functools import *
from pprint import pprint

import lp_utils as lut
from constants import *
from polynom import Polynom, Context

regsize = 0
tgp = Polynom(2, 5)
tcp = Polynom(3, 5)
tgr = Polynom(2, 5)
tcr = Polynom(3, 5)

context = Context()
context.assign(tgp)
context.assign(tcp)
context.assign(tgr)
context.assign(tcr)


def cer2cer(x,p):
    return p[1](x,[0, 1, 0]) - coef["a"]*(p[1](x,[0,0,2]) + 2/x[2] * p[1](x,[0,0,1]))

def cer2cerz(x,p):
    return p[1](x,[0, 0, 1])
    
    
def gas2gasr(x,p):
    return (p[0](x[:-1]) - p[1](x)) * coef["k1"] - coef["k2"] * (p[0](x[:-1], [1, 0]) * coef["wg"] + p[0](x[:-1], [0, 1]))

    
def gas2gasp(x,p):
    return (p[0](x[:-1]) - p[1](x)) * coef["k1"] + coef["k2"] * (p[0](x[:-1], [1, 0]) * coef["wg"] + p[0](x[:-1], [0, 1]))

def gas2cer(x, p):
    return (p[0](x[:-1]) - p[1](x)) * coef["alpha"] - coef["lam"] * p[1](x, [0, 0, 1])

def tcp2tcr(x, p):
    x1 = x[0],T[-1],x[1]
    r1 = tcp(x1)
    x2 = x[0],T[0],x[1]
    r2 = tcr(x2)
    return r2 - r1

def tcr2tcp(x, p):
    x1 = x[0],T[-1],x[1]
    r1 = tcr(x1)
    x2 = x[0],T[ 0],x[1]
    r2 = tcp(x2)
    return r2 - r1


balance_coeff = 10
temp_coeff = 1
cer_coeff = 0.001
prb_chain = []

coeffs = {
    "polynom"  : temp_coeff,
    "tcp2tcr"  : temp_coeff,
    "tcr2tcp"  : temp_coeff,
    "gas2gasp" : balance_coeff,
    "gas2gasr" : balance_coeff,
    "gas2cer"  : cer_coeff,
    "cer2cer"  : cer_coeff,
    "cer2cerz" : cer_coeff
}

def add_residual(var_num, monoms, val=0):
    prb_chain.append(sc.hstack([val,monoms,[1]]))
    prb_chain.append(sc.hstack([-val,-monoms,[1]]))

def add_residuals(var_num, domain, diff_eq, p=None, val=0):
    print(diff_eq.__name__)
    for x in domain:
        if diff_eq.__name__ == "polynom":
            r = diff_eq(x)
        else:
            r = diff_eq(x,p)
        r /= coeffs[diff_eq.__name__]
#        val /= coeffs[diff_eq.__name__]
        add_residual(var_num,r[1:],val)

def main():    
    pp = [tgp,tcp]
    pr = [tgr,tcr]

    var_num = 0
    for el in [tgp,tcp,tgr,tcr]:
        var_num+=el.coeff_size

    
    print(var_num,1)
    var_num+=1
    
    print ("primal process")
    add_residuals(var_num, product(X,T,R[:1]),gas2gasp,pp)
    add_residuals(var_num, product(X,T,R[:1]),gas2cer,pp)
    add_residuals(var_num, product(X,T,R[1:]),cer2cer,pp)
    # add_residuals(var_num, product(X,T,R[-1:]),cer2cerz,pp)    
    add_residuals(var_num, product(X[:1],T),tgp,pp,TGZ)
    add_residuals(var_num, product(X,R),tcp2tcr,pp)

    print ("reverse process")
    add_residuals(var_num, product(X,T,R[:1]),gas2gasr,pr)
    add_residuals(var_num, product(X,T,R[:1]),gas2cer,pr)
    add_residuals(var_num, product(X,T,R[1:]),cer2cer,pr)
    # add_residuals(var_num, product(X,T,R[-1:]),cer2cerz,pr)    
    add_residuals(var_num, product(X[-1:],T),tgr,pr,val=TBZ)
    add_residuals(var_num, product(X,R),tcr2tcp,pr)
        
    prb = sc.vstack(prb_chain)
    x,dx,dz = lut.slvlprd(prb, var_num, TGZ,False)
    pc = sc.split(x[:-1],max_reg)

    sc.savetxt("poly_coeff_3d",pc)

    

if __name__ == "__main__":
    main()

         
