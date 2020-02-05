import numpy as np
from itertools import *
from functools import *
from pprint import pprint

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
    x1 = x[0],T[ 0],x[1]
    r1 = tcp(x1)
    x2 = x[0],T[-1],x[1]
    r2 = tcr(x2)
    return r2 - r1

def tcr2tcp(x, p):
    x1 = x[0],T[-1],x[1]
    r1 = tcp(x1)
    x2 = x[0],T[ 0],x[1]
    r2 = tcr(x2)
    return r2 - r1


balance_coeff = 20
temp_coeff = 1
prb_chain = []

def add_residual(var_num, monoms, val=0):
    prb_chain.append(sc.hstack([val,monoms,[1]]))
    prb_chain.append(sc.hstack([-val,-monoms,[1]]))

    # s.CLP_addConstraint(var_num, np.arange(var_num,dtype=np.int32),
    #                     np.hstack([monoms,[1]]),val,np.inf)
    # s.CLP_addConstraint(var_num, np.arange(var_num,dtype=np.int32),
    #                     np.hstack([monoms,[-1]]),-np.inf,val)

def add_residuals(var_num, domain, diff_eq, p=None, val=0):
    print(diff_eq.__name__)
    for x in domain:
        if diff_eq.__name__ == "polynom":
            r = diff_eq(x)
        else:
            r = diff_eq(x,p)
        if diff_eq.__name__ in ["polynom", "tcp2tcr", "tcr2tcp"]:
            r /= temp_coeff
        elif diff_eq.__name__ in ["gas2cer","cer2cer", "cer2cerz"]:
            r /= 0.001
        else:
            r /= balance_coeff
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
    add_residuals(var_num, product(X,T,R[-1:]),gas2gasp,pp)
    add_residuals(var_num, product(X,T,R[-1:]),gas2cer,pp)
    add_residuals(var_num, product(X,T,R[1:-1]),cer2cer,pp)
    add_residuals(var_num, product(X,T,R[:1]),cer2cerz,pp)    
    add_residuals(var_num, product(X[:1],T),tgp,pp,TGZ)
    add_residuals(var_num, product(X,R),tcp2tcr,pp)

    print ("reverse process")
    add_residuals(var_num, product(X,T,R[-1:]),gas2gasr,pr)
    add_residuals(var_num, product(X,T,R[-1:]),gas2cer,pr)
    add_residuals(var_num, product(X,T,R[1:-1]),cer2cer,pr)
    add_residuals(var_num, product(X,T,R[:1]),cer2cerz,pr)    
    add_residuals(var_num, product(X[-1:],T),tgr,pr,val=TBZ)
    add_residuals(var_num, product(X,R),tcr2tcp,pr)
        
    prb = sc.vstack(prb_chain)
    x,dx,dz = lut.slvlprd(prb, var_num, TGZ,False)
    print(x)
    pc = sc.split(x[:-1],max_reg)

    sc.savetxt("poly_coeff_3d",pc)

    

if __name__ == "__main__":
    main()

         
