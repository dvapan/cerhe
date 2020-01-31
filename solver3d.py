import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray


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
    if x[2] > 0.01:
        return p[1](x,[0, 1, 0]) - coef["a"]*(p[1](x,[0,0,2]) + 2/x[2] * p[1](x,[0,0,1]))
    else:
        return p[1](x,[0, 0, 1])

def gas2gasr(x,p):
    return (p[0](x[:-1]) - p[1](x)) * coef["k1"] - coef["k2"] * (p[0](x[:-1], [1, 0]) * coef["wg"] + p[0](x[:-1], [0, 1]))

    
def gas2gasp(x,p):
    return (p[0](x[:-1]) - p[1](x)) * coef["k1"] + coef["k2"] * (p[0](x[:-1], [1, 0]) * coef["wg"] + p[0](x[:-1], [0, 1]))

def gas2cer(x, p):
    return (p[0](x[:-1]) - p[1](x)) * coef["alpha"] - coef["lam"] * p[1](x, [0, 0, 1])

def tcp2tcr(x, p):
    x1 = x[0],T[ 0],x[1]
    r1 = p[1](x1)
    x2 = x[0],T[-1],x[1]
    r2 = p[1](x2)
    return r2 - r1

def tcr2tcp(x, p):
    x1 = x[0],T[-1],x[1]
    r1 = p[1](x1)
    x2 = x[0],T[ 0],x[1]
    r2 = p[1](x2)
    return r2 - r1


balance_coeff = 20
temp_coeff = 10

def add_residual(s,var_num, monoms, val=0):
    s.CLP_addConstraint(var_num, np.arange(var_num,dtype=np.int32),
                        np.hstack([monoms,[1]]),val,np.inf)
    s.CLP_addConstraint(var_num, np.arange(var_num,dtype=np.int32),
                        np.hstack([monoms,[-1]]),-np.inf,val)

def add_residuals(s, var_num, domain, diff_eq, p=None, val=0):
    print(diff_eq.__name__)
    for x in domain:
        if diff_eq.__name__ == "polynom":
            r = diff_eq(x)[1:]
        else:
            r = diff_eq(x,p)[1:]
        add_residual(s,var_num,r,val)

def main():    
    pp = [tgp,tcp]
    pr = [tgr,tcr]

    var_num = 0
    for el in [tgp,tcp,tgr,tcr]:
        var_num+=el.coeff_size

    
    s = CyClpSimplex()
    print(var_num,1)
    var_num+=1
    
    s.resize(0,var_num)
    print ("primal process")
    add_residuals(s, var_num, product(X,T,R[-1:]),gas2gasp,pp)
    add_residuals(s, var_num, product(X,T,R[-1:]),gas2cer,pp)
    add_residuals(s, var_num, product(X,T,R),cer2cer,pp)
    add_residuals(s, var_num, product(X[:1],T),tgp,pp,TGZ)
    add_residuals(s, var_num, product(X,R),tcp2tcr,pp)

    print ("reverse process")
    add_residuals(s, var_num, product(X,T,R[-1:]),gas2gasr,pr)
    add_residuals(s, var_num, product(X,T,R[-1:]),gas2cer,pr)
    add_residuals(s, var_num, product(X,T,R),cer2cer,pr)
    add_residuals(s, var_num, product(X[-1:],T),tgr,pr,val=TBZ)
    add_residuals(s, var_num, product(X,R),tcr2tcp,pr)
    
    obj = np.zeros(var_num,dtype=np.float64)
    obj[-1] = 1
    s.setObjectiveArray(obj)
    s.dual()
    res = np.array(s.primalVariableSolution)
    print(res)
    np.savetxt("poly3d",res)

        


if __name__ == "__main__":
    main()

         
