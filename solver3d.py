import numpy as np
from itertools import *
from functools import *
from pprint import pprint

import lp_utils as lut
from constants import *
from polynom import Polynom, Context
import sys
import progressbar

t_def = 1000
regsize = 0
tgp = Polynom(2, max_poly_degree, t_def)
tcp = Polynom(3, max_poly_degree, t_def)
tgr = Polynom(2, max_poly_degree, t_def)
tcr = Polynom(3, max_poly_degree, t_def)

context = Context()
context.assign(tgp)
context.assign(tcp)
context.assign(tgr)
context.assign(tcr)

pp = [tgp,tcp]
pr = [tgr,tcr]

var_num = 0
for el in [tgp,tcp,tgr,tcr]:
    print(el.coeff_size)
    var_num+=el.coeff_size
    
print(var_num,1)
print(var_num*max_reg,1)

from gas_properties import TGZ, gas_coefficients
from air_properties import TBZ, air_coefficients

import ceramic_properties as cp

def cer2cer(x,p,defval=None):
    tc = p[1](x)
    dtcdt = p[1](x,[0, 1, 0])
    dtcdr = p[1](x,[0,0,1])
    dtcdr2 = p[1](x,[0,0,2])
    A = cp.a(t_def)
    # print("cer2cer",A)
    return (dtcdt - A*(dtcdr2 + 2/x[2] * dtcdr))

def gas2gasr(x,p):
    tg = p[0](x[:-1])
    dtgdx = p[0](x[:-1], [1, 0])
    dtgdt = p[0](x[:-1], [0, 1])
    tc = p[1](x)
    ALF,PO, CG, WG= air_coefficients(t_def)
    # print("air",ALF, PO, CG, WG)
    lbalance = (tg - tc) * ALF* surf_spec
    rbalance = PO*fgib* CG*  (dtgdx* WG + dtgdt)
    return  lbalance - rbalance

def gas2gasp(x,p):
    tg = p[0](x[:-1])
    dtgdx = p[0](x[:-1], [1, 0])
    dtgdt = p[0](x[:-1], [0, 1])
    tc = p[1](x)
    ALF,PO, CG, WG= gas_coefficients(t_def)
    # print("gas",ALF, PO, CG, WG)
    lbalance = (tg - tc) * ALF* surf_spec
    rbalance = PO*fgib* CG*  (dtgdx* WG + dtgdt)
    return  lbalance + rbalance

def gas2cer(x, p):
    tg = p[0](x[:-1])
    tc = p[1](x)
    dtcdr = p[1](x, [0, 0, 1])
    ALF,_, _, _= gas_coefficients(t_def)
    LAM = cp.lam(t_def)
    # print("gas2cer", ALF, LAM)
    lbalance = (tg - tc) * ALF
    rbalance =  LAM * dtcdr
    return lbalance - rbalance

def air2cer(x, p):
    tg = p[0](x[:-1])
    tc = p[1](x)
    dtcdr = p[1](x, [0, 0, 1])
    ALF,_, _, _= air_coefficients(t_def)
    LAM = cp.lam(t_def)
    # print("air2cer", ALF, LAM)

    lbalance = (tg - tc) * ALF
    rbalance = LAM * dtcdr
    return lbalance - rbalance


def tcp2tcr(x, p):
    x1 = x[0],T[-1],x[1]
    r1 = p[0](x1)
    x2 = x[0],T[0],x[1]
    r2 = p[1](x2)
    return r2 - r1

def tcr2tcp(x, p):
    x1 = x[0],T[-1],x[1]
    r1 = p[0](x1)
    x2 = x[0],T[ 0],x[1]
    r2 = p[1](x2)
    return r2 - r1

def difference(x,p):
    r1 = p[0](x)
    r2 = p[1](x)
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
    "air2cer"  : cer_coeff,
    "cer2cer"  : cer_coeff,
}

def coeffs_default(ind,name=None):
    return coeffs[name]

counted_coeffs = None

def coeffs_counted(ind,name=None):
    return counted_coeffs[ind]

def shifted(cffs,shift):
    psize = len(cffs[1:-1])
    lzeros = sc.zeros(psize * shift)
    rzeros = sc.zeros((max_reg - shift-1) * psize)
    cffs = sc.hstack([cffs[0],lzeros,cffs[1:-1],rzeros,cffs[-1]])
    return cffs

def make_id(i,j):
    return i*xreg + j


def add_residual(ind, var_num, monoms, val=0):
    part_prim = shifted(sc.hstack([val,monoms,[1]]),ind)
    part_revr = shifted(sc.hstack([-val,-monoms,[1]]),ind)
    prb_chain.append(part_prim)
    prb_chain.append(part_revr)

def add_residual_interreg(ind1,ind2, var_num, monoms1,monoms2):
    val = 0
    part_prim1 = shifted(sc.hstack([val,monoms1,[1]]),ind1)
    part_revr1 = shifted(sc.hstack([-val,-monoms1,[1]]),ind1)

    part_prim2 = shifted(sc.hstack([val,monoms2,[1]]),ind2)
    part_revr2 = shifted(sc.hstack([-val,-monoms2,[1]]),ind2)
 
    part_prim = part_prim2 - part_prim1
    part_prim[-1] = 1
    prb_chain.append(part_prim)
    part_revr = part_revr2 - part_revr1
    part_revr[-1] = 1
    prb_chain.append(part_revr)


def add_residuals(ind, var_num, domain, diff_eq, p, val=0, coeffs=coeffs_default):
    # print(diff_eq.__name__)
    for x in domain:
        if diff_eq.__name__ == "polynom":
            r = diff_eq(x)
        else:
            r = diff_eq(x,p)

        r /= coeffs(ind,diff_eq.__name__)
        add_residual(ind, var_num,r[1:],val)


def add_residuals_interreg(ind1,ind2, var_num, domain1, domain2, diff_eq1, diff_eq2, coeffs):
    for x1,x2 in zip(domain1,domain2):
        r1 = diff_eq1(x1)
        r2 = diff_eq2(x2)
        r1 /= coeffs(ind1,diff_eq1.__name__)
        r2 /= coeffs(ind2,diff_eq2.__name__)
        add_residual_interreg(ind1,ind2, var_num,r1[1:],r2[1:])


def count_part(ind,X,T,R, coeffs):
    add_residuals(ind, var_num, product(X,T,R[:1]),gas2gasp,pp)
    add_residuals(ind, var_num, product(X,T,R[:1]),gas2cer,pp)
    add_residuals(ind, var_num, product(X,T,R[1:]),cer2cer,pp)

    add_residuals(ind, var_num, product(X,T,R[:1]),gas2gasr,pr)
    add_residuals(ind, var_num, product(X,T,R[:1]),air2cer,pr)
    add_residuals(ind, var_num, product(X,T,R[1:]),cer2cer,pr)

    
def heating_gas(ind, X, T, R, coeffs):
    add_residuals(ind, var_num, product(X[:1],T),tgp,pp,TGZ, coeffs)

    
def cooling_gas(ind, X, T, R, coeffs):
    add_residuals(ind, var_num, product(X[-1:],T),tgr,pr, TBZ, coeffs)


def make_solution(tgp,tcp,tgr,tcr, coeffs=coeffs_default):
    global prb_chain
    prb_chain = []

    #########################################################################
    print("count bound constraints for gas")
    print("heating")
    for i in range(treg):
        ind = make_id(i,0)
        heating_gas(ind, X_part[0], T_part[i], R, coeffs)

    print("cooling")
    for i in range(treg):
        ind = make_id(i,xreg-1)
        cooling_gas(ind, X_part[xreg-1], T_part[i], R, coeffs)

    ##########################################################################
    print("count bound constraints for ceramic")
    print("heating")
    for j in range(xreg):
        ind1 = make_id(0,j)
        ind2 = make_id(treg-1,j)
        add_residuals_interreg(ind1, ind2, var_num,
                               product(X_part[j],T_part[0][:1],R),
                               product(X_part[j],T_part[-1][-1:],R),
                               tcp,tcr, coeffs)


    print("cooling")
    for j in range(xreg):
        ind1 = make_id(0,j)
        ind2 = make_id(treg-1,j)
        add_residuals_interreg(ind1, ind2, var_num,
                               product(X_part[j],T_part[0][:1],R),
                               product(X_part[j],T_part[-1][-1:],R),
                               tcr,tcp, coeffs)

    ##########################################################################
        
    for i in range(treg):
        for j in range(xreg):
            print ("count reg:",i,j)
            ind = make_id(i,j)
            count_part(ind, X_part[j], T_part[i], R, coeffs)

    print("connect regions")
    for i in range(1,treg):
        for j in range(1,xreg):
            ind1 = make_id(i,j)
            ind2 = make_id(i,j-1)
            add_residuals_interreg(ind1,ind2, var_num,
                                   product(X_part[j-1][-1:],T_part[i]),
                                   product(X_part[j-1][-1:],T_part[i]),
                                   tgp,tgp, coeffs)

            add_residuals_interreg(ind1,ind2, var_num,
                                   product(X_part[j-1][-1:],T_part[i],R),
                                   product(X_part[j-1][-1:],T_part[i],R),
                                   tcp,tcp, coeffs)

            ind1 = make_id(i,j)
            ind2 = make_id(i-1,j)
            add_residuals_interreg(ind1,ind2, var_num,
                                   product(X_part[j],T_part[i-1][-1:]),
                                   product(X_part[j],T_part[i-1][-1:]),
                                   tgp,tgp,coeffs)

            add_residuals_interreg(ind1,ind2, var_num,
                                   product(X_part[j],T_part[i-1][-1:],R),
                                   product(X_part[j],T_part[i-1][-1:],R),
                                   tcp,tcp,coeffs)

    for i in range(treg-1):
        for j in range(xreg-1):
            ind1 = make_id(i,j)
            ind2 = make_id(i,j+1)
            add_residuals_interreg(ind1,ind2, var_num,
                                   product(X_part[j+1][:1],T_part[i]),
                                   product(X_part[j+1][:1],T_part[i]),
                                   tgr,tgr, coeffs)

            add_residuals_interreg(ind1,ind2, var_num,
                                   product(X_part[j+1][:1],T_part[i],R),
                                   product(X_part[j+1][:1],T_part[i],R),
                                   tcr,tcr, coeffs)

            ind1 = make_id(i,j)
            ind2 = make_id(i+1,j)
            add_residuals_interreg(ind1,ind2, var_num,
                                  product(X_part[j],T_part[i+1][:1]),
                                  product(X_part[j],T_part[i+1][:1]),
                                  tgr,tgr, coeffs)

            add_residuals_interreg(ind1,ind2, var_num,
                                  product(X_part[j],T_part[i+1][:1],R),
                                  product(X_part[j],T_part[i+1][:1],R),
                                  tcr,tcr, coeffs)


    
def main():
    global prb_chain
    global counted_coeffs
    make_solution(tgp,tcp,tgr,tcr)
    prb = sc.vstack(prb_chain)
    prb_base = sc.vstack(prb_chain)
    sc.savetxt("prb",prb)
    x,dx,dz = lut.slvlprd(prb, var_num*max_reg+1, TGZ)
    pc = sc.split(x[:-1],max_reg)
    residual = x[-1]
    sc.savetxt("poly_coeff_3d",pc)
    sc.savetxt("tmp", dx.reshape((-1,1)))
    cnt_iter = 0
    x_old = x[-1]
    # while True:
    #     prb_chain = []
    #     cnt_iter += 1
    #     print ("{:#^100}".format("ITERATION {}".format(cnt_iter)))
    #     i = 0
    #     s,f = 0,tgp.coeff_size
    #     tgp.coeffs = pc[i][s:f]
    #     s,f = s+tgp.coeff_size,f+tcp.coeff_size
    #     tcp.coeffs = pc[i][s:f]
    #     s,f = s+tcp.coeff_size,f+tgr.coeff_size
    #     tgr.coeffs = pc[i][s:f]
    #     s,f = s+tgr.coeff_size,f+tcr.coeff_size
    #     tcr.coeffs = pc[i][s:f]

    #     make_solution(tgp,tcp,tgr,tcr)
    #     prb = sc.vstack(prb_chain)
    #     x,dx,dz = lut.slvlprd(prb, var_num*max_reg+1, TGZ)
    #     pc = sc.split(x[:-1],max_reg)
    #     if abs(x_old-x[-1])<0.01:
    #         break
    #     x_old = x[-1]
    #     residual = x[-1]

    
    sc.savetxt("poly_coeff_3d",pc)


if __name__ == "__main__":
    main()

         
