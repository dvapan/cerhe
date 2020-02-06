import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils

from solver3d import coeffs
regsize = 0

pc = np.loadtxt("poly_coeff_3d")

tgp = Polynom(2, 5)
tcp = Polynom(3, 5)
tgr = Polynom(2, 5)
tcr = Polynom(3, 5)

context = Context()
context.assign(tgp)
context.assign(tcp)
context.assign(tgr)
context.assign(tcr)

s,f = 0,tgp.coeff_size
tgp.coeffs = pc[s:f]
s,f = s+tgp.coeff_size,f+tcp.coeff_size
tcp.coeffs = pc[s:f]
s,f = s+tcp.coeff_size,f+tgr.coeff_size
tgr.coeffs = pc[s:f]
s,f = s+tgr.coeff_size,f+tcr.coeff_size
tcr.coeffs = pc[s:f]

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



def main():
    var_num = 0
    for el in [tgp,tcp,tgr,tcr]:
        var_num += el.coeff_size
    print(var_num,1)
    # print("residual: ", pc[-1])
    var_num += 1
    
    info = " ","t","r","temp","balance left", "residual"
    fmts = "|{:8}{:7.3}{:7.3}{:12.2f}{:16.6}{:16.6}"
    info_fmt = "|{:8}{:>7}{:>7}{:>12}{:>16}{:>16}"
    w = 1+8+7*2+12*1+16*2
    space = " "*((w+1)*len(X))
    s = "{:<{w}}".format("primal",w=((w+1)*len(X)))
    print (s)
    print ("="*len(s))

    
    for t in T:
        for x in X:
            print("{:^{w}.3}".format(x,w=w), end=" ")
        print()
        for x in X:
            print(info_fmt.format(*info),end = " ")
        print()
        
        for x in X:
            tg = tgp([x,t])[0]
            dtgdx = tgp([x,t],[1,0])[0]
            dtgdt = tgp([x,t],[0,1])[0]
            tc = tcp([x,t,R[0]])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = -coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
            row_type = "gas2gasp"
            d = eq_right-eq_left
            print(fmts.format(row_type,t,"",tg,eq_left,d),end = " ")
        print()
        
        for r in R:
            for x in X:
                tg = tgp([x,t])[0]
                tc = tcp([x,t,r])[0]
                dtcdt = tcp([x,t,r],[0,1,0])[0]
                dtcdr = tcp([x,t,r],[0,0,1])[0]
                d2tcdr2 = tcp([x,t,r],[0,0,2])[0]
                if r == R[0]:
                    row_type = "gas2cer"
                    eq_right = coef["lam"] * dtcdr 
                    eq_left = (tg-tc)*coef["alpha"]
                else:
                    eq_right = coef["a"]*(d2tcdr2 + 2/r * dtcdr)
                    eq_left = dtcdt
                    row_type = "cer2cer"
                print(fmts.format(row_type,"",r,tc,eq_left,eq_right-eq_left),end = " ")
            print()
        print(space)

    print ("{:<{w}}".format("reverse",w=(w*len(X))))
    print ("="*len(s))

    for t in T:
        for x in X:
            print("{:^{w}.3}".format(x,w=w), end=" ")
        print()
        for x in X:
            print(info_fmt.format(*info),end = " ")
        print()
        
        for x in X:
            tg = tgr([x,t])[0]
            dtgdx = tgr([x,t],[1,0])[0]
            dtgdt = tgr([x,t],[0,1])[0]
            tc = tcr([x,t,R[0]])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
            row_type = "gas2gasr"
            eq_right /= coeffs[row_type]
            eq_left /= coeffs[row_type]
            print(fmts.format(row_type,t,"",tg,eq_left,eq_right-eq_left),end = " ")
        print()
        
        for r in R:
            for x in X:
                tg = tgr([x,t])[0]
                tc = tcr([x,t,r])[0]
                dtcdt = tcr([x,t,r],[0,1,0])[0]
                dtcdr = tcr([x,t,r],[0,0,1])[0]
                d2tcdr2 = tcr([x,t,r],[0,0,2])[0]
                if r == R[0]:
                    eq_right = coef["lam"] * dtcdr
                    eq_left = (tg-tc)*coef["alpha"]
                    row_type = "gas2cer"
                else:
                    eq_right = coef["a"]*(d2tcdr2 + 2/r * dtcdr)
                    eq_left = dtcdt
                    row_type = "cer2cer"
                eq_right /= coeffs[row_type]
                eq_left /= coeffs[row_type]
                print(fmts.format(row_type,"",r,tc,eq_left,eq_right-eq_left),end = " ")
            print()
        print(space)

    print ("="*len(s))

        
if __name__ == "__main__":
    main()


