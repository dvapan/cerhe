import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils
from solver2d import coeffs

regsize = 0

pc = np.loadtxt("poly_coeff")

tgp = Polynom(2, 5)
tcp = Polynom(2, 5)
tgr = Polynom(2, 5)
tcr = Polynom(2, 5)

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


def main():

    var_num = 0
    for el in [tgp,tcp,tgr,tcr]:
        var_num += el.coeff_size
    print(var_num,1)
    var_num += 1

    info = " ","t","temp","balance left", "balance right", "residual"
    fmts = "|{:8}{:7.3}{:12.2f}{:16.6}{:16.6}{:16.6}"
    info_fmt = "|{:8}{:>7}{:>12}{:>16}{:>16}{:>16}"
    w = 1+8+7*1+12*1+16*3
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
            tc = tcp([x,t])[0]
            dtcdt = tcp([x,t],[0,1])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = -coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
            row_type = "gas2gasp"
            eq_right /= coeffs[row_type]
            eq_left /= coeffs[row_type]
            d = eq_right-eq_left
            print(fmts.format(row_type,t,tg,eq_left,eq_right,d),end = " ")
        print()

        for x in X:
            tg = tgp([x,t])[0]
            tc = tcp([x,t])[0]
            dtcdt = tcp([x,t],[0,1])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = coef["k3"] * (dtcdt)
            row_type = "gas2cer"
            eq_right /= coeffs[row_type]
            eq_left /= coeffs[row_type]
            d = eq_right-eq_left
            print(fmts.format(row_type,t,tc,eq_left,eq_right,d),end = " ")
        print()


    s = "{:<{w}}".format("reverse",w=((w+1)*len(X)))
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
            tg = tgr([x,t])[0]
            dtgdx = tgr([x,t],[1,0])[0]
            dtgdt = tgr([x,t],[0,1])[0]
            tc = tcr([x,t])[0]
            dtcdt = tcr([x,t],[0,1])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
            row_type = "gas2gasp"
            eq_right /= coeffs[row_type]
            eq_left /= coeffs[row_type]
            d = eq_right-eq_left
            print(fmts.format(row_type,t,tg,eq_left,eq_right,d),end = " ")
        print()

        for x in X:
            tg = tgr([x,t])[0]
            tc = tcr([x,t])[0]
            dtcdt = tcr([x,t],[0,1])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = coef["k3"] * (dtcdt)
            row_type = "gas2cer"
            eq_right /= coeffs[row_type]
            eq_left /= coeffs[row_type]
            d = eq_right-eq_left
            print(fmts.format(row_type,t,tc,eq_left,eq_right,d),end = " ")
        print()



    
if __name__ == "__main__":
    main()

