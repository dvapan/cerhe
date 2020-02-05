import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils

regsize = 0

pc = np.loadtxt("poly_coeff_3d")

tgp = Polynom(2, 3)
tcp = Polynom(3, 3)
tgr = Polynom(2, 3)
tcr = Polynom(3, 3)

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
    
    print ("primal")

    
    for t in T:
        for x in X:
            print("{:^{w}.3}".format(x,w=1+7+7*2+16*6), end=" ")
        print()
        for x in X:
            print("|{:7}{:>7}{:>7}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}".format(" ","t","r","temp","d/dt","d/dr","d2/dr2","balance left", "balance right"),end = " ")
        print()
        
        for x in X:
            tg = tgp([x,t])[0]
            dtgdx = tgp([x,t],[1,0])[0]
            dtgdt = tgp([x,t],[0,1])[0]
            tc = tcp([x,t,R[-1]])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
            print("|{:7}{:7.3f}{:7.3}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}".format("gas2gas",t,"",tg,dtgdt,np.nan,np.nan,eq_left,eq_right),end = " ")
        print()
        
        for r in R[::-1]:
            for x in X:
                tg = tgp([x,t])[0]
                tc = tcp([x,t,r])[0]
                dtcdt = tcp([x,t,r],[0,1,0])[0]
                dtcdr = tcp([x,t,r],[0,0,1])[0]
                d2tcdr2 = tcp([x,t,r],[0,0,2])[0]
                showt = ""
                if r == R[-1]:
                    eq_right = coef["lam"] * dtcdr
                    eq_left = (tg-tc)*coef["alpha"]
                    row_type = "gas2cer"
                elif r == R[0]:
                    eq_right = dtcdr
                    eq_left = 0.0
                    row_type = "cer2zer"
                else:
                    eq_right = coef["a"]*(d2tcdr2 + 2/r * dtcdr)
                    eq_left = dtcdt
                    row_type = "cer2cer"
                print("|{:7}{:7.3}{:7.3f}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}".format(row_type,showt,r,tc,dtcdt,dtcdr,d2tcdr2,eq_left,eq_right),end = " ")
            print()
        print()

    print ("reverse")

    for t in T:

        for x in X:
            print("|{:7}{:>16}{:>16}{:>16}{:>16}{:>16}{:>16}".format(" ","temp","d/dt","d/dr","d2/dr2","balance left", "balance right"),end = " ")
        print()
        
        for x in X:
            tg = tgr([x,t])[0]
            dtgdx = tgr([x,t],[1,0])[0]
            dtgdt = tgr([x,t],[0,1])[0]
            tc = tcr([x,t,R[-1]])[0]
            eq_left = (tg - tc)*coef["k1"]
            eq_right = coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
            print("|{:7}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}".format("gas2gas",tg,dtgdt,np.nan,np.nan,eq_left,eq_right),end = " ")
        print()
        
        for r in R[::-1]:
            for x in X:
                tg = tgr([x,t])[0]
                tc = tcr([x,t,r])[0]
                dtcdt = tcr([x,t,r],[0,1,0])[0]
                dtcdr = tcr([x,t,r],[0,0,1])[0]
                d2tcdr2 = tcr([x,t,r],[0,0,2])[0]
                if r == R[-1]:
                    eq_right = coef["lam"] * dtcdr
                    eq_left = (tg-tc)*coef["alpha"]
                    row_type = "gas2cer"
                elif r == R[0]:
                    eq_right = dtcdr
                    eq_left = 0.0
                    row_type = "cer2zer"
                else:
                    eq_right = coef["a"]*(d2tcdr2 + 2/r * dtcdr)
                    eq_left = dtcdt
                    row_type = "cer2cer"
                print("|{:7}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}{:16.6}".format(row_type,tc,dtcdt,dtcdr,d2tcdr2,eq_left,eq_right),end = " ")
            print()
        print()

    # for t in T:
    #     for x in X:
    #         print("{:16.6}".format(tgr([x,t])[0]),end = " ")
    #     print()
    #     for r in R[::-1]:
    #         for x in X:
    #             print("{:16.6}".format(tcr([x,t,r])[0]),end = " ")
    #         print()
    #     print()

                

    
if __name__ == "__main__":
    main()


