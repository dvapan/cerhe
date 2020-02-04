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


def cer2cer(x,p):
    if x[2] > 0.00001:
        return p[1](x,[0, 1, 0]) - coef["a"]*(p[1](x,[0,0,2]) + 2/x[2] * p[1](x,[0,0,1]))
    else:
        return p[1](x,[0, 0, 1])


def main():
    var_num = 0
    for el in [tgp,tcp,tgr,tcr]:
        var_num += el.coeff_size
    print(var_num,1)
    var_num += 1
    
    print ("primal")

    for x in X:
        print("|{:>16}{:>16}{:>16}{:>16}".format("temp","dtc/dt","dtc/dr","d2tc/dr2"),end = " ")
    print()
    
    for t in T:
        for x in X:
            gas_temp = tgp([x,t])[0]
            print("|{:16.6}{:16.6}{:16.6}{:16.6}".format(gas_temp,np.nan,0.0,0.0),end = " ")
        print()

        
        for r in R[::-1]:
            for x in X:
                tc = tcp([x,t,r])[0]
                dtcdt = tcp([x,t,r],[0,1,0])[0]
                dtcdr = tcp([x,t,r],[0,0,1])[0]
                d2tcdr2 = tcp([x,t,r],[0,0,2])[0]
                print("|{:16.6}{:16.6}{:16.6}{:16.6}".format(tc,dtcdt,dtcdr,d2tcdr2),end = " ")
            print()
        print()

    print ("reverse")

    for t in T:
        for x in X:
            print("{:16.6}".format(tgr([x,t])[0]),end = " ")
        print()
        for r in R[::-1]:
            for x in X:
                print("{:16.6}".format(tcr([x,t,r])[0]),end = " ")
            print()
        print()

                

    
if __name__ == "__main__":
    main()


