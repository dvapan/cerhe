import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils

regsize = 0

pc = np.loadtxt("poly_coeff")

tgp = Polynom(2, 3)
tcp = Polynom(2, 3)
tgr = Polynom(2, 3)
tcr = Polynom(2, 3)

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
    
    print("primal_process")
    for t in T:
        for x in X:
            print("{:10.5}".format(tgp([x,t])[0]),end = " ")
        print()
        for x in X:
            print("{:10.5}".format(tcp([x,t])[0]),end = " ")
        print()
        print()

    print("reverse process")
    for t in T:
        for x in X:
            print("{:10.5}".format(tgr([x,t])[0]),end = " ")
        print()
        for x in X:
            print("{:10.5}".format(tcr([x,t])[0]),end = " ")
        print()
        print()


    
if __name__ == "__main__":
    main()

