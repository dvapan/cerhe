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
            print("{:^{w}.3}".format(x,w=1+7+7*2+16*1), end=" ")
        print()
        for x in X:
            print("|{:7}{:>7}{:>7}{:>16}".format(" ","t","r","temp"),end = " ")
        print()
        
        for x in X:
            tg = tgp([x,t])[0]
            print("|{:7}{:7.3f}{:7.3}{:16.2f}".format("gas2gas",t,"",tg),end = " ")
        print()
        
        for r in R[::-1]:
            for x in X:
                tc = tcp([x,t,r])[0]

                if r == R[-1]:
                    row_type = "gas2cer"
                elif r == R[0]:
                    row_type = "cer2zer"
                else:
                    row_type = "cer2cer"
                
                print("|{:7}{:7.3}{:7.3f}{:16.2f}".format(row_type,"",r,tc),end = " ")
            print()
        print()

    print ("reverse")

    for t in T:
        for x in X:
            print("{:^{w}.3}".format(x,w=1+7+7*2+16*1), end=" ")
        print()

        for x in X:
            print("|{:7}{:7.3}{:7.3}{:>16}".format(" ","t","r","temp"),end = " ")
        print()
        
        for x in X:
            tg = tgr([x,t])[0]
            print("|{:7}{:7.3f}{:7.3}{:16.2f}".format("gas2gas",t,"",tg),end = " ")
        print()
        
        for r in R[::-1]:
            for x in X:
                tc = tcr([x,t,r])[0]

                if r == R[-1]:
                    row_type = "gas2cer"
                elif r == R[0]:
                    row_type = "cer2zer"
                else:
                    row_type = "cer2cer"

                print("|{:7}{:7.3}{:7.3f}{:16.2f}".format(row_type,"",r,tc),end = " ")
            print()
        print()

    
if __name__ == "__main__":
    main()


