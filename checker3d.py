import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils

from solver3d import coeffs
regsize = 0

def make_id(i,j):
    return i*xreg + j


pc = np.loadtxt("poly_coeff_3d")
pc = pc.reshape((max_reg, -1))
print(pc)

tgp = list()
tcp = list()
tgr = list()
tcr = list()

for i in range(max_reg):
    tgp.append(Polynom(2, max_poly_degree))
    tcp.append(Polynom(3, max_poly_degree))
    tgr.append(Polynom(2, max_poly_degree))
    tcr.append(Polynom(3, max_poly_degree))

    context = Context()
    context.assign(tgp[i])
    context.assign(tcp[i])
    context.assign(tgr[i])
    context.assign(tcr[i])

    s,f = 0,tgp[i].coeff_size
    tgp[i].coeffs = pc[i][s:f]
    s,f = s+tgp[i].coeff_size,f+tcp[i].coeff_size
    tcp[i].coeffs = pc[i][s:f]
    s,f = s+tcp[i].coeff_size,f+tgr[i].coeff_size
    tgr[i].coeffs = pc[i][s:f]
    s,f = s+tgr[i].coeff_size,f+tcr[i].coeff_size
    tcr[i].coeffs = pc[i][s:f]


def main():
    ppr = 10                        # Точек на регион

    totalx = xreg*ppr - xreg + 1
    totalt = treg*ppr - treg + 1

    
    dx = length/xreg
    dt = time/treg

    X = sc.linspace(0, length, totalx)
    T = sc.linspace(0, time, totalt)
    R = sc.linspace(0.01*rball, rball, 10)
    R = R[::-1]

    X_part = list(mit.windowed(X,n=ppr,step = ppr-1))
    T_part = list(mit.windowed(T,n=ppr,step = ppr-1))


    var_num = 0
    for el in [tgp[0],tcp[0],tgr[0],tcr[0]]:
        var_num += el.coeff_size
    print(var_num,1)
    # print("residual: ", pc[-1])
    var_num += 1

    f = open("tbl", "w+")
    
    info = " ","t","r","temp","balance left", "residual"
    fmts = "|{:8}{:12.3}{:12.3}{:12.2f}{:16.6}{:16.6}"
    info_fmt = "|{:8}{:>12}{:>12}{:>12}{:>16}{:>16}"
    w = 1+8+12*3+16*2
    space = " "*((w+1)*len(X))
    s = "{:<{w}}".format("primal",w=((w+1)*len(X)))
    f.write (s+"\n")
    f.write ("="*len(s)+"\n")

    max_residual = 0

    for i in range(treg):
        for t in T_part[i]:
            for j in range(xreg):
                for x in X_part[j]:
                    f.write("{:^{w}.3}".format(x,w=w) + " ")
            f.write("\n")
            for j in range(xreg):
                for x in X_part[j]:
                    f.write(info_fmt.format(*info) + " ")
            f.write("\n")
            
            for j in range(xreg):
                for x in X_part[j]:
                    ind = make_id(i,j)
                    tg = tgp[ind]([x,t])[0]
                    dtgdx = tgp[ind]([x,t],[1,0])[0]
                    dtgdt = tgp[ind]([x,t],[0,1])[0]
                    tc = tcp[ind]([x,t,R[0]])[0]
                    eq_left = (tg - tc)*coef["k1"]
                    eq_right = -coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
                    row_type = "gas2gasp"
                    d = eq_right-eq_left
                    max_residual = max(d/coeffs[row_type],max_residual)
                    f.write(fmts.format(row_type,t,"",tg,eq_left,d) + " ")
            f.write("\n")
        
            for r in R:
                for j in range(xreg):
                    for x in X_part[j]:
                        ind = make_id(i,j)
                        tg = tgp[ind]([x,t])[0]
                        tc = tcp[ind]([x,t,r])[0]
                        dtcdt = tcp[ind]([x,t,r],[0,1,0])[0]
                        dtcdr = tcp[ind]([x,t,r],[0,0,1])[0]
                        d2tcdr2 = tcp[ind]([x,t,r],[0,0,2])[0]
                        if r == R[0]:
                            row_type = "gas2cer"
                            eq_right = coef["lam"] * dtcdr 
                            eq_left = (tg-tc)*coef["alpha"]
                        else:
                            eq_right = coef["a"]*(d2tcdr2 + 2/r * dtcdr)
                            eq_left = dtcdt
                            row_type = "cer2cer"
                        d = eq_right-eq_left
                        max_residual = max(d/coeffs[row_type],max_residual)                
                        f.write(fmts.format(row_type,"",r,tc,eq_left,d) + " ")
                f.write("\n")
            f.write(space+"\n")

    f.write ("{:<{w}}".format("reverse",w=(w*len(X))) + "\n")
    f.write ("="*len(s) + "\n")

    for i in range(treg):
        for t in T_part[i]:
            for j in range(xreg):
                for x in X_part[j]:
                    f.write("{:^{w}.3}".format(x,w=w) + " ")
            f.write("\n")
            for j in range(xreg):
                for x in X_part[j]:
                    f.write(info_fmt.format(*info) + " ")
            f.write("\n")
            
            for j in range(xreg):
                for x in X_part[j]:
                    ind = make_id(i,j)
                    tg = tgr[ind]([x,t])[0]
                    dtgdx = tgr[ind]([x,t],[1,0])[0]
                    dtgdt = tgr[ind]([x,t],[0,1])[0]
                    tc = tcr[ind]([x,t,R[0]])[0]
                    eq_left = (tg - tc)*coef["k1"]
                    eq_right = coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
                    row_type = "gas2gasr"
                    d = eq_right-eq_left
                    max_residual = max(d/coeffs[row_type],max_residual)
                    f.write(fmts.format(row_type,t,"",tg,eq_left,d) + " ")
            f.write("\n")
        
            for r in R:
                for j in range(xreg):
                    for x in X_part[j]:
                        ind = make_id(i,j)
                        tg = tgr[ind]([x,t])[0]
                        tc = tcr[ind]([x,t,r])[0]
                        dtcdt = tcr[ind]([x,t,r],[0,1,0])[0]
                        dtcdr = tcr[ind]([x,t,r],[0,0,1])[0]
                        d2tcdr2 = tcr[ind]([x,t,r],[0,0,2])[0]
                        if r == R[0]:
                            row_type = "gas2cer"
                            eq_right = coef["lam"] * dtcdr 
                            eq_left = (tg-tc)*coef["alpha"]
                        else:
                            eq_right = coef["a"]*(d2tcdr2 + 2/r * dtcdr)
                            eq_left = dtcdt
                            row_type = "cer2cer"
                        d = eq_right-eq_left
                        max_residual = max(d/coeffs[row_type],max_residual)
                        f.write(fmts.format(row_type,"",r,tc,eq_left,d) + " ")
                f.write("\n")
            f.write(space+"\n")

    f.write ("="*len(s) + "\n")

    f.close()

    print("maximum residual: ",max_residual)
        
if __name__ == "__main__":
    main()


