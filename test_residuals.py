import numpy as np
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils

from solver3d import coeffs

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

X = sc.linspace(0, length, 100)
T = sc.linspace(0, time, 100)
R = sc.linspace(radius_inner, radius, 20)
R = R[::-1]
residual = 0.47669397
max_residual = 0

for t in T:
    print(t, max_residual)
    for x in X:
        tg = tgp([x,t])[0]
        dtgdx = tgp([x,t],[1,0])[0]
        dtgdt = tgp([x,t],[0,1])[0]
        tc = tcp([x,t,R[0]])[0]
        eq_left = (tg - tc)*coef["k1"]
        eq_right = -coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
        row_type = "gas2gasp"
        d = eq_right-eq_left
        d = abs(d)
        max_residual = max(d/coeffs[row_type],abs(max_residual))
    print("gas")
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
            d = eq_right-eq_left
            d = abs(d)
            max_residual = max(d/coeffs[row_type],abs(max_residual))
    print("cer")

for t in T:
    print(t, max_residual)
    for x in X:
        tg = tgr([x,t])[0]
        dtgdx = tgr([x,t],[1,0])[0]
        dtgdt = tgr([x,t],[0,1])[0]
        tc = tcr([x,t,R[0]])[0]
        eq_left = (tg - tc)*coef["k1"]
        eq_right = coef["k2"] * (dtgdx * coef["wg"] + dtgdt)
        row_type = "gas2gasr"
        d = eq_right-eq_left
        d = abs(d)
        max_residual = max(d/coeffs[row_type],abs(max_residual))
    print("gas")
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
            d = eq_right-eq_left
            d = abs(d)
            max_residual = max(d/coeffs[row_type],abs(max_residual))
    print("cer")
    
print("{:7} {:7} {:7}".format("residual", "max_residual", "delta"))
print("{:7.5f} {:7.5f} {:7.5f}".format(residual, max_residual, abs(residual - max_residual)))