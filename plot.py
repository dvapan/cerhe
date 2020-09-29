import numpy as np
import scipy as sc
from itertools import *
from functools import *
from pprint import pprint

from constants import *
from polynom import Polynom, Context
import utils

import sys
import matplotlib.pyplot as plt

def make_id(i,j):
    return i*xreg + j

def peval(poly,x):
    j = int(x[0]/px)
    i = int(x[1]/pt)
    ind = make_id(i,j)
    val = poly[ind](x)[0]
    return val

vpeval = sc.vectorize(peval)

name = sys.argv[1]
xreg,treg = map(int,sys.argv[2:])
max_reg = xreg*treg

px = length/xreg
pt = time/treg

regsize = 0


pc = np.loadtxt(name)
pc = pc.reshape((max_reg, -1))

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


X = sc.arange(0, length, 0.01)
T = sc.arange(0, time, 0.1)
R = sc.linspace(0.01*rball, rball, 10)

R = R[::-1]
# Time moment
x = X.reshape((-1,1))
t = np.full_like(X,T[-1]).reshape((-1,1))
r = np.full_like(X,R[0]).reshape((-1,1))
r1 = np.full_like(X,R[-1]).reshape((-1,1))

xx = np.hstack([x,t])

valgp = np.array([peval(tgp,x) for x in xx])
plt1= plt.subplot(321)
plt1.plot(X,valgp)

valgr = np.array([peval(tgr,x) for x in xx])
plt2 = plt.subplot(322)
plt2.plot(X,valgr)


xx = np.hstack([x,t,r])
valcp = [peval(tcr,x) for x in xx]
plt1.plot(X,valcp)
valcr = [peval(tcp,x) for x in xx]
plt2.plot(X,valcr)


xx = np.hstack([x,t,r1])
valcp = [peval(tcr,x) for x in xx]
plt1.plot(X,valcp)
valcr = [peval(tcp,x) for x in xx]
plt2.plot(X,valcr)


plt3 = plt.subplot(312)
# plt.plot(t,val)

t = T.reshape((-1,1))
x = np.full_like(T,X[-1]).reshape((-1,1))
r = np.full_like(T,R[0]).reshape((-1,1))
r1 = np.full_like(T,R[-1]).reshape((-1,1))
xx = np.hstack([x,t])

tvalgp = np.array([peval(tgp,x) for x in xx])

tvalgr = [peval(tgr,x) for x in xx]

tt = np.vstack([t,t[-1]+t])
vv = np.hstack([tvalgp,tvalgr])
plt3.plot(tt,vv)

xx = np.hstack([x,t,r])

tvalcp = [peval(tcp,x) for x in xx]
tvalcr = [peval(tcr,x) for x in xx]

tt = np.vstack([t,t[-1]+t])
vv = np.hstack([tvalcp,tvalcr])

plt3.plot(tt,vv)

xx = np.hstack([x,t,r1])

tvalcp = [peval(tcp,x) for x in xx]
tvalcr = [peval(tcr,x) for x in xx]

tt = np.vstack([t,t[-1]+t])
vv = np.hstack([tvalcp,tvalcr])

plt3.plot(tt,vv)



R = R[::-1]
# Time moment
x = X.reshape((-1,1))
t = np.full_like(X,T[len(T)//2]).reshape((-1,1))
r = np.full_like(X,R[0]).reshape((-1,1))
r1 = np.full_like(X,R[-1]).reshape((-1,1))

xx = np.hstack([x,t])

valgp = np.array([peval(tgp,x) for x in xx])
plt4= plt.subplot(325)
plt4.plot(X,valgp)

valgr = np.array([peval(tgr,x) for x in xx])
plt5 = plt.subplot(326)
plt5.plot(X,valgr)


xx = np.hstack([x,t,r])
valcp = [peval(tcr,x) for x in xx]
plt4.plot(X,valcp)
valcr = [peval(tcp,x) for x in xx]
plt5.plot(X,valcr)


xx = np.hstack([x,t,r1])
valcp = [peval(tcr,x) for x in xx]
plt4.plot(X,valcp)
valcr = [peval(tcp,x) for x in xx]
plt5.plot(X,valcr)



plt.show()
