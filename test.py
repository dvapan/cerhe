import numpy as np
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
import matplotlib.pyplot as plt

from poly import mvmonos, powers


from constants import *
from gas_properties import TGZ, gas_coefficients
from air_properties import TBZ, air_coefficients
import ceramic_properties as cp

def mvmonoss(x,powers,shift_ind,cff_cnt,diff=None):
    lzeros = sum((cff_cnt[i] for i in range(shift_ind)))
    rzeros = sum((cff_cnt[i] for i in range(shift_ind+1,len(cff_cnt))))
    monos = mvmonos(x,powers,diff)
    lzeros = np.zeros((len(x),lzeros))
    rzeros = np.zeros((len(x),rzeros))
    return np.hstack([lzeros,monos,rzeros])
    

t_def = 1000
cff_cnt = [10,20,10,20]


def ceramic(T,X,R):
    #Inner points for ceramic
    tt,xx,rr = np.meshgrid(T,X,R)
    in_pts_cr = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

    #Ceramic to ceramic heat transfer
    tch = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt)
    dtchdt = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt,[1,0,0])
    dtchdr = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt,[0,0,1]) 
    dtchdr2 = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt,[0,0,2])
    a = cp.a(t_def)
    monos_cerp = dtchdt - a*(dtchdr2 + 2/rr.flatten()[:,np.newaxis] * dtchdr)

    tcc = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt)
    dtccdt = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt,[1,0,0])
    dtccdr = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt,[0,0,1]) 
    dtccdr2 = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt,[0,0,2])
    a = cp.a(t_def)
    monos_cerr = dtccdt - a*(dtccdr2 + 2/rr.flatten()[:,np.newaxis] * dtccdr)

    monos = np.vstack([monos_cerp, monos_cerr])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 0.001)
    return monos,rhs,cff

def gas_air(T,X,R):
    #Inner points for gas and air
    tt,xx,rr = np.meshgrid(T,X,R[0])
    in_pts_gs = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

    #Gas to gas transfer
    tgh = mvmonoss(in_pts_gs[:,:-1],powers(3,2),0,cff_cnt)
    dtghdt = mvmonoss(in_pts_gs[:,:-1],powers(3,2),0,cff_cnt,[1,0])
    dtghdx = mvmonoss(in_pts_gs[:,:-1],powers(3,2),0,cff_cnt,[0,1])
    tch = mvmonoss(in_pts_gs,powers(3,3),1,cff_cnt)
    ALF,PO, CG, WG= gas_coefficients(t_def)
    lb= (tgh - tch) * ALF* surf_spec
    rb= PO*fgib* CG*  (dtghdx* WG + dtghdt)
    monos_gash = lb+rb

    tgc = mvmonoss(in_pts_gs[:,:-1],powers(3,2),2,cff_cnt)
    dtgcdt = mvmonoss(in_pts_gs[:,:-1],powers(3,2),2,cff_cnt,[1,0])
    dtgcdx = mvmonoss(in_pts_gs[:,:-1],powers(3,2),2,cff_cnt,[0,1])
    tcc = mvmonoss(in_pts_gs,powers(3,3),3,cff_cnt)
    ALF,PO, CG, WG= air_coefficients(t_def)
    lb= (tgc - tcc) * ALF* surf_spec
    rb= PO*fgib* CG*  (dtgcdx* WG + dtgcdt)
    monos_gasc = lb-rb

    monos = np.vstack([monos_gash,monos_gasc])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 10)
    return monos,rhs,cff

def ceramic_surface(T,X,R):
    # Ceramic surface
    tt,xx,rr = np.meshgrid(T,X,R[0])
    in_pts = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

    tch = mvmonoss(in_pts,powers(3,3),1,cff_cnt)
    tgh = mvmonoss(in_pts[:,:-1],powers(3,2),0,cff_cnt)
    dtchdr = mvmonoss(in_pts,powers(3,3),1,cff_cnt,[0,0,1]) 
    ALF,_, _, _= gas_coefficients(t_def)
    LAM = cp.lam(t_def)
    lbalance = (tgh - tch) * ALF
    rbalance =  LAM * dtchdr
    monos_g2cp = lbalance - rbalance

    tcc = mvmonoss(in_pts,powers(3,3),3,cff_cnt)
    tgc = mvmonoss(in_pts[:,:-1],powers(3,2),2,cff_cnt)
    dtccdr = mvmonoss(in_pts,powers(3,3),3,cff_cnt,[0,0,1]) 
    ALF,_, _, _= air_coefficients(t_def)
    LAM = cp.lam(t_def)
    lbalance = (tgc - tcc) * ALF
    rbalance =  LAM * dtccdr
    monos_g2cr = lbalance - rbalance

    monos =  np.vstack([monos_g2cp, monos_g2cr])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 0.001)
    return monos,rhs,cff

#Boundary points for start gas supply from left side of Heat Exchanger
def boundary(T,X,R,val,ind):
    tt,xx,rr = np.meshgrid(T,X,R)
    sb_pts_x0 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
    monos = mvmonoss(sb_pts_x0[:,:-1],powers(3,2),ind,cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff

#Boundary points for ceramic rever
def boundary_revert(T_start,T_end,X,R):
    tt,xx,rr = np.meshgrid(T_start,X,R)
    sb_pts_t0 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
    tt,xx,rr = np.meshgrid(T_end,X,R)
    sb_pts_t1 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

    tch = mvmonoss(sb_pts_t0,powers(3,3),1,cff_cnt)
    tcc = mvmonoss(sb_pts_t1,powers(3,3),3,cff_cnt)
    revtc1 = tch - tcc
    tch = mvmonoss(sb_pts_t1,powers(3,3),1,cff_cnt)
    tcc = mvmonoss(sb_pts_t0,powers(3,3),3,cff_cnt)
    revtc2 = tch - tcc

    monos = np.vstack([revtc1,revtc2])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 1)
    return monos,rhs,cff

conditions = (gas_air(T,X,R),
              ceramic_surface(T,X,R),
              ceramic(T,X,R),
              boundary(T,X[0],R[0], TGZ, 0),
              boundary(T,X[-1],R[0], TBZ, 2),
              boundary_revert(T[0],T[-1],X,R),
              )
monos = []
rhs = []
cff = []
for m,r,c in conditions:
    monos.append(m)
    rhs.append(r)
    cff.append(c)

A = sc.vstack(monos)

rhs = np.hstack(rhs)
cff = np.hstack(cff).reshape(-1,1)

print (rhs)


xdop = 10
s = CyClpSimplex()
lp_dim = A.shape[1]+1

A1 = np.hstack([ A,cff])
A2 = np.hstack([-A,cff])

x = s.addVariable('x',lp_dim)
A1 = np.matrix(A1)
A2 = np.matrix(A2)

b1 = CyLPArray(rhs)
b2 = CyLPArray(-rhs)

np.savetxt("A",np.vstack([A1,A2]))
np.savetxt("b",np.hstack([b1,b2]))


s += A1*x >= b1
s += A2*x >= b2

s += x[lp_dim-1] >= 0
s += x[lp_dim-1] <= xdop
s.objective = x[lp_dim-1]
print ("START")
s.primal()
outx = s.primalVariableSolution['x']

np.savetxt("test_cff",outx[:-1])


