import numpy as np
import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers

from constants import *
from gas_properties import TGZ, gas_coefficients
from air_properties import TBZ, air_coefficients
import ceramic_properties as cp

def mvmonoss(x,powers,shift_ind,cff_cnt,diff=None):
    lzeros = sum((cff_cnt[i] for i in range(shift_ind)))
    rzeros = sum((cff_cnt[i] for i in range(shift_ind+1,len(cff_cnt))))
    monoms = mvmonos(x,powers,diff)
    lzeros = np.zeros((len(x),lzeros))
    rzeros = np.zeros((len(x),rzeros))
    return np.hstack([lzeros,monoms,rzeros])

    

t_def = 1000
cff_cnt = [10,20,10,20]

#Inner points for ceramic
tt,xx,rr = np.meshgrid(T,X,R)
in_pts_cr = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

#Ceramic to ceramic heat transfer
tcp = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt)
dtcpdt = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt,[1,0,0])
dtcpdr = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt,[0,0,1]) 
dtcpdr2 = mvmonoss(in_pts_cr,powers(3,3),1,cff_cnt,[0,0,2])
a = cp.a(t_def)
monos_cerp = dtcpdt - a*(dtcpdr2 + 2/rr.flatten()[:,np.newaxis] * dtcpdr)

tcr = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt)
dtcrdt = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt,[1,0,0])
dtcrdr = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt,[0,0,1]) 
dtcrdr2 = mvmonoss(in_pts_cr,powers(3,3),3,cff_cnt,[0,0,2])
a = cp.a(t_def)
monos_cerr = dtcrdt - a*(dtcrdr2 + 2/rr.flatten()[:,np.newaxis] * dtcrdr)

#Inner points for gas
tt,xx,rr = np.meshgrid(T,X,R[-1])
in_pts_gs = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

#Gas to gas transfer
tgp = mvmonoss(in_pts_gs[:,:-1],powers(3,2),0,cff_cnt)
dtgpdt = mvmonoss(in_pts_gs[:,:-1],powers(3,2),0,cff_cnt,[1,0])
dtgpdx = mvmonoss(in_pts_gs[:,:-1],powers(3,2),0,cff_cnt,[0,1])
tcp = mvmonoss(in_pts_gs,powers(3,3),1,cff_cnt)
ALF,PO, CG, WG= air_coefficients(t_def)
lb= (tgp - tcp) * ALF* surf_spec
rb= PO*fgib* CG*  (dtgpdx* WG + dtgpdt)
monos_gasp = lb-rb

tgr = mvmonoss(in_pts_gs[:,:-1],powers(3,2),2,cff_cnt)
dtgrdt = mvmonoss(in_pts_gs[:,:-1],powers(3,2),2,cff_cnt,[1,0])
dtgrdx = mvmonoss(in_pts_gs[:,:-1],powers(3,2),2,cff_cnt,[0,1])
tcr = mvmonoss(in_pts_gs,powers(3,3),3,cff_cnt)
ALF,PO, CG, WG= gas_coefficients(t_def)
lb= (tgr - tcr) * ALF* surf_spec
rb= PO*fgib* CG*  (dtgrdx* WG + dtgrdt)
monos_gasr = lb+rb

# Gas to ceramic transfer
tt,xx,rr = np.meshgrid(T,X,R[-1])
in_pts = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

tcp = mvmonoss(in_pts,powers(3,3),1,cff_cnt)
tgp = mvmonoss(in_pts[:,:-1],powers(3,2),0,cff_cnt)
dtcpdr = mvmonoss(in_pts,powers(3,3),1,cff_cnt,[0,0,1]) 
ALF,_, _, _= gas_coefficients(t_def)
LAM = cp.lam(t_def)
lbalance = (tgp - tcp) * ALF
rbalance =  LAM * dtcpdr
monos_g2cp = lbalance - rbalance

tcr = mvmonoss(in_pts,powers(3,3),3,cff_cnt)
tgr = mvmonoss(in_pts[:,:-1],powers(3,2),2,cff_cnt)
dtcrdr = mvmonoss(in_pts,powers(3,3),3,cff_cnt,[0,0,1]) 
ALF,_, _, _= gas_coefficients(t_def)
LAM = cp.lam(t_def)
lbalance = (tgr - tcr) * ALF
rbalance =  LAM * dtcrdr
monos_g2cr = lbalance - rbalance

#Boundary points for start gas supply from left side of Heat Exchanger
tt,xx,rr = np.meshgrid(T,X[0],R[-1])
sb_pts_x0 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
sbtgp = mvmonoss(sb_pts_x0[:,:-1],powers(3,2),0,cff_cnt)

#Boundary points for start air supply from right side of Heat Exchanger
tt,xx,rr = np.meshgrid(T,X[-1],R[-1])
sb_pts_x1 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
sbtgr = mvmonoss(sb_pts_x1[:,:-1],powers(3,2),2,cff_cnt)

#Boundary points for ceramic rever
tt,xx,rr = np.meshgrid(T[0],X,R)
sb_pts_t0 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
tt,xx,rr = np.meshgrid(T[-1],X,R)
sb_pts_t1 = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T

tcp = mvmonoss(sb_pts_t0,powers(3,3),1,cff_cnt)
tcr = mvmonoss(sb_pts_t1,powers(3,3),3,cff_cnt)
revtc1 = tcp - tcr
tcp = mvmonoss(sb_pts_t1,powers(3,3),1,cff_cnt)
tcr = mvmonoss(sb_pts_t0,powers(3,3),3,cff_cnt)
revtc2 = tcp - tcr

prb_chain = [monos_gasp, monos_gasr, monos_cerp, monos_cerr,
             monos_g2cp, monos_g2cr,
                sbtgp, sbtgr, revtc1,revtc2]
rhs_vals = [0,0,0,0,0,0,TGZ,TBZ,0,0]
A = sc.vstack(prb_chain)

rhs = np.hstack(
        list(map(lambda x,y: np.full(len(x),y),prb_chain,rhs_vals))
)


xdop = 1
s = CyClpSimplex()
lp_dim = A.shape[1]+1
A1 = np.hstack([ A,np.ones((len(A),1))])
A2 = np.hstack([-A,np.ones((len(A),1))])
x = s.addVariable('x',lp_dim)
A1 = np.matrix(A1)
A2 = np.matrix(A2)

b1 = CyLPArray(rhs)
b2 = CyLPArray(-rhs)
s += A1*x >= b1
s += A2*x >= b2

s += x[lp_dim-1] >= 0
s += x[lp_dim-1] <= xdop
s.objective = x[lp_dim-1]
print ("START")
s.primal()
outx = s.primalVariableSolution['x']

np.savetxt("poly_coeff_3d",outx)
