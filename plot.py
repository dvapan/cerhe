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

pc = np.loadtxt("poly_coeff")

cff_cnt = [10,20,10,20]

s,f = 0,cff_cnt[0]
tgh_cf = pc[s:f]
s,f = s+cff_cnt[0],f+cff_cnt[1]
tch_cf = pc[s:f]
s,f = s+cff_cnt[1],f+cff_cnt[2]
tgc_cf = pc[s:f]
s,f = s+cff_cnt[2],f+cff_cnt[3]
tcc_cf = pc[s:f]


X = sc.linspace(0, length, totalx*3)
T = sc.linspace(0, time, totalt*3)
R = sc.linspace(0.01*rball, rball, 10*3)
R = R[::-1]

#gas
tt,xx = np.meshgrid(T,X)
in_pts_cr = np.vstack([tt.flatten(),xx.flatten()]).T
pp = mvmonos(in_pts_cr,powers(3,2))

tt,xx = np.meshgrid(T,X)
u = pp.dot(tgh_cf)
uu = u.reshape((len(T), len(X)))

print(uu[0,:])
plt.plot(tt[0,:],uu[-1,:])
# ceramic

tt,xx,rr = np.meshgrid(T,X,R[0])
in_pts_cr = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
pp = mvmonos(in_pts_cr,powers(3,3))
u = pp.dot(tch_cf)
uu = u.reshape((len(T), len(X)))

plt.plot(tt[0,:],uu[-1,:])


tt,xx,rr = np.meshgrid(T,X,R[-1])
in_pts_cr = np.vstack([tt.flatten(),xx.flatten(),rr.flatten()]).T
pp = mvmonos(in_pts_cr,powers(3,3))
u = pp.dot(tch_cf)
uu = u.reshape((len(T), len(X)))

plt.plot(tt[0,:],uu[-1,:])


# fig, ax = plt.subplots()
# p = ax.contourf(tt, xx, uu, np.linspace(700, 1900, 100), cmap='inferno')

# fig.colorbar(p, ax=ax)
# fig.tight_layout()
plt.xlim(0, 300)
plt.ylim(760, 800)
plt.show()
