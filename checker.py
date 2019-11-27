import scipy as sc
from itertools import *
from functools import *
from pprint import pprint
import utils as ut
import lp_utils as lut
from constants import *

def make_id(x):
    return x[0]*treg + x[1]

pc = sc.loadtxt("poly_coeff")


vals = list()

for i in range(xreg):
    for j in range(treg):
        gc,cc,gcr,ccr = sc.split(pc[make_id((i,j))], 4)
        tg, tc, tgr, tcr = ut.make_gas_cer_quad(2, 3, gc,cc,gcr,ccr)
        xv, tv = sc.meshgrid(X_part[j], T_part[i])
        xv = xv.reshape(-1)
        tv = tv.reshape(-1)
        xt = sc.vstack([xv, tv]).T
        g = lambda x:tg(x,[0,0])[0]
        cffs = sc.hstack(list(map(g, xt)))
        g = lambda x:tc(x,[0,0])[0]
        cffs2 = sc.hstack(list(map(g, xt)))
        vals.append(sc.hstack([xt, cffs.reshape(-1,1), cffs2.reshape(-1,1)]))

sc.savetxt("out",sc.vstack(vals),fmt="%10.5f")

