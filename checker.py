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
        if max_reg != 1:
            gc,cc,gcr,ccr = sc.split(pc[make_id((i,j))], 4)
        else:
            gc,cc,gcr,ccr = sc.split(pc, 4)
        tg, tc, tgr, tcr = ut.make_gas_cer_quad(2, 3, gc,cc,gcr,ccr)


        balanceq = dict({
            'be1l':lambda x: (tg(x) - tc(x)) * coef["k1"],
            'be1r':lambda x: coef["k2"] * (tg(x, [1, 0]) * coef["wg"] + tg(x, [0, 1])),
            'be3l':lambda x: (tg(x) - tc(x)) * coef["k1"],
            'be3r':lambda x: coef["k3"] * tc(x, [0, 1]),
        })

        balanceqr = dict({
            'be2l':lambda x: (tgr(x) - tcr(x)) * coef["k1"],
            'be2r':lambda x: coef["k2"] * (tgr(x, [1, 0]) * coef["wg"] + tgr(x, [0, 1])),
            'be3l':lambda x: (tgr(x) - tcr(x)) * coef["k1"],
            'be3r':lambda x: coef["k3"] * tcr(x, [0, 1]),
        })
        
        xv, tv = sc.meshgrid(X_part[j], T_part[i])
        xv = xv.reshape(-1)
        tv = tv.reshape(-1)
        xt = sc.vstack([xv, tv]).T
        g = lambda x:balanceq['be1l'](x)[0]
        cffs = sc.hstack(list(map(g, xt)))
        g = lambda x:balanceq['be1r'](x)[0]
        cffs1 = sc.hstack(list(map(g, xt)))
        g = lambda x:balanceq['be3l'](x)[0]
        cffs2 = sc.hstack(list(map(g, xt)))
        g = lambda x:balanceq['be3r'](x)[0]
        cffs3 = sc.hstack(list(map(g, xt)))
        g = lambda x:tg(x,[0,0])[0]
        cffs2t = sc.hstack(list(map(g, xt)))
        g = lambda x:tc(x,[0,0])[0]
        cffs3t = sc.hstack(list(map(g, xt)))
        g = lambda x:tgr(x,[0,0])[0]
        cffs2tr = sc.hstack(list(map(g, xt)))
        g = lambda x:tcr(x,[0,0])[0]
        cffs3tr = sc.hstack(list(map(g, xt)))

        # vals.append(sc.hstack([xt, cffs.reshape(-1,1), cffs3.reshape(-1,1), cffs2.reshape(-1,1), cffs4.reshape(-1,1)]))
        vals.append(sc.hstack([xt,
                               cffs.reshape(-1,1),
                               cffs1.reshape(-1,1),
                               cffs2.reshape(-1,1),
                               cffs3.reshape(-1,1),                               
                               cffs2t.reshape(-1,1),
                               cffs2tr.reshape(-1,1),
                               cffs3t.reshape(-1,1),
                               cffs3tr.reshape(-1,1),
        ]))

sc.savetxt("out",sc.vstack(vals),fmt="%20.5f")

