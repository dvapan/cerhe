import argparse
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Slider, Button, RadioButtons


import sys
from poly import mvmonos, powers
from constants import *
import matplotlib.pyplot as plt
from model import make_id, cff_cnt, ppwrs2, ppwrs3, psize2, psize3
from gas_properties import TGZ
from air_properties import TBZ

def get_pv(pc, in_pts, R,params):
    ids = in_pts // np.array([ltreg,lxreg])
    pids = np.apply_along_axis(lambda x: int(make_id(*x,params)),1,ids)
    cf = pc[pids]
    ht,cl = np.hsplit(cf,2)
    tgh,tch = ht[:,:psize2],ht[:,psize2:]
    tgc,tcc = cl[:,:psize2],cl[:,psize2:]
    p2 = mvmonos(in_pts, ppwrs2, [0, 0])
    utgh = np.sum(p2*tgh,axis=1)
    utgc = np.sum(p2*tgc,axis=1)
    utch = []
    utcc = []
    for r in R:
        cpts = np.hstack([in_pts,np.full((len(in_pts),1),r)])
        p3 = mvmonos(cpts, ppwrs3, [0, 0, 0])
        utch.append(np.sum(p3*tch,axis=1))
        utcc.append(np.sum(p3*tcc,axis=1))
    return utgh,utch,utgc,utcc


parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='*')
parser.add_argument("--xreg", default=1,type=int)
parser.add_argument("--treg", default=1,type=int)
parser.add_argument("--pprx", default=7,type=int)
parser.add_argument("--pprt", default=7,type=int)
parser.add_argument("--TBZscl", default=0.99,type=float)
parser.add_argument("--TGZscl", default=1.01,type=float)
args = parser.parse_args(sys.argv[1:])
p = vars(args)
xreg = args.xreg
treg = args.treg
pprx = args.pprx
pprt = args.pprt
TBZscl = args.TBZscl
TGZscl = args.TGZscl


totalx = xreg*pprx - xreg + 1
totalt = treg*pprt - treg + 1
X = np.linspace(0, length, totalx)
T = np.linspace(0, total_time, totalt)
X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
lxreg = X_part[0][-1] - X_part[0][0]
ltreg = T_part[0][-1] - T_part[0][0]

pcs = []
for filename in args.filenames:
    pc = np.loadtxt(filename)
    print("Polynom approximate with: {}".format(pc[-1]))
    pc = pc[:-1]
    pc = pc.reshape(-1,sum(cff_cnt))
    pcs.append(pc)


X = np.array([0])
T = np.arange(0, total_time, 0.1)
R = np.linspace(0.015*rball, rball, 6)
R = R[::-1]

fig, axs = plt.subplots(1)
plt.subplots_adjust(left=0.1, bottom=0.25)

tt,xx = np.meshgrid(T,X)
in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
lh = []
lc = []
for i,pc in enumerate(pcs):
    tgh,tch,tgc,tcc = get_pv(pc,in_pts, R ,p)
    TT = np.hstack([T,T+T[-1],T+T[-1]*2,T+T[-1]*3])
    vv = np.hstack([tgh,tgc,tgh,tgc])
    l1, = axs.plot(TT, vv, lw=2)
    lh.append([l1])
    for j in range(len(R)):
        vv2 = np.hstack([tch[j],tcc[j],tch[j],tcc[j]])
        lj1, = axs.plot(TT, vv2, lw=2)
        lh[-1].append(lj1)
axs.axis([0, total_time*4, TBZscl*TBZ, TGZscl*TGZ])


axcolor = 'lightgoldenrodyellow'
axtime = plt.axes([0.25, 0.1, 0.65, 0.03], facecolor=axcolor)

spos = Slider(axtime, 'position', 0, length-0.0001, valinit=0)

def update(val):
    x = spos.val
    X = np.array([x])

    tt,xx = np.meshgrid(T,X)
    in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
    for i,pc in enumerate(pcs):
        tgh,tch,tgc,tcc = get_pv(pc,in_pts, R ,p)
        vv = np.hstack([tgh,tgc,tgh,tgc])
        lh[i][0].set_ydata(vv)
        for j in range(len(R)):
            vv2 = np.hstack([tch[j],tcc[j],tch[j],tcc[j]])
            lh[i][j+1].set_ydata(vv2)
    fig.canvas.draw_idle()
spos.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    spos.reset()
button.on_clicked(reset)

setaxs = plt.axes([0.7, 0.025, 0.1, 0.04])
button1 = Button(setaxs, 'Scale', color=axcolor, hovercolor='0.975')

def axsset(event):
    x = spos.val
    X = np.array([x])

    tt,xx = np.meshgrid(T,X)
    in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
    minv = []
    maxv = []
    for i,pc in enumerate(pcs):
        tgh,tch,tgc,tcc = get_pv(pc,in_pts, R ,p)
        maxv.append(np.max(np.hstack([tgh.flatten(), np.array(tch).flatten(),
                                tgc.flatten(), np.array(tcc).flatten()])))
        minv.append(np.min(np.hstack([tgh.flatten(), np.array(tch).flatten(),
                                tgc.flatten(), np.array(tcc).flatten()])))
    mina = np.min(np.array(minv))
    maxa = np.max(np.array(maxv))
    axs.axis([0, total_time*4, TBZscl*mina, TGZscl*maxa])
    fig.canvas.draw_idle()
button1.on_clicked(axsset)

setaxs_def = plt.axes([0.6, 0.025, 0.1, 0.04])
button2 = Button(setaxs_def, 'Def Scale', color=axcolor, hovercolor='0.975')

def axsset_def(event):
    axs.axis([0, total_time*4, TBZscl*TBZ, TGZscl*TGZ])
button2.on_clicked(axsset_def)
plt.show()
