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
    tgc,tcc = cl[:,:psize2],ht[:,psize2:]
    p2 = mvmonos(in_pts, ppwrs2, [0, 0])
    utgh = np.sum(p2*tgh,axis=1)
    utgc = np.sum(p2*tgc,axis=1)
    utch = []
    utcc = []
    for r in R:
        p3 = mvmonos(np.hstack([in_pts,np.full((len(in_pts),1),r)]),
            ppwrs3, [0, 0, 0])
        utch.append(np.sum(p3*tch,axis=1))
        utcc.append(np.sum(p3*tcc,axis=1))
    return utgh,utch,utgc,utcc


parser = argparse.ArgumentParser()
parser.add_argument("filenames", nargs='*')
parser.add_argument("--xreg", default=1,type=int)
parser.add_argument("--treg", default=1,type=int)
parser.add_argument("--pprx", default=7,type=int)
parser.add_argument("--pprt", default=7,type=int)
parser.add_argument("--TBZscl", default=0.9,type=float)
parser.add_argument("--TGZscl", default=1.1,type=float)
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
R = np.linspace(0.01*rball, rball, 3)
R = R[::-1]

fig, axs = plt.subplots(2)
plt.subplots_adjust(left=0.1, bottom=0.25)

tt,xx = np.meshgrid(T,X)
in_pts = np.vstack([tt.flatten(),xx.flatten()]).T
lh = []
lc = []
for i,pc in enumerate(pcs):
    tgh,tch,tgc,tcc = get_pv(pc,in_pts, R ,p)
    l1, = axs[0].plot(T, tgh, lw=2)
    l2, = axs[1].plot(T, tgc, lw=2)
    lh.append([l1])
    lc.append([l2])
    for j in range(len(R)):
        lj1, = axs[0].plot(T, tch[j], lw=2)
        lj2, = axs[1].plot(T, tcc[j], lw=2)
        lh[-1].append(lj1)
        lc[-1].append(lj2)
axs[0].axis([0, total_time, TBZscl*TBZ, TGZscl*TGZ])
axs[1].axis([0, total_time, TBZscl*TBZ, TGZscl*TGZ])


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
        lh[i][0].set_ydata(tgh)
        lc[i][0].set_ydata(tgc)
        for j in range(len(R)):
            lh[i][j+1].set_ydata(tch[j])
            lc[i][j+1].set_ydata(tcc[j])
    fig.canvas.draw_idle()
spos.on_changed(update)

resetax = plt.axes([0.8, 0.025, 0.1, 0.04])
button = Button(resetax, 'Reset', color=axcolor, hovercolor='0.975')

def reset(event):
    spos.reset()
button.on_clicked(reset)

plt.show()
