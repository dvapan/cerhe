import scipy as sc
from polynom import Polynom
from polynom import Context

import dbalance as db


if __name__ == '__main__':
    xdop = 5
    xreg, treg = 3, 3
    cnt_var = 2
    degree = 3
    X = sc.linspace(0, 1, 50)
    T = sc.linspace(0, 1, 50)
    X_part = sc.split(X, (17, 33))
    T_part = sc.split(T, (17, 33))
    vert_bounds = sc.linspace(0, 1, xreg+1)
    hori_bounds = sc.linspace(0, 1, treg+1)
    xt_part = [(x, t) for x in X_part[0] for t in T_part[0]]
    for i in range(xreg):
        for j in range(treg):
            pass
