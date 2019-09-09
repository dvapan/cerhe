import scipy as sc
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from utils import *


if __name__ == '__main__':
    xdop = 5
    xreg, treg = 3, 3
    cnt_var = 2
    degree = 3
    X = sc.linspace(0, 1, 50)
    T = sc.linspace(0, 1, 50)
    xt = [(x, t) for x in X for t in T]
    xt_vals = sc.repeat(sc.linspace(1, 0, 50), 50).reshape(-1, 50).transpose()
    i = 0
    j = 2
    k = int(50/3)
    i_part0 = i*k
    i_part1 = (i+1)*k
    j_part0 = j*k
    j_part1 = (j+1)*k
    sc.set_printoptions(precision=3, linewidth=110)
