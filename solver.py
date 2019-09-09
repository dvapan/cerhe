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
    print(boundary_coords((X, T)))
