import scipy as sc

import utils as ut

xreg,treg = 5,5
max_reg = xreg*treg
X = sc.linspace(0, 1, 50)
T = sc.linspace(1, 50, 50)
X_part = sc.split(X, xreg)
T_part = sc.split(T, treg)
index_info = 0
cnt_var = 0

length = 1
time = 50

TGZ = 1800
TBZ = 778.17

coef = dict()
coef["alpha"] = 0.027 
coef["fyd"] = 2262.0 
coef["po"] = 1.0
coef["fgib"] = 5.0
coef["wg"] = 1.5 
coef["cg"] = 0.3
coef["lam"] = 0.0038 
coef["a"] = 2.3e-6
coef["ck"] = 0.3
coef["qmacyd"] = 29405.0 

coef["k1"] = coef["alpha"] * coef["fyd"]
coef["k2"] = coef["po"] * coef["fgib"] * coef["cg"]
coef["k3"] = coef["ck"] * coef["qmacyd"]


