import scipy as sc

import utils as ut

xreg,treg = 1,1
max_reg = xreg*treg

length = 1                      # Длина теплообменника                   | [м]
time = 20                       # Время работы теплообменника            | [с]
radius = 0.01                   # Радиус заполнителя (керамический шарик)| [м]

X = sc.linspace(0, length, 10)
T = sc.linspace(0, time, 10)
R = sc.linspace(0, radius, 5)
X_part = sc.split(X, xreg)
T_part = sc.split(T, treg)
index_info = 0
cnt_var = 0


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

