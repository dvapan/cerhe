import scipy as sc

import utils as ut

xreg,treg = 1,1
max_reg = xreg*treg

length = 1                      # Длина теплообменника        | [м]
time = 20                       # Время работы теплообменника | [с]
radius = 0.04                   # Радиус заполнителя          | [м]
radius_inner = 0.01*radius

X = sc.linspace(0, length, 20)
T = sc.linspace(0, time, 20)
R = sc.linspace(radius_inner, radius, 10)
R = R[::-1]
X_part = sc.split(X, xreg)
T_part = sc.split(T, treg)
index_info = 0
cnt_var = 0


TGZ = 1800
TBZ = 778.17

coef = dict()
coef["alpha"] = 0.027           # Коэффициент теплопередачи       | [ккал/м^2*с*K]
coef["fyd"] = 2262.0            # Удельная поверхность            | [м^2]
coef["po"] = 1.0                # Плотность газа                  | [кг/м^3]
coef["fgib"] = 5.0              # Живое сечение                   | [м^2
coef["wg"] = 1.5                # Скорость газа                   | [м/с]
coef["cg"] = 0.3                # Теплоемкость газа               | [ккал/кг*К]
coef["ck"] = 0.3                # Теплоемкость керамики           | [ккал/кг*К]
coef["lam"] = 0.0038            # Теплопроводность керамики       | [ккал/м*с*К]
coef["a"] = 2.3e-6              # Температуропроводность керамики | [м^2/с]
coef["qmacyd"] = 29405.0        # Удельная масса керамики         | [кг/м]

coef["k1"] = coef["alpha"] * coef["fyd"]
coef["k2"] = coef["po"] * coef["fgib"] * coef["cg"]
coef["k3"] = coef["ck"] * coef["qmacyd"]

