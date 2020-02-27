import scipy as sc
import more_itertools as mit
import utils as ut

xreg,treg = 3,3
max_reg = xreg*treg
max_poly_degree = 3
ppr = 10                        # Точек на регион

totalx = xreg*ppr - xreg + 1
totalt = treg*ppr - treg + 1

length = 4                      # Длина теплообменника        | [м]
time = 300                      # Время работы теплообменника | [с]
radius = 0.01                   # Радиус заполнителя          | [м]
radius_inner = 0.01*radius

dx = length/xreg
dt = time/treg

X = sc.linspace(0, length, totalx)
T = sc.linspace(0, time, totalt)
R = sc.linspace(radius_inner, radius, 10)
R = R[::-1]

X_part = list(mit.windowed(X,n=ppr,step = ppr-1))
T_part = list(mit.windowed(T,n=ppr,step = ppr-1))

index_info = 0
cnt_var = 0


TGZ = 1800
TBZ = 778.17

coef = dict()
coef["alpha"] = 0.027 # Коэффициент теплопередачи       | [ккал/м^2*с*K]
coef["fyd"] = 2262.0  # Удельная поверхность            | [м^2]
coef["po"] = 1.0      # Плотность газа                  | [кг/м^3]
coef["pc"] = 3990     # Плотность керамики              | [кг/м^3]
coef["fgib"] = 5.0    # Живое сечение                   | [м^2
coef["wg"] = 1.5      # Скорость газа                   | [м/с]
coef["cg"] = 0.3      # Теплоемкость газа               | [ккал/кг*К]
coef["ck"] = 0.3      # Теплоемкость керамики           | [ккал/кг*К]
coef["lam"] = 0.0018  # Теплопроводность керамики       | [ккал/м*с*К]

coef["qmacyd"] = 29405.0    # Удельная масса керамики   | [кг/м]

coef["a"] = coef["lam"]/(coef["pc"]*coef["ck"]) # Температуропроводность керамики | [м^2/с]

coef["k1"] = coef["alpha"] * coef["fyd"]
coef["k2"] = coef["po"] * coef["fgib"] * coef["cg"]
coef["k3"] = coef["ck"] * coef["qmacyd"]

