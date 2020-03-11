import scipy as sc
import more_itertools as mit



length = 4                      # Длина теплообменника         [м]
time = 300                      # Время работы теплообменника  [с]
rball = 0.01                    # Радиус однгого шара засыпки  [м]
rbckfill = 2                    # Радиус засыпки               [м]
fi = 0.4                        # Пористость                   [доля]
MN = 4186.                      # Коэффициент перевода в килокаллории

# Расчет объема засыпки [м^3]
vbckfill = sc.pi * rbckfill**2 * (1 - fi)

# Расчет количества шаров [шт]
cball = vbckfill/((4*sc.pi/3)*rball**3)

# Расчет удельной площади теплообмена [м^2]
surf_spec = cball * 4 * sc.pi * rball ** 2

# Расчет площади живого сечения для прохождения теплоносителя через засыпку  [м^2]
fgib = sc.pi*fi*rbckfill**2

# Расчет эквивалентного диаметра засыпки (для расчета теплообмена) [м]
dekb=(4/3)*(fi/(1-fi))*rball




TG = 1000






# coef = dict()
# coef["alpha"] = ALF     # Коэффициент теплопередачи       | [ккал/м^2*с*K]
# coef["fyd"] = surf_spec    # Удельная поверхность            | [м^2]
# coef["po"] = PO           # Плотность газа                  | [кг/м^3]
# coef["pc"] = rho_cer(1000) # Плотность керамики              | [кг/м^3]
# coef["fgib"] = fgib        # Живое сечение                   | [м^2
# coef["wg"] = 1.5           # Скорость газа                   | [м/с]
# coef["cg"] = 0.3           # Теплоемкость газа               | [ккал/кг*К]
# coef["ck"] = ccer(1000)    # Теплоемкость керамики           | [ккал/кг*К]
# coef["lam"] = lam(1000)    # Теплопроводность керамики       | [ккал/м*с*К]

# coef["qmacyd"] = 29405.0    # Удельная масса керамики   | [кг/м]


# coef["a"] = coef["lam"]/(coef["pc"]*coef["ck"]) # Температуропроводность керамики | [м^2/с]

# coef["k1"] = coef["alpha"] * coef["fyd"]
# coef["k2"] = coef["po"] * coef["fgib"] * coef["cg"]
# coef["k3"] = coef["ck"] * coef["qmacyd"]


# print(ALF, coef["alpha"])
# print(PO, coef["po"])



# Параметры сетки

xreg,treg = 3,3
max_reg = xreg*treg
max_poly_degree = 3
ppr = 10                        # Точек на регион

totalx = xreg*ppr - xreg + 1
totalt = treg*ppr - treg + 1

dx = length/xreg
dt = time/treg

X = sc.linspace(0, length, totalx)
T = sc.linspace(0, time, totalt)
R = sc.linspace(0.01*rball, rball, 10)
R = R[::-1]

X_part = list(mit.windowed(X,n=ppr,step = ppr-1))
T_part = list(mit.windowed(T,n=ppr,step = ppr-1))

index_info = 0
cnt_var = 0


