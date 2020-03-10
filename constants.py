import scipy as sc
import more_itertools as mit

TGZ = 1800
TBZ = 778.17
MASS = 26.98154*2+15.9994*3 # Al2O3 г/моль


length = 4                      # Длина теплообменника         [м]
time = 300                      # Время работы теплообменника  [с]
rball = 0.01                    # Радиус однгого шара засыпки  [м]
rbckfill = 2                    # Радиус засыпки               [м]
fi = 0.4                        # Пористость                   [доля]

# Коэффициент перевода в килокаллории
MN = 4186.

# Расчет объема засыпки [м^3]
vbckfill = sc.pi * rbckfill**2 * (1 - fi)

# Расчет количества шаров [шт]
cball = vbckfill/((4*sc.pi/3)*rball**3)

# Расчет удельной площади теплообмена [м^2]
surf_spec = cball * 4 * sc.pi * rball ** 2


# Расчет площади живого сечения для прохождения теплоносителя через засыпку, м2
fgib = sc.pi*fi*rbckfill**2

# Теплопроводность керамики        [ккал/м*с*К]
lamcer_A = sc.array([-43.9595654, 0.0113006956,
              1251.80322, 719874.068])
def lam(TC):
    A = lamcer_A/MN
    return A[0] + A[1]* TC + A[2]/sc.sqrt(TC) + A[3]/TC**2

lam_default = lam(1000)


# Теплоемкость керамики           [ккал/кг*К]
ccer_A1 = sc.array([153.43, 1.9681e-3,
               -900.63,-2.0307e+6])
ccer_A2 = sc.array([100.849518,0.150388616, 
               -1176.47884,149808.151])    

def ccer(TC):
    MN = MASS*4.186
    if TC > 298.1:
        A = ccer_A1/MN
    else:
        A = ccer_A2/MN
    return A[0] + A[1]* TC + A[2]/sc.sqrt(TC) + A[3]/TC**2

ccer_default = ccer(1000)

# Плотность керамики [кг/м^3]
rho_cer_A= sc.array([-8.90655002e-06,-9.31235452e-02,3.97251537e+03])

def rho_cer(TC):
    A = sc.array([-8.90655002e-06,-9.31235452e-02,3.97251537e+03])
    T = TC - 273.15
    return (A[0]*T+A[1])*T+A[2]

rho_cer_default = rho_cer(1000)

# Температуропроводность керамики | [м^2/с]
def a(TC):
    return lam(TC)/ccer(TC)/rho_cer(TC)

# Удельная масса керамики   | [кг/м]
qmass_spec = vbckfill*rho_cer(293.15)

def vgs(tg, pg, ga, gk, gy, gw):
    m = sc.array([28.016, 32., 44.01, 18.02])
    gcym = ga + gk + gy + gw
    y = sc.array([
        ga/gcym,
        gk/gcym,
        gy/gcym,
        gw/gcym
    ])
    v = 22.41/ m
    rm = sum(v*y)
    rm = 1/rm
    rm = rm*pg*(735./760.)
    rom = 273./tg*rm
    vm = 1/rom
    return vm 

def CA(T):
    A = [sc.nan,5.10173,20.50549,-60.2800,89.7273,-28.7247,-86.8081,95.590,-22.473]
    M=28.016
    AM1=-7.8225
    AM1=AM1/10000.
    X=T/10000.
    H=8.*A(8)*X
       for I in range (1,7)
          H=(H+A(8-I)*(8-I))*X
       end do
    H=(AM1*(-1.)/X**2+A(1)+H)/M
    return H


def CK(T):
    A = [sc.nan, 5.20537,30.47837,-146.8618,421.1702,-584.5084,132.5392,525.966,-409.939]
    M=32.
    AM1=-2.689
    AM1=AM1/10000.
    X=T/10000.
    H=8.*A(8)*X
    for I in range(1,7):
        H=(H+A(8-I)*(8-I))*X
    H=(AM1*(-1.)/X**2.+A(1)+H)/M
    return H

def reprs(ga,gk,gy,gw,tk,pg,w,d1):
    a = [0.1222359E-4,0.7434563E-7,0.11242929E-9,-0.40384652E-13,0.82038929E-17]
    m = [28.013,32.0,44.01,18.015]
    b = [.19640869E-1,.72610926E-4,.14915302E-8]
    gcym = ga+gk+gy+gw
    y = sc.array([ga,gk,gy,gw])/gcym
    t = tk - 273.15
    
    


coef = dict()
coef["alpha"] = 0.027      # Коэффициент теплопередачи       | [ккал/м^2*с*K]
coef["fyd"] = surf_spec    # Удельная поверхность            | [м^2]
coef["po"] = 1.0           # Плотность газа                  | [кг/м^3]
coef["pc"] = rho_cer(1000) # Плотность керамики              | [кг/м^3]
coef["fgib"] = fgib        # Живое сечение                   | [м^2
coef["wg"] = 1.5           # Скорость газа                   | [м/с]
coef["cg"] = 0.3           # Теплоемкость газа               | [ккал/кг*К]
coef["ck"] = ccer(1000)    # Теплоемкость керамики           | [ккал/кг*К]
coef["lam"] = lam(1000)    # Теплопроводность керамики       | [ккал/м*с*К]

coef["qmacyd"] = 29405.0    # Удельная масса керамики   | [кг/м]


coef["a"] = coef["lam"]/(coef["pc"]*coef["ck"]) # Температуропроводность керамики | [м^2/с]

coef["k1"] = coef["alpha"] * coef["fyd"]
coef["k2"] = coef["po"] * coef["fgib"] * coef["cg"]
coef["k3"] = coef["ck"] * coef["qmacyd"]

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




