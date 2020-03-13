import scipy as sc
from constants import MN,vbckfill

MASS = 26.98154*2+15.9994*3       # Al2O3 г/моль

# Теплопроводность керамики        [ккал/м*с*К]
lamcer_A = sc.array([-43.9595654, 0.0113006956,
              1251.80322, 719874.068])/MN
def lam(TC):
    A = lamcer_A
    return A[0] + A[1]* TC + A[2]/sc.sqrt(TC) + A[3]/TC**2

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

# Плотность керамики [кг/м^3]
def rho_cer(TC):
    A = sc.array([-8.90655002e-06,-9.31235452e-02,3.97251537e+03])
    T = TC - 273.15
    return (A[0]*T+A[1])*T+A[2]

# Температуропроводность керамики | [м^2/с]
def a(TC):
    return lam(TC)/ccer(TC)/rho_cer(TC)

# Удельная масса керамики   | [кг/м]
qmass_spec = vbckfill*rho_cer(293.15)

