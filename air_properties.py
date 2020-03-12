import scipy as sc
from constants import fgib,dekb,MN

TBZ = 778.17                  # Т-ра воздуха в начальный момент времени [К]
PB = 20.9                     # Давление воздуха на входе
GB = 11.426                   # расход воздуха
MM = 28.98                    # Молекулярная масса воздуха

# Состав воздуха
COMPOSITION = sc.array([
    0.7594,                     # N_2
    0.2306,                     # O_2
    0.01                        # H_2
])

def volume(P, T):
    return T/(353.27*P)

def CB(T):    # Изобарная теплоемкость воздуха
    A = [sc.nan,0.31519,3.56195E-6,6.07609E-8,-5.13003E-11,1.77164E-14,-2.26167E-18]
    TC=T-273.15
    D=28.97
    B=A[6]*TC*6.
    for I in range(4):
        B=(B+A[6-I]*(6-I))*TC
    B=(B+A[1])*(22.41/D)
    return B

def REPRB(TK, P, W, D):
    VB = volume(P, TK)
    C = CB(TK)
    T = TK - 273.15
    # Кинематическая вязкость
    ZNU = ((((0.23531342E-16 * T - 0.56219934E-13) * T + 0.11379092E-9) * T + 0.86303219E-7) + 0.13334426E-4) * 760. / (P * 735.)
    # Коэффициент теплопроводности
    ALA = (((-0.10448614E-14 * T + 0.73307439E-11) * T - 0.22533805E-7) * T + 0.64300069E-4) * T + 0.021049487
    print (ZNU,C,ALA,VB)
    PR = 3600. * ZNU * C / (ALA * VB)
    ZMU = ZNU / (9.81 * VB) # Динамическая вязкость 
    RE = W * D / ZNU
    return RE, PR, VB, ALA, ZMU

def gas_speed(G,V,F):
    return G*V/F

def calc_ES(RE): # Коэффициент для аэродинамического расчета
    if RE < 2000:
        ES = 36.4 / RE + 0.45
    else:
        ES = 1.09 / RE**0.4
    return ES

def calc_NUS(RE, PR):
    QNUS = 0.39 * PR**0.33 * RE**0.64
    return QNUS


def air_coefficients(T):
    VB = volume(PB, T)
    CG = CB(T)
    WG = gas_speed(GB, VB, fgib)
    RE,PR,VM,ALA,ZMUM = REPRB(T,PB,WG,dekb)
    # print(RE,PR,ALA,dekb,MN, WG)
    ALF = (calc_NUS(RE, PR) * ALA /dekb) /MN
    PO = 1/VB
    return ALF, PO

if __name__ == "__main__":
    import sys
    T = float(sys.argv[1])
    ALF, PO = air_coefficients(T)
    print("Коэффициент теплопередачи [kcal/m^2*s*K] ALF =", ALF)
    print("Плотность газа            [кг/м^3]       PO  =", PO)
