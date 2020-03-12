import scipy as sc

from constants import fgib,dekb,MN

TGZ = 1800                    # Т-ра газа на входе в теплообменник [К]
PG = 7.6                      # Давление газа на входе 

# Покомпонентный расход газа
GG = sc.array([6.126,         # N_2
               0.659,         # O_2
               1.53780,         # CO_2
               0.41788          # H_2O
]) 


# TODO: Сделать код более питонообразным.

def CA(T):                      
    A = [sc.nan,5.10173,20.50549,-60.2800,89.7273,-28.7247,-86.8081,95.590,-22.473]
    M=28.016
    AM1=-7.8225
    AM1=AM1/10000.
    X=T/10000.
    H=8.*A[8]*X
    for I in range (1,7):
        H=(H+A[8-I]*(8-I))*X
    H=(AM1*(-1./X**2) + A[1] + H)/M
    return H


def CK(T):
    A = [sc.nan, 5.20537,30.47837,-146.8618,421.1702,-584.5084,132.5392,525.966,-409.939]
    M=32.
    AM1=-2.689
    AM1=AM1/10000.
    X=T/10000.
    H=8.*A[8]*X
    for I in range(1,7):
        H=(H+A[8-I]*(8-I))*X
    H=(AM1*(-1./X**2) + A[1] + H)/M
    return H

def CY(T):
    A = [sc.nan, 6.8842,55.2554,-222.559,514.499,-557.259,-48.917,663.79,-423.16]
    M=44.011
    AM1=7.474
    AM1=AM1/10000.
    X=T/10000.
    H=8.*A[8]*X
    for I in range (1,7):
        H=(H+A[8-I]*(8-I))*X
    H=(AM1*(-1.)/X**2 + A[1] + H)/M
    return H

def CW(T):
    A = [sc.nan, 6.8699,10.3562,65.580,-327.904,506.943,-212.793,-445.63,387.42]
    M=18.016
    AM1=-3.961
    AM1=AM1/10000.
    X=T/10000.
    H=8.*A[8]*X
    for I in range(1,7):
       H=(H+A[8-I]*(8-I))*X
    H=(AM1*(-1.)/X**2.+A[1]+H)/M
    return H

def VGS(tg, pg, ga, gk, gy, gw):
    m = sc.array([28.016, 32., 44.01, 18.02])
    gcym = ga + gk + gy + gw
    y = sc.array([
        ga/gcym,
        gk/gcym,
        gy/gcym,
        gw/gcym
    ])
    v = 22.41/ m
    rm = v.dot(y)
    rm = 1/rm
    rm = rm*pg*(735./760.)
    rom = 273./tg*rm
    vm = 1/rom
    return vm 


def CGSS(T, GA, GK, GY, GW):
    CPW = CW(T)
    CPK = CK(T)
    CPA = CA(T)
    CPY = CY(T)
    CPS = (GW * CPW + GA * CPA + GY * CPY + GK * CPK) / (GA + GK + GY + GW)
    return CPS


def REPRS(GG,TK,PG,W,D1):
    A = sc.array([0.1222359E-4,0.7434563E-7,0.11242929E-9,-0.40384652E-13,0.82038929E-17])[::-1]
    M = sc.array([28.013,32.0,44.01,18.015])
    V = 22.41/M
    B = sc.array([.19640869E-1,.72610926E-4,.14915302E-8])[::-1]
    Y = GG/sum(GG)
    T = TK - 273.15
    CP = sc.array([
        CA(TK),
        CK(TK),
        CY(TK),
        CW(TK)])
    CPM = CP.dot(Y)
    RH2O = Y[3]*V[3]/Y.dot(V)
    ROM = (273.15/(T+273.15))/Y.dot(V)
    ROM *= PG*(735./760.)
    VM = 1./ROM
    NU = sc.polyval(A,T)
    NU=NU*760./(PG*735.)
    LA=sc.polyval(B,T)
    print(NU,CPM,LA,VM)
    PR=3600.*NU*CPM/(LA*VM)
    PR=(0.94+0.56*RH2O)*PR
    RE=W*D1/NU
    ZMUM=NU*ROM
    ALA=LA
    return RE, PR, VM, ALA, ZMUM

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


def gas_coefficients(TG):
    VG = VGS(TG, PG, *GG)
    CG = CGSS(TG, *GG)
    WG = gas_speed(sum(GG), VG, fgib)
    RE, PR, VM, ALA, ZMUM = REPRS(GG,TG,PG,WG, dekb)
    # print(RE,PR,ALA,dekb,MN, WG)
    ALF = calc_NUS(RE, PR)*ALA/dekb / MN
    PO = 1. / VG
    return ALF, PO
    
if __name__ == "__main__":
    import sys
    T = float(sys.argv[1])
    ALF, PO = gas_coefficients(T)
    print("Коэффициент теплопередачи [kcal/m^2*s*K] ALF =", ALF)
    print("Плотность газа            [кг/м^3]       PO  =", PO)

