import scipy as sc

length = 1
time = 50

TGZ = 1800
TBZ = 778.17


coef = dict()
coef["alpha"] = 0.027*time
coef["fyd"] = 2262.0 * length
coef["po"] = 1.0
coef["fgib"] = 5.0
coef["wg"] = 1.5 * time
coef["cg"] = 0.3
coef["lam"] = 0.0038 * time * length
coef["a"] = 2.3e-6 * time
coef["ck"] = 0.3
coef["qmacyd"] = 29405.0 * length

coef["k1"] = coef["alpha"]*coef["fyd"]
coef["k2"] = coef["po"]*coef["fgib"]*coef["cg"]
coef["k3"] = coef["ck"]*coef["qmacyd"]


def g2c(x, tc, tg):
    de_bln_1 = (tg(x)- tc(x))*coef["k1"] + coef["k2"]*(tg(x,[1,0])*coef["wg"] +
                                                        tg(x,[0,1]))
    de_bln_2 = (tg(x) - tc(x))*coef["k1"] - coef["k3"]*tc(x,[0,1])
    return sc.vstack((de_bln_1, de_bln_2))

def c2a(x, tc, tg):
    de_bln_1 = (tg(x)- tc(x))*coef["k1"] - coef["k2"]*(tg(x,[1,0])*coef["wg"] +
                                                        tg(x,[0,1]))
    de_bln_2 = (tg(x) - tc(x))*coef["k1"] - coef["k3"]*tc(x,[0,1])
    return sc.vstack((de_bln_1, de_bln_2))

def delta_polynom_val(x, polynom, vals, deriv = None):
    t = polynom(x, deriv)
    t[:, 0] = t[:, 0] - vals
    return t
