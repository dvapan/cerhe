import scipy as sc
from itertools import *
from functools import *
from pprint import pprint
import utils as ut
import lp_utils as lut
from constants import *
from polynom import *

    funcs = dict({
        'gas2gas' : lambda x: (tgp(x[:-1]) - tcp(x)) * coef["k1"] + coef["k2"] * (tgp(x[:-1], [1, 0]) * coef["wg"] + tgp(x[:-1], [0, 1])),
        'gas2cer' : lambda x: (tgp(x[:-1]) - tcp(x)) * coef["alpha"] - coef["lam"] * tcp(x, [0, 0, 1]),
        'cer3'    : lambda x: tcp(x,[0, 1, 0]) - coef["a"]*(tcp(x,[0,0,2]) + 2/R[3] * tcp(x,[0,0,1])),
        'cer2'    : lambda x: tcp(x,[0, 1, 0]) - coef["a"]*(tcp(x,[0,0,2]) + 2/R[2] * tcp(x,[0,0,1])),
        'cer1'    : lambda x: tcp(x,[0, 1, 0]) - coef["a"]*(tcp(x,[0,0,2]) + 2/R[1] * tcp(x,[0,0,1])),
        'cer0'    : lambda x: tcp(x,[0, 0, 1]),
        'gas'     : lambda x: tgp(x[:-1]),
        'cer'     : lambda x: tcp(x),
    })



def make_gas_cer_quad(gas_coeffs=None,
                      cer_coeffs=None,
                      gasr_coeffs=None,
                      cerr_coeffs=None):
    cer = Polynom(3, 3)
    gas = Polynom(2, 3)
    cerr = Polynom(3, 3)
    gasr = Polynom(2, 3)

    if gas_coeffs is not None:
        gas.coeffs = gas_coeffs
    if cer_coeffs is not None:
        cer.coeffs = cer_coeffs
    if gasr_coeffs is not None:
        gasr.coeffs = gasr_coeffs
    if cerr_coeffs is not None:
        cerr.coeffs = cerr_coeffs
    context_test = Context()
    context_test.assign(gas)
    context_test.assign(cer)
    context_test.assign(gasr)
    context_test.assign(cerr)
    return gas, cer, gasr, cerr


pc = sc.loadtxt("poly_coeff")

def parse_reg(pr):
    if type(pr) is str:
        return pr
    else:
        return pr[0]+pr[1][0] + "_" + str(pr[1][1])


def make_coords(ids,type):
    i, j = ids
    print(i,j)
    xv, tv = sc.meshgrid(X_part[i], T_part[j])
    xv = xv.reshape(-1)
    tv = tv.reshape(-1)

    if type in "lr":
        xt = ut.boundary_coords((X_part[i], T_part[j]))[type]
    elif type in "tb":
        xt = ut.boundary_coords((X_part[i], T_part[j]))[type]        
    elif type == "i":
        xt = sc.vstack([xv, tv]).T
    elif type == "c":
        xt = None
    return xt


def make_id(x):
    return x[0]*treg + x[1]
    
def parse(eq, regs):
    rg = regs[0]
    rg1 = regs[1]

    if max_reg != 1:
        gc,cc,gcr,ccr = sc.split(pc[make_id(rg[1][1])], [10,30,40])
    else:
        gc,cc,gcr,ccr = sc.split(pc, [10,30,40])
    tgp, tcp, tgr, tcr = make_gas_cer_quad(gc,cc,gcr,ccr)

    if max_reg != 1:
        gc1,cc1,gcr1,ccr1 = sc.split(pc[make_id(rg1[1][1])], [10,30,40])
    else:
        gc1,cc1,gcr1,ccr1 = sc.split(pc, [10,30,40])
    tgp1, tcp1, tgr1, tcr1 = make_gas_cer_quad(gc1,cc1,gcr1,ccr1)
    print(len(pc))

    funcsr = dict({
        'gas2gas' : lambda x: (tgr(x[:-1]) - tcr(x)) * coef["k1"] - coef["k2"] * (tgr(x[:-1], [1, 0]) * coef["wg"] + tgr(x[:-1], [0, 1])),
        'gas2cer' : lambda x: (tgr(x[:-1]) - tcr(x)) * coef["alpha"] - coef["lam"] * tcr(x, [0, 0, 1]),
        'cer3'    : lambda x: tcr(x,[0, 1, 0]) - coef["a"]*(tcr(x,[0,0,2]) + 2/R[3] * tcr(x,[0,0,1])),
        'cer2'    : lambda x: tcr(x,[0, 1, 0]) - coef["a"]*(tcr(x,[0,0,2]) + 2/R[2] * tcr(x,[0,0,1])),
        'cer1'    : lambda x: tcr(x,[0, 1, 0]) - coef["a"]*(tcr(x,[0,0,2]) + 2/R[1] * tcr(x,[0,0,1])),
        'cer0'    : lambda x: tcr(x,[0, 0, 1]),
        'gas'     : lambda x: tgr(x[:-1]),
        'cer'     : lambda x: tcr(x),
    })


    funcs1 = dict({
        'gas' : lambda x: tgp1(x[:-1]),
        'cer' : lambda x: tcp1(x),
    })
    funcsr1 = dict({
        'gas' : lambda x: tgr1(x[:-1]),
        'cer' : lambda x: tcr1(x),
    })

    if eq not in ['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0']:
        print(eq + " : " + " ".join(map(parse_reg, regs)))
    if eq in ['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0']:
        if rg[1][0].endswith("_p"):
            fnc1 = lambda x: tgp(x[:-1]) - tcp(x)
            fnc2 = lambda x: tgp(x[:-1], [1, 0]) * coef["wg"] + tgp(x[:-1], [0, 1])
        else:
            fnc1 = lambda x: tgr(x[:-1]) - tcr(x)
            fnc2 = lambda x: tgr(x[:-1], [1, 0]) * coef["wg"] + tgr(x[:-1], [0, 1])

        for i in range(4):
            crds = make_coords(rg[1][1],rg[0])
            crdsq = sc.hstack([crds,sc.full((len(crds),1),R[i])])
            g = lambda x:"({:4.3f}, {:4.3f}, {:4.3f}) {:4.3f} {:4.3f}".format(*x, fnc1(x)[0], fnc2(x)[0])

            print("\n".join(list(map(g, crdsq))))

    elif regs[1][1][0].startswith('base'):
        return
        if regs[0][1][0].endswith("_p"):
            T = TGZ
        else:
            T = TBZ

        ind = make_id(rg[1][1])
        crds = make_coords(rg[1][1],rg[0])
        if rg[1][0].endswith("_p"):
            fnc = funcs
        elif rg[1][0].endswith("_r"):
            fnc = funcsr
        g = lambda x:"({:4.3f}, {:4.3f}, {:4.3f}) {:4.3f} {:4.3f}".format(*x, fnc[eq](x)[0],T)
        crdsq = sc.hstack([crds,sc.full((len(crds),1),R[0])])
        print("\n".join(list(map(g, crdsq))))      
        crdsq = sc.hstack([crds,sc.full((len(crds),1),R[1])])
        print("\n".join(list(map(g, crdsq))))
        crdsq = sc.hstack([crds,sc.full((len(crds),1),R[2])])
        print("\n".join(list(map(g, crdsq))))
        crdsq = sc.hstack([crds,sc.full((len(crds),1),R[3])])
        print("\n".join(list(map(g, crdsq))))      



    else:
        return
        if regs[0][1][0].endswith("_p"):
            T = TGZ
        else:
            T = TBZ

        ind = make_id(rg[1][1])
        crds = make_coords(rg[1][1],rg[0])
        crds1 = make_coords(rg1[1][1],rg1[0])
        if rg[1][0].endswith("_p"):
            fnc = funcs
        elif rg[1][0].endswith("_r"):
            fnc = funcsr

        if rg1[1][0].endswith("_p"):
            fnc1 = funcs1
        elif rg1[1][0].endswith("_r"):
            fnc1 = funcsr1

            
        g = lambda x:"({:4.3f}, {:4.3f}, {:4.3f}) - ({:4.3f}, {:4.3f}, {:4.3f}) {:4.3f} {:4.3f} {:4.3f}".format(*x[0],*x[1], x[2],x[3],abs(x[2]-x[3]))
        g1 = lambda x:fnc[eq](x)[0]
        g2 = lambda x:fnc1[eq](x)[0]
        for i in range(4):
            crdsq = sc.hstack([crds,sc.full((len(crds),1),R[i])])
            crdsq1 = sc.hstack([crds1,sc.full((len(crds),1),R[i])])
            print("\n".join(list(map(g,zip(crdsq,crdsq1,map(g1,crdsq),map(g2,crdsq1))))))




def main():
    list(starmap(parse,
                 chain(
                     ut.construct_mode(['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0'],
                                       'base_1', 1, "l",
                                       ['gas_p', 'cer_p']),
                     ut.construct_mode(['gas2gas', 'gas2cer', 'cer3', 'cer2', 'cer1', 'cer0'],
                                       'base_2', xreg, "r",
                                       ['gas_r', 'cer_r']),
                     ut.intemod_constraints(['cer'], "cer_p", "cer_r"))))

if __name__ == "__main__":
    main()


