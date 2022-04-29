import numpy as np
import scipy.sparse as sps

from poly import mvmonos, powers
from constants import *
from gas_properties import TGZ, gas_coefficients
from air_properties import TBZ, air_coefficients
import ceramic_properties as cp

ppwrs2 = powers(max_poly_degree, 2)
ppwrs3 = powers(max_poly_degree, 3)
psize2 = len(ppwrs2)
psize3 = len(ppwrs3)

t_def = 1000
cff_cnt = [psize2, psize3, psize2, psize3]

def mvmonoss(x, powers, shift_ind, cff_cnt, diff=None):
    lzeros = sum((cff_cnt[i] for i in range(shift_ind)))
    rzeros = sum((cff_cnt[i] for i in range(shift_ind + 1, len(cff_cnt))))
    monos = mvmonos(x, powers, diff)
    lzeros = np.zeros((len(x), lzeros))
    rzeros = np.zeros((len(x), rzeros))
    return np.hstack([lzeros, monos, rzeros])

def nodes(*grid_base):
    """
    Make list of nodes from given space of points
    """
    grid = np.meshgrid(*grid_base)
    grid_flat = map(lambda x: x.flatten(), grid)
    return np.vstack(list(grid_flat)).T


def make_id(i,j,p):
    return i*p["xreg"] + j

def shifted(cffs,shift,p):
    pcount = len(cffs)
    psize = len(cffs[0])
    max_reg = p["xreg"]*p["treg"]
    lzeros = np.zeros((pcount, psize * shift))
    rzeros = np.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = np.hstack([lzeros,cffs,rzeros])
    return cffs

def make_cnst_name(first, second=None):
    cnst_name = first
    if second != None:
        cnst_name += "_" + name
    return cnst_name


def boundary_val(val,eps,ppwrs, ind, *grid_base, name=None, cf_cff=None):
    """
    Boundary points
    """
    pts = nodes(*grid_base)
    left = mvmonoss(pts, ppwrs, ind, cff_cnt)
    right = np.zeros_like(left)
    rhs = np.full(len(pts), val)
    cff = np.full(len(pts), eps)
    return left, right, rhs, cff, [make_cnst_name("bnd_val",name)]*len(pts)

def boundary_fnc(fnc,eps, ind,  *grid_base, name=None,cf_cff=None):
    """
    Boundary points
    """
    pts = nodes(*grid_base)
    left = mvmonoss(pts, ppwrs, ind, cff_cnt)
    right = np.zeros_like(left)
    rhs = np.apply_along_axis(fnc, 1, pts)
    cff = np.full(len(pts), eps)
    return left, right, rhs, cff, [make_cnst_name("bnd_fnc",name)]*len(pts)

def boundary_revert(ind1, bnd1, ind2, bnd2, eps,params, *grid_base):
    """
    Boundary points for ceramic rever
    """
    sb_pts_t0 = nodes(bnd1, *grid_base)
    sb_pts_t1 = nodes(bnd2, *grid_base)
    lv = []
    rv = []
    tch1 = mvmonoss(sb_pts_t0, powers(3, 3), 1, cff_cnt)
    tch1 = shifted(tch1, ind1,params)
    tcc1 = mvmonoss(sb_pts_t1, powers(3, 3), 3, cff_cnt)
    tcc1 = shifted(tcc1, ind2,params)
    lv.append(tch1)
    rv.append(tcc1)

    tcc2 = mvmonoss(sb_pts_t0, powers(3, 3), 3, cff_cnt)
    tcc2 = shifted(tcc2, ind1, params)
    tch2 = mvmonoss(sb_pts_t1, powers(3, 3), 1, cff_cnt)
    tch2 = shifted(tch2, ind2, params)
    lv.append(tcc2)
    rv.append(tch2)

    lv = np.vstack(lv)
    rv = np.vstack(rv)
    rhs = np.full(len(lv), 0)
    cff = np.full(len(rv), eps)
    return lv,rv, rhs, cff

def betw_blocks(pws, gind,dind, pind, eps, params, X_part, T_part, R=None):
    xreg = params["xreg"]
    treg = params["treg"]
    i, j = gind
    di,dj = dind
    ind = make_id(i, j, params)
    monos = []
    lv = []
    rv = []

    if i < treg - 1:
        if R is None:
            grid_base = T_part[i][-1], X_part[j]
        else:
            grid_base = T_part[i][-1], X_part[j],R
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind, params)
        lv.append(val)

        ni, nj = i+di, j
        indn = make_id(ni, nj, params)
        if R is None:
            grid_basen = T_part[ni][0], X_part[nj]
        else:
            grid_basen = T_part[ni][0], X_part[nj], R
        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn, params)
        rv.append(valn)
        monos.append(valn - val)
    if j < xreg - 1:
        if R is None:
            grid_base = T_part[i], X_part[j][-1]
        else:
            grid_base = T_part[i][-1], X_part[j],R
        ptr_bnd = nodes(*grid_base)
        val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
        val = shifted(val, ind, params)
        lv.append(val)

        ni, nj = i, j+dj
        indn = make_id(ni, nj, params)
        if R is None:
            grid_basen = T_part[ni], X_part[nj][0]
        else:
            grid_basen = T_part[ni], X_part[nj][0], R

        ptr_bndn = nodes(*grid_basen)
        valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
        valn = shifted(valn, indn, params)
        rv.append(valn)
        
        monos.append(valn - val)
    lv = np.vstack(lv)
    rv = np.vstack(rv)
    monos = np.vstack(monos)
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), eps)
    return lv, rv, rhs, cff, ["betw_blocks"]*len(monos)

def count_points(params, v_0=None, cff0=None, a=None, sqp0=None,
        lnp0=None, lnp20=None):
    lvals = []
    rvals = []
    monos = []
    rhs = []
    cff = []
    cnst_type = []
    xreg = params["xreg"]
    treg = params["treg"]
    pprx = params["pprx"]
    pprt = params["pprt"]
    totalx = xreg*pprx - xreg + 1
    totalt = treg*pprt - treg + 1
    X = np.linspace(0, length, totalx)
    T = np.linspace(0, total_time, totalt)
    R = np.linspace(0.01*rball, rball, 10)
    R = R[::-1]
    X_part = list(mit.windowed(X,n=pprx,step=pprx - 1))
    T_part = list(mit.windowed(T,n=pprt,step=pprt - 1))
    bsize = sum(cff_cnt)
    refine_vals = True

    v_old = None
    print('start prepare')
    dtchdt = []
    dtchdr = []
    dtchdr2 = []
    radius = []

    dtccdt = []
    dtccdr = []
    dtccdr2 = []

    tgh = []
    dtghdt = []
    dtghdx = []
    tch = []

    tgc = []
    dtgcdt = []
    dtgcdx = []
    tcc = []


    sh_v = 0
    for i in range(treg):
        for j in range(xreg):
            ind = make_id(i, j, params)
            grid_base = T_part[i], X_part[j],R
            pts = nodes(*grid_base)
            
            dtchdt.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 1,
                cff_cnt, [1, 0, 0]),ind,params)))
            dtchdr.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 1,
                cff_cnt, [0, 0, 1]),ind,params)))
            dtchdr2.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 1,
                cff_cnt, [0, 0, 2]),ind,params)))
            radius.append(pts[:,-1])
            
            dtccdt.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 3,
                cff_cnt, [1, 0, 0]),ind,params)))
            dtccdr.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 3,
                cff_cnt, [0, 0, 1]),ind,params)))
            dtccdr2.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 3,
                cff_cnt, [0, 0, 2]),ind,params)))

            tgh.append(sps.csr_matrix(shifted(mvmonoss(pts[:, :-1], powers(3, 2), 0,
                cff_cnt),ind,params)))
            dtghdt.append(sps.csr_matrix(shifted(mvmonoss(pts[:, :-1], powers(3, 2), 0,
                cff_cnt, [1, 0]),ind,params)))
            dtghdx.append(sps.csr_matrix(shifted(mvmonoss(pts[:, :-1], powers(3, 2), 0,
                cff_cnt, [0, 1]),ind,params)))
            tch.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 1,
                cff_cnt),ind,params)))

            tgc.append(sps.csr_matrix(shifted(mvmonoss(pts[:, :-1], powers(3, 2), 2,
                cff_cnt),ind,params)))
            dtgcdt.append(sps.csr_matrix(shifted(mvmonoss(pts[:, :-1], powers(3, 2), 2,
                cff_cnt, [1, 0]),ind,params)))
            dtgcdx.append(sps.csr_matrix(shifted(mvmonoss(pts[:, :-1], powers(3, 2), 2,
                cff_cnt, [0, 1]),ind,params)))
            tcc.append(sps.csr_matrix(shifted(mvmonoss(pts, powers(3, 3), 3,
                cff_cnt),ind,params)))
    print('end prepare')

    dtchdt = sps.vstack(dtchdt)
    dtchdr = sps.vstack(dtchdr)
    dtchdr2 = sps.vstack(dtchdr2)
    radius = np.hstack(radius)

    dtccdt = sps.vstack(dtccdt)
    dtccdr = sps.vstack(dtccdr)
    dtccdr2 = sps.vstack(dtccdr2)

    tgh = sps.vstack(tgh)
    dtghdt = sps.vstack(dtghdt)
    dtghdx = sps.vstack(dtghdx)
    tch = sps.vstack(tch)

    tgc = sps.vstack(tgc)
    dtgcdt = sps.vstack(dtgcdt)
    dtgcdx = sps.vstack(dtgcdx)
    tcc = sps.vstack(tcc)

    num_points = tgh.shape[0]#totalt*totalx
    lmch = dtchdt
    a = cp.a(t_def)
    rmch = a * (dtchdr2 + 2/radius * dtchdr)

    lmcc = dtccdt
    a = cp.a(t_def)
    rmcc = a * (dtccdr2 + 2/radius * dtccdr)
    
    ALF, PO, CG, WG = gas_coefficients(t_def)
    lmgh = (tgh - tch) * ALF * surf_spec
    rmgh = -PO * fgib * CG * (dtghdx * WG + dtghdt)

    ALF, PO, CG, WG = air_coefficients(t_def)
    lmgc = (tgc - tcc) * ALF * surf_spec
    rmgc = PO * fgib * CG * (dtgcdx * WG + dtgcdt)

    ALF, _, _, _ = gas_coefficients(t_def)
    LAM = cp.lam(t_def)
    lmsh = (tgh - tch) * ALF
    rmsh = LAM * dtchdr

    ALF, _, _, _ = air_coefficients(t_def)
    LAM = cp.lam(t_def)
    lmsc = (tgc - tcc) * ALF
    rmsc = LAM * dtccdr

    monos_cerh = lmch - rmch 
    monos_cerc = lmcc - rmcc 
    monos_gash = lmgh - rmgh
    monos_gasc = lmgc - rmgc
    monos_surh = lmsh - rmsh
    monos_surc = lmsc - rmsc


    rhsch = np.full(num_points, 0)
    rhscc = np.full(num_points, 0)
    rhsgh = np.full(num_points, 0)
    rhsgc = np.full(num_points, 0)
    rhssh = np.full(num_points, 0)
    rhssc = np.full(num_points, 0)

    cffch = np.full(num_points, accs["eq_cer_heat"])
    cffcc = np.full(num_points, accs["eq_cer_cool"])
    cffgh = np.full(num_points, accs["eq_gas_heat"])
    cffgc = np.full(num_points, accs["eq_gas_cool"])
    cffsh = np.full(num_points, accs["eq_sur_heat"])
    cffsc = np.full(num_points, accs["eq_sur_cool"])

    ctch = np.full(num_points, "eq_cer_heat")
    ctcc = np.full(num_points, "eq_cer_cool")
    ctgh = np.full(num_points, "eq_gas_heat")
    ctgc = np.full(num_points, "eq_gas_cool")
    ctsh = np.full(num_points, "eq_sur_heat")
    ctsc = np.full(num_points, "eq_sur_cool")

    lvals.append(lmch)
    lvals.append(lmcc)
    lvals.append(lmgh)
    lvals.append(lmgc)
    lvals.append(lmsh)
    lvals.append(lmsc)

    rvals.append(rmch)
    rvals.append(rmcc)
    rvals.append(rmgh)
    rvals.append(rmgc)
    rvals.append(rmsh)
    rvals.append(rmsc)

    monos.append(monos_cerh)
    monos.append(monos_cerc)
    monos.append(monos_gash)
    monos.append(monos_gasc)
    monos.append(monos_surh)
    monos.append(monos_surc)

    rhs.append(rhsch)
    rhs.append(rhscc)
    rhs.append(rhsgh)
    rhs.append(rhsgc)
    rhs.append(rhssh)
    rhs.append(rhssc)

    cff.append(cffch)
    cff.append(cffcc)
    cff.append(cffgh)
    cff.append(cffgc)
    cff.append(cffsh)
    cff.append(cffsc)

    cnst_type.append([f"{q}-{j}x{i}" for q in ctch] )
    cnst_type.append([f"{q}-{j}x{i}" for q in ctcc] )
    cnst_type.append([f"{q}-{j}x{i}" for q in ctgh] )
    cnst_type.append([f"{q}-{j}x{i}" for q in ctgc] )
    cnst_type.append([f"{q}-{j}x{i}" for q in ctsh] )
    cnst_type.append([f"{q}-{j}x{i}" for q in ctsc] )


    for i in range(treg):
        ind = make_id(i, xreg-1, params)
        lm,rm,r,c,t = boundary_val(TGZ,accs["temp"], ppwrs2, 0,
                T_part[i],X_part[0][0])
        lm = sps.csr_matrix(shifted(lm, ind, params))
        rm = sps.csr_matrix(shifted(rm, ind, params))
        m = lm - rm
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{xreg - 1}x{i}-gaz-bound" for q in t])

    for j in range(xreg):
        ind = make_id(0, j, params)
        lm,rm,r,c,t = boundary_val(TBZ,accs["temp"], ppwrs2, 2,
                T_part[i], X_part[xreg-1][-1])
        lm = sps.csr_matrix(shifted(lm, ind, params))
        rm = sps.csr_matrix(shifted(rm, ind, params))
        m = lm - rm
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{j}x{0}-air-bound" for q in t])

    for j in range(xreg):
        ind1 = make_id(0, j, params)
        ind2 = make_id(treg-1, j, params)
        lm,rm,r,c = boundary_revert(ind1,T_part[0][0],
                ind2, T_part[treg - 1][-1],accs["temp"],params,
                X_part[j],R)
        lm = sps.csr_matrix(lm)
        rm = sps.csr_matrix(rm)
        m = lm - rm
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append([f"{q}-{0}x{i}-revert" for q in t])

    conditions = []
    for i in range(treg):
        for j in range(xreg):
            if i < treg - 1 or j < xreg - 1:
                lm,rm, r, c, t = betw_blocks(ppwrs2, (i, j),(1,1), 0,
                        accs["temp"],params, X_part, T_part)
                t = [f"{q}-{j}x{i}" for q in t]
                m1 = sps.csr_matrix(lm - rm)
                conditions.append((m1,r,c,t))

                lm,rm, r, c, t = betw_blocks(ppwrs3, (i, j),(1,1), 1,
                        accs["temp"], params, X_part, T_part, R)
                t = [f"{q}-{j}x{i}" for q in t]
                m2 = sps.csr_matrix(lm - rm)
                conditions.append((m2,r,c,t))

                lm,rm, r, c, t = betw_blocks(ppwrs2, (i, j),(1,1), 2,
                        accs["temp"], params, X_part, T_part)
                t = [f"{q}-{j}x{i}" for q in t]
                m3 = sps.csr_matrix(lm - rm)
                conditions.append((m3,r,c,t))

                lm,rm, r, c, t = betw_blocks(ppwrs3, (i, j),(1,1), 3,
                        accs["temp"], params, X_part, T_part, R)
                t = [f"{q}-{j}x{i}" for q in t]
                m4 = sps.csr_matrix(lm - rm)
                conditions.append((m4,r,c,t))
    for m, r, c, t in conditions:
        monos.append(m)
        rhs.append(r)
        cff.append(c)
        cnst_type.append(t)

    lvals = sps.vstack(lvals)
    rvals = sps.vstack(rvals)

    monos = sps.vstack(monos)
    rhs = sps.hstack(rhs)
    if cff0 is None:
        cff = np.hstack(cff)
    else:
        cff = cff0
    rhs /= cff
    monos /= cff.reshape(-1,1)

    cnst_type = np.hstack(cnst_type)

    return monos, rhs, cnst_type 
