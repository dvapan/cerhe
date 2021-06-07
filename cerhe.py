import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

from poly import mvmonos, powers

from constants import *
from gas_properties import TGZ, gas_coefficients
from air_properties import TBZ, air_coefficients
import ceramic_properties as cp

import cProfile


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


t_def = 1000
cff_cnt = [10, 20, 10, 20]


def ceramic(*grid_base, df=None):
    """ Ceramic to ceramic heat transfer
    """
    in_pts_cr = nodes(*grid_base)

    dtchdt = mvmonoss(in_pts_cr, powers(3, 3), 1, cff_cnt, [1, 0, 0])
    dtchdr = mvmonoss(in_pts_cr, powers(3, 3), 1, cff_cnt, [0, 0, 1])
    dtchdr2 = mvmonoss(in_pts_cr, powers(3, 3), 1, cff_cnt, [0, 0, 2])
    radius = in_pts_cr[:, -1].reshape(-1, 1)
    a = cp.a(t_def)
    monos_cerh = dtchdt - a * (dtchdr2 + 2 / radius * dtchdr)

    dtccdt = mvmonoss(in_pts_cr, powers(3, 3), 3, cff_cnt, [1, 0, 0])
    dtccdr = mvmonoss(in_pts_cr, powers(3, 3), 3, cff_cnt, [0, 0, 1])
    dtccdr2 = mvmonoss(in_pts_cr, powers(3, 3), 3, cff_cnt, [0, 0, 2])
    a = cp.a(t_def)
    monos_cerc = dtccdt - a * (dtccdr2 + 2 / radius * dtccdr)

    monos = np.vstack([monos_cerh, monos_cerc])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 0.001)
    return monos, rhs, cff


def gas_air(*grid_base):
    """
    Gas to gas transfer
    """
    in_pts_gs = nodes(*grid_base)
    tgh = mvmonoss(in_pts_gs[:, :-1], powers(3, 2), 0, cff_cnt)
    dtghdt = mvmonoss(in_pts_gs[:, :-1], powers(3, 2), 0, cff_cnt, [1, 0])
    dtghdx = mvmonoss(in_pts_gs[:, :-1], powers(3, 2), 0, cff_cnt, [0, 1])
    tch = mvmonoss(in_pts_gs, powers(3, 3), 1, cff_cnt)
    ALF, PO, CG, WG = gas_coefficients(t_def)
    lb = (tgh - tch) * ALF * surf_spec
    rb = PO * fgib * CG * (dtghdx * WG + dtghdt)
    monos_gash = lb + rb

    tgc = mvmonoss(in_pts_gs[:, :-1], powers(3, 2), 2, cff_cnt)
    dtgcdt = mvmonoss(in_pts_gs[:, :-1], powers(3, 2), 2, cff_cnt, [1, 0])
    dtgcdx = mvmonoss(in_pts_gs[:, :-1], powers(3, 2), 2, cff_cnt, [0, 1])
    tcc = mvmonoss(in_pts_gs, powers(3, 3), 3, cff_cnt)
    ALF, PO, CG, WG = air_coefficients(t_def)
    lb = (tgc - tcc) * ALF * surf_spec
    rb = PO * fgib * CG * (dtgcdx * WG + dtgcdt)
    monos_gasc = lb - rb

    monos = np.vstack([monos_gash, monos_gasc])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 10)
    return monos, rhs, cff


def ceramic_surface(*grid_base):
    """
    Transfer Heat from gas or air to ceramic surface
    """
    in_pts = nodes(*grid_base)

    tch = mvmonoss(in_pts, powers(3, 3), 1, cff_cnt)
    tgh = mvmonoss(in_pts[:, :-1], powers(3, 2), 0, cff_cnt)
    dtchdr = mvmonoss(in_pts, powers(3, 3), 1, cff_cnt, [0, 0, 1])
    ALF, _, _, _ = gas_coefficients(t_def)
    LAM = cp.lam(t_def)
    lbalance = (tgh - tch) * ALF
    rbalance = LAM * dtchdr
    monos_surfh = lbalance - rbalance

    tcc = mvmonoss(in_pts, powers(3, 3), 3, cff_cnt)
    tgc = mvmonoss(in_pts[:, :-1], powers(3, 2), 2, cff_cnt)
    dtccdr = mvmonoss(in_pts, powers(3, 3), 3, cff_cnt, [0, 0, 1])
    ALF, _, _, _ = air_coefficients(t_def)
    LAM = cp.lam(t_def)
    lbalance = (tgc - tcc) * ALF
    rbalance = LAM * dtccdr
    monos_surfc = lbalance - rbalance

    monos = np.vstack([monos_surfh, monos_surfc])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 0.001)
    return monos, rhs, cff


def boundary(val, ind, *grid_base):
    """
    Boundary points for start gas supply from left side of Heat Exchanger
    """
    sb_pts_x0 = nodes(*grid_base)
    monos = mvmonoss(sb_pts_x0[:, :-1], powers(3, 2), ind, cff_cnt)
    rhs = np.full(len(monos), val)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff


def boundary_revert(ind1, bnd1, ind2, bnd2, *grid_base):
    """
    Boundary points for ceramic rever
    """
    sb_pts_t0 = nodes(bnd1, *grid_base)
    sb_pts_t1 = nodes(bnd2, *grid_base)

    tch = mvmonoss(sb_pts_t0, powers(3, 3), 1, cff_cnt)
    tch = shifted(tch, ind1)
    tcc = mvmonoss(sb_pts_t1, powers(3, 3), 3, cff_cnt)
    tcc = shifted(tcc, ind2)
    revtc1 = tch - tcc
    tcc = mvmonoss(sb_pts_t0, powers(3, 3), 3, cff_cnt)
    tcc = shifted(tcc, ind1)
    tch = mvmonoss(sb_pts_t1, powers(3, 3), 1, cff_cnt)
    tch = shifted(tch, ind2)
    revtc2 = tcc - tch

    monos = np.vstack([revtc1, revtc2])
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff


def make_id(i,j):
    return i*xreg + j

def shifted(cffs,shift):
    pcount = len(cffs)
    psize = len(cffs[0])
    lzeros = sc.zeros((pcount, psize * shift))
    rzeros = sc.zeros((pcount, (max_reg - shift-1) * psize))
    cffs = sc.hstack([lzeros,cffs,rzeros])
    return cffs



monos = []
rhs = []
cff = []
for i in range(treg):
    for j in range(xreg):
        conditions = (gas_air(T_part[i], X_part[j], R[0]),
                      ceramic_surface(T_part[i], X_part[j], R[0]),
                      ceramic(T_part[i], X_part[j], R))

        ind = make_id(i, j)
        for m, r, c in conditions:
            m = shifted(m, ind)
            monos.append(m)
            rhs.append(r)
            cff.append(c)

for i in range(treg):
    m,r,c = boundary(TGZ,0, T_part[i],X_part[0][0],R[0])
    ind = make_id(i, 0)
    m = shifted(m, ind)
    monos.append(m)
    rhs.append(r)
    cff.append(c)

for i in range(treg):
    m,r,c = boundary(TBZ, 2, T_part[i], X_part[xreg-1][-1], R[0])
    ind = make_id(i, xreg-1)
    m = shifted(m, ind)
    monos.append(m)
    rhs.append(r)
    cff.append(c)


for j in range(xreg):
    ind1 = make_id(0, j)
    ind2 = make_id(treg-1,j)
    m,r,c = boundary_revert(ind1,T_part[0][0], ind2, T_part[treg - 1][-1], X_part[j],R)
    monos.append(m)
    rhs.append(r)
    cff.append(c)

def betw_blocks(pws, gind,dind, pind, R=None):
    i, j = gind
    di,dj = dind
    if di > 0:
        Ti1 = -1
        Ti2 = 0
    else:
        Ti1 = 0
        Ti2 = -1
    ind = make_id(i, j)
    if R is None:
        grid_base = T_part[i][Ti1], X_part[j]
    else:
        grid_base = T_part[i][Ti1], X_part[j],R
    ptr_bnd = nodes(*grid_base)
    val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
    val = shifted(val, ind)

    ni, nj = i+di, j
    indn = make_id(ni, nj)
    if R is None:
        grid_basen = T_part[ni][Ti2], X_part[nj]
    else:
        grid_basen = T_part[ni][Ti2], X_part[nj], R
    ptr_bndn = nodes(*grid_basen)
    valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
    valn = shifted(valn, indn)

    monos = []

    monos.append(valn - val)

    if dj > 0:
        Tj1 = -1
        Tj2 = 0
    else:
        Tj1 = 0
        Tj2 = -1
    if R is None:
        grid_base = T_part[i], X_part[j][Tj1]
    else:
        grid_base = T_part[i], X_part[j][Tj1],R
    ptr_bnd = nodes(*grid_base)
    val = mvmonoss(ptr_bnd, pws, pind, cff_cnt)
    val = shifted(val, ind)

    ni, nj = i, j+dj
    indn = make_id(ni, nj)
    if R is None:
        grid_basen = T_part[ni], X_part[nj][Tj2]
    else:
        grid_basen = T_part[ni], X_part[nj][Tj2], R

    ptr_bndn = nodes(*grid_basen)
    valn = mvmonoss(ptr_bndn, pws, pind, cff_cnt)
    valn = shifted(valn, indn)

    monos.append(valn - val)
    monos = np.vstack(monos)
    rhs = np.full(len(monos), 0)
    cff = np.full(len(monos), 1)
    return monos, rhs, cff


conditions = []
for i in range(treg - 1):
    for j in range(xreg - 1):
        #gas heat connect blocks
        conditions.append(betw_blocks(powers(3, 2), (i, j),(1,1), 0))
        #cer heat connect blocks
        conditions.append(betw_blocks(powers(3, 3), (i, j),(1,1), 1, R))
for i in range(1,treg):
    for j in range(1, xreg):
        #gas cooling connect blocks
        conditions.append(betw_blocks(powers(3, 2), (i, j),(-1,-1), 2))
        #cer cooling connect blocks
        conditions.append(betw_blocks(powers(3, 3), (i, j),(-1,-1), 3, R))

for m, r, c in conditions:
    monos.append(m)
    rhs.append(r)
    cff.append(c)




def profile(func):
    """Decorator for run function profile"""
    def wrapper(*args, **kwargs):
        profile_filename = func.__name__ + '.prof'
        profiler = cProfile.Profile()
        result = profiler.runcall(func, *args, **kwargs)
        profiler.dump_stats(profile_filename)
        return result
    return wrapper
    

def solve_simplex_splitted(A,rhs, parts):
    import numpy.random
    indeces = np.arange(len(A))
    task = np.hstack([A,rhs.reshape(-1, 1),indeces.reshape(-1,1)])
    np.random.shuffle(task)
    tasks = np.array_split(task,parts)
    indeces_part = np.array_split(indeces, parts)
    old_tasks = []
    state = 'f'
    while len(tasks) > 1:
        new_tasks = []
        xs = []
        for i,task in enumerate(tasks):
            A = task[:,:-2]
            rhs = task[:,-2]
            indeces = task[:,-1]
            x = solve_simplex(A,rhs)
            xs.append(x)
            cnst = np.dot(A,x) - rhs
            sorted_task = task[cnst.argsort()]
            worst_constraint = sorted_task[:len(task)//2, :]
            excl = indeces[:len(task)//2].astype(int)
            cnst_excludes[excl] += 1
            if state == 'spf' or state == 'f':
                new_tasks.append(worst_constraint)
                state = 'sps'
            elif state == 'sps':
                new_tasks[-1] = np.vstack(
                    [new_tasks[-1], worst_constraint])
                state = 'spf'
        state = 'spf'       
        tasks = new_tasks
        old_tasks.append(tasks)

    A_big = task[:,:-2]
    rhs_big = task[:,-2]

    task = tasks[0]
    A = task[:,:-2]
    rhs = task[:,-2]
    x = solve_simplex(A,rhs)
    otkl_big = np.dot(A_big,x) - rhs_big
    worst_constraints_big = task[otkl_big < 1e-10]
    i = np.argmin(otkl_big)
    print(i,otkl_big[i], cnst_excludes[i])
    
    
    # cnst = np.dot(A,x) - rhs
    # sorted_task = task[cnst.argsort()]
    # worst_constraint = sorted_task[:len(task)//2, :]

    task = np.vstack([task, worst_constraints_big])
    A = task[:,:-2]
    rhs = task[:,-2]
    print("add all unfulfilled constraints")
    x = solve_simplex(A,rhs)

    return x
        
        

def solve_simplex(A, rhs):
    s = CyClpSimplex()
    s.logLevel = 1
    lp_dim = A.shape[1] 

    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)

    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    s += x[lp_dim - 1] <= TGZ
    s.objective = x[lp_dim - 1]

    nnz = np.count_nonzero(A)
    print (f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    print("START SIMPLEX")

    
    s.primal()
    # outx = s.primalVariableSolution['x']
    k = list(s.primalConstraintSolution.keys())
    print("END SIMPLEX")
    print(f"{s.getStatusString()} objective: {s.objectiveValue}")
    print("nonzeros rhs:",np.count_nonzero(s.primalConstraintSolution[k[0]]))
    return s.primalVariableSolution['x']



import sys
np.set_printoptions(threshold=sys.maxsize)

A = sc.vstack(monos)
rhs = np.hstack(rhs)
cff = np.hstack(cff).reshape(-1, 1)


A1 = np.hstack([A, cff])
A2 = np.hstack([-A, cff])
A = np.vstack([A1,A2])

rhs = np.hstack([rhs,-rhs])

cnst_excludes = np.zeros(len(A))


lp_dim = len(A[0])
res = solve_simplex_splitted(A,rhs,16)
otkl = np.dot(A,res)-rhs

i = np.argmin(otkl)
print(i,otkl[i], cnst_excludes[i])

np.savetxt("cnst_excludes.dat",cnst_excludes, fmt = "%d")
np.savetxt("otkl.dat",np.sort(otkl), fmt="%.3f")

pc = sc.split(res[:-1],max_reg)
np.savetxt("cerhe_cff", pc)
