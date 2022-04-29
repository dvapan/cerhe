import numpy as np
from numpy.linalg import inv
from numpy.linalg import det

# from scipy.sparse import csr_matrix
# from scipy.sparse.linalg import inv
from .simplex import solve as solve_simplex

import logging


def count_next_iter_cycle(outx, worst_A, worst_rhs, ds):
    minf_ind= 0
    maxf_ind = 0
    max_min_ind = 0
    minf = np.inf
    maxf = -np.inf
    min_x = outx
    max_x = outx
    for l in range(len(worst_A)):
        upper = np.sum(worst_A[l]* outx)- worst_rhs[l]
        for s in range(len(outx)):
            u = - upper / np.sum(worst_A[l]*ds[s])
            if u < 0: continue
            x = u*ds[s] + outx
            f = x[-1]
            # print(u,ds[s][-1],outx[-1])
            if f <= minf:
                minf = f
                minf_ind = s
                min_x = x
        if minf > maxf:
            maxf = minf
            maxf_ind = l
            max_min_ind = minf_ind
            max_x = min_x


    # print(maxf)
    ind_remove, ind_insert = maxf_ind,max_min_ind
    outx = max_x
    return outx, max_min_ind, maxf_ind

def count_next_iter(outx, worst_A, worst_rhs, ds):
    upper = np.sum(worst_A*outx,axis=1) - worst_rhs
    minf_inds = np.zeros(worst_A.shape[0],dtype=np.int)
    minf = np.zeros(worst_A.shape[0])
    xs = np.zeros((worst_A.shape[0],len(outx)))
    for i in range(worst_A.shape[0]):
        down = np.sum(worst_A[i]* ds, axis=1)
        u_ls = - upper[i] / down

        u_ls[u_ls < 0] = np.nan
        # print(u_ls.shape)
        # print(ds.shape)

        # print("u_ls")
        # print(u_ls)
        # print("ds")
        # print(ds)
        # print("u_ls*ds")
        # print(u_ls*ds)

        x_ls = np.tile(u_ls,(ds.shape[0],1)).T*ds + outx
        f_ls = x_ls[:,-1]

        minf_inds[i] = np.argmin(f_ls)
        minf[i] = np.min(f_ls)
        xs[i] = x_ls[minf_inds[i]]
    maxf_ind = np.nanargmax(minf)
    max_min_ind=minf_inds[maxf_ind]
    # print(np.nanmax(minf), f_ls[maxf_ind,max_min_ind], f_ls[maxf_ind,max_min_ind] - outx[-1])
    # print(minf_inds)
    # print("-----------", ds[max_min_ind][-1])
    # print("-----------", u_ls[maxf_ind*len(outx) + max_min_ind])
    # print("-----------", outx[-1],ds[max_min_ind][-1]* u_ls[maxf_ind*len(outx) + max_min_ind])

    outx = xs[maxf_ind]

    return outx, max_min_ind, maxf_ind


def solve(A, rhs, eps=0.01, next_iter=count_next_iter_cycle, ct=None):
    is_basis_matrix_square = False
    lp_dim = A.shape[1]
    while not is_basis_matrix_square:
        m1 = lp_dim*1
        num_cnst_add = m1*10
        # fix_idx = np.any(np.vstack([ct=="bnd_fnc",ct == "bnd_val",ct == "betw_blocks"]), axis=0)
        # fixed_points = A[fix_idx][::6]
        # nonfixed_points = A[~fix_idx]

        # nfix_idx = np.random.choice(len(nonfixed_points), num_cnst_add, replace=False)
        # nonfixed_points = nonfixed_points[nfix_idx]

        # task_A = np.vstack([fixed_points,nonfixed_points])
        # task_rhs = np.hstack([rhs[fix_idx][::6], rhs[nfix_idx]])
        # task_ct = np.hstack([ct[fix_idx][::6], ct[nfix_idx]])
        nfix_idx = np.random.choice(A.shape[0], num_cnst_add, replace=False)
        task_A = A[nfix_idx]
        task_rhs = rhs[nfix_idx]

        outx,cnst,lmd = solve_simplex(task_A, task_rhs,ct=None, logLevel=1, extnd=True)

        if np.count_nonzero(lmd) == len(outx):
            is_basis_matrix_square = True

    scal = 1e10
    # A /= scal
    # rhs /= scal
    act_A = task_A[lmd != 0]
    act_b = task_rhs[lmd != 0]
    act_A /= scal
    act_b /= scal
    run = True
    iter = 0
    while run:
        iter += 1
        logging.info(f"iteration {iter}")
        inv_A = inv(act_A)

        logging.debug(f"DET(A) {det(act_A)}")
        logging.debug(f"DET(INV(A)) {det(inv_A)}")

        dss = np.eye(len(outx))
        d = dss + act_b
        xs = np.dot(inv_A, d.T).T
        ds = xs - outx
        # print(outx)
        # print(xs[0])
        # print((np.dot(act_A, outx) - act_b))

        logging.debug("resd")
        resd = np.dot(A,outx) - rhs
        logging.info(f"{np.min(resd)} optimal: {outx[-1]}")
        if abs(np.min(resd)) < 1e-2:
            run = False
            break
        logging.debug("worst_A")
        ind = resd.argsort()
        worst_A = A[ind][:1000]#[:m1*2]
        logging.debug("worst_rhs")
        worst_rhs = rhs[ind][:1000]#[:m1*2]

        logging.debug("start_iter_count")
        outx,ind_remove, ind_insert = next_iter(outx, worst_A, worst_rhs, ds)
        logging.debug("finish_iter_count")

        # print((np.dot(worst_A, outx) - worst_rhs))
        act_A = np.delete(act_A, ind_remove, 0)
        act_b = np.delete(act_b, ind_remove)
        act_A = np.vstack([act_A, worst_A[ind_insert]/scal])
        act_b = np.hstack([act_b, worst_rhs[ind_insert]/scal])


    resd = np.dot(A,outx) - rhs
    print(np.min(resd), np.max(resd))
    return outx
