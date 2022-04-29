import numpy as np
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray

import logging

import pandas as pd

def solve(A, rhs, ct=None, logLevel=0, extnd=False, 
        basis=False, mps_file=None, outx=None):
    import time
    s = CyClpSimplex()
    s.logLevel = logLevel
    lp_dim = A.shape[1]

    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)
    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    s.objective = x[lp_dim - 1]

    if outx is not None:
        logging.info("start basis status ")
        t0 = time.time()
        resid = np.array(A.dot(outx) - rhs).flatten()
        rstat = np.ones(A.shape[0],dtype=np.int32)
        rstat[resid<1.0e-5] = 2
        cstat = np.ones(A.shape[1],dtype=np.int32)
        t1 = time.time()
        logging.info("start basis status setup")
        s.setBasisStatus(cstat,rstat)
        logging.info("end basis status setup")
        t2 = time.time()
        logging.info(f"time basis setup: {t2-t1}")

    if not mps_file is None:
        s.writeMps(mps_file)
        return None
    logging.debug(f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.primal()
    
    k = list(s.primalConstraintSolution.keys())[0]
    k2 =list(s.dualConstraintSolution.keys())[0]
    q = s.dualConstraintSolution[k2]
    logging.debug(f"{s.getStatusString()} objective: {s.objectiveValue}")

    if extnd and not basis:
        return s.primalVariableSolution['x'],s.primalConstraintSolution[k],s.dualConstraintSolution[k2]
    elif not extnd and basis:
        basis = s.getBasisStatus()
        return s.primalVariableSolution['x'],*basis
    else:
        return s.primalVariableSolution['x']
