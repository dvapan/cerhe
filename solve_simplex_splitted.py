import numpy as np
import scipy as sc
from scipy.sparse import csr_matrix
from cylp.cy import CyClpSimplex
from cylp.py.modeling.CyLPModel import CyLPArray
from cylp.py.pivots import DantzigPivot
from cylp.py.pivots import MostFrequentPivot


def solve_simplex_splitted(A,rhs, parts):
    import numpy.random
    base_task = np.hstack([A,rhs.reshape(-1, 1)])
    hyp = None
    np.random.shuffle(base_task)
    tasks = np.array_split(base_task,parts)
    old_tasks = []
    state = 'f'
    while len(tasks) > 1:
        new_tasks = []
        wc = []
        for i,task in enumerate(tasks):
            print("#"*102)
            print("#{:^100s}#".format("HEAP ITERATION {}".format(len(tasks))))
            print("#"*102)

            x, u, wc, hyp = solve_simplex_hyperplanes(task,hyp,1)
         
            print(hyp.shape)
            if state == 'spf' or state == 'f':
                new_tasks.append(wc)
                state = 'sps'
            elif state == 'sps':
                new_tasks[-1] = np.vstack(
                    [new_tasks[-1], wc])
                state = 'spf'
        state = 'spf'       
        tasks = new_tasks
        old_tasks.append(tasks)


    print("#"*102)
    print("#{:^100s}#".format("HEAP ITERATION {}".format(len(tasks))))
    print("#"*102)

        
    task = tasks[0]
    x, u, wc, hyp = solve_simplex_hyperplanes(task,hyp,1)
    
       
    print("#"*102)
    print("#{:^100s}#".format("HEAP SOLUTION FINISH"))
    print("#"*102)

    print(wc.shape)
    
    ones = np.ones(len(A)).reshape(-1,1)

    A_pos = np.hstack([A, ones])
    A_neg = np.hstack([-A, ones])

    task_pos = np.hstack([A_pos[:,:-1], rhs.reshape(-1,1), np.arange(len(A_pos)).reshape(-1,1), np.zeros(len(A_pos)).reshape(-1,1)])
    task_neg = np.hstack([A_neg[:,:-1], -rhs.reshape(-1,1), np.arange(len(A_neg)).reshape(-1,1), np.ones(len(A_pos)).reshape(-1,1)])
    task = np.vstack([task_pos, task_neg])

    wc_all = None
    
    iteration = 0
    while iteration < 100:
        print("#"*102)
        print ("#{:^100s}#".format("ITERATION {}".format(iteration)))

        otkl_pos = np.dot(A_pos,x) - rhs
        otkl_neg = np.dot(A_neg,x) + rhs
        otkl    = np.hstack([otkl_pos, otkl_neg])
        sorted_task = task[otkl.argsort()]
    
        worst_constraint = sorted_task[:2000, :]
        
        worst_constraint_pos = worst_constraint[worst_constraint[:,-1] == 0,:-1]
        worst_constraint_neg = worst_constraint[worst_constraint[:,-1] == 1,:-1]
        print("wcn",worst_constraint_neg.shape)
        print("wcp",worst_constraint_pos.shape)

        ind = worst_constraint_neg[:,-1].astype(int)
        worst_constraint_neg_added = np.hstack([A_pos[ind,:-1],rhs[ind].reshape(-1,1)])
        
        # worst_constraint_pos = np.vstack([worst_constraint_pos[:,:-1], worst_constraint_neg_added])

        wc_big = np.vstack([worst_constraint_pos[:,:-1], worst_constraint_neg_added])

        if wc_all is None:
            wc_all = np.vstack([worst_constraint_pos[:,:-1], worst_constraint_neg[:,:-1]])
        else:
            wc_all = np.vstack([wc_all, worst_constraint_pos[:,:-1], worst_constraint_neg[:,:-1]])

        i = np.argmin(otkl)
        print("#{:^100s}#".format("nonzero otkl: {} / {}".format(len(otkl[otkl < 0]), len(otkl))))
        print("#{:^100s}#".format("{} {}".format(i,otkl[i])))
        print("#"*102)
     
        cur_task = np.vstack([wc,wc_all])
        np.savetxt("task.dat",cur_task, fmt = "%.3f")

        x,u,wc,hyp = solve_simplex_hyperplanes(cur_task,hyp,1)

        #print(u)

        iteration += 1
    
    return x


def solve_simplex_hyperplanes(task, hyperplanes, logLevel=0):
    
    A = task[:,:-1]
    rhs = task[:,-1]

    ones = np.ones(len(A)).reshape(-1,1)
    
    A_pos = np.hstack([A, ones])
    A_neg = np.hstack([-A, ones])
    if hyperplanes is not None:       
        A_hyp = hyperplanes[:,:-1]
        rhs_hyp = hyperplanes[:,-1]
        print("Apos",A_pos.shape)
        print("Ahyp",A_hyp.shape)
        A_all = np.vstack([A_pos,A_neg])
        rhs_all = np.hstack([rhs,-rhs])
        # A_all = np.vstack([A_pos,A_neg,A_hyp])
        # rhs_all = np.hstack([rhs,-rhs, rhs_hyp])
    else:
        A_all = np.vstack([A_pos,A_neg])
        rhs_all = np.hstack([rhs,-rhs])

    x, u = solve_simplex(A_all,rhs_all, logLevel)
    otkl_pos = np.dot(A_pos,x) - rhs
    otkl_neg = np.dot(A_neg,x) + rhs
    otkl = np.hstack([otkl_pos, otkl_neg])
  
    task_pos = np.hstack([A_pos[:,:-1], rhs.reshape(-1,1), np.arange(len(A_pos)).reshape(-1,1), np.zeros(len(A_pos)).reshape(-1,1)])
    task_neg = np.hstack([A_neg[:,:-1], -rhs.reshape(-1,1), np.arange(len(A_neg)).reshape(-1,1), np.ones(len(A_pos)).reshape(-1,1)])
    task = np.vstack([task_pos, task_neg])
    
    sorted_task = task[otkl.argsort()]
    worst_constraint = sorted_task[:len(task)//4, :]
    worst_constraint_pos = worst_constraint[worst_constraint[:,-1] == 0,:-1]
    worst_constraint_neg = worst_constraint[worst_constraint[:,-1] == 1,:-1]

    ind = worst_constraint_neg[:,-1].astype(int)
    worst_constraint_neg_added = np.hstack([A_pos[ind,:-1],rhs[ind].reshape(-1,1)])
    print("wcn+",worst_constraint_neg_added.shape)
    print("wc",worst_constraint_pos.shape)
    worst_constraint_pos = np.vstack([worst_constraint_pos[:,:-1], worst_constraint_neg_added])

    worst_constraint = worst_constraint_pos

    # ind = worst_constraint_neg[:,-1].astype(int)
    # worst_constraint_neg_added = A_pos[ind]

    # worst_constraint = np.vstack([worst_constraint_pos[:,:-2], worst_constraint_pos_added,
    #                               worst_constraint_neg[:,:-2], worst_constraint_neg_added])

    hyp_rhs = np.dot(np.dot(A_all,x),u)
    hyp_a = np.dot(A_all.T, u)
    hyp_cnst = np.hstack([hyp_a,[hyp_rhs]])
    if hyperplanes is None:
        hyperplanes = hyp_cnst.reshape(1,-1)
    else:
        hyperplanes = np.vstack([hyperplanes, hyp_cnst])

    print("wc",worst_constraint.shape)
    print("hyp",hyperplanes.shape)

    return x,u, worst_constraint, hyperplanes

def solve_simplex(A, rhs, logLevel=0):
    s = CyClpSimplex()
    s.logLevel = logLevel
    lp_dim = A.shape[1] 

   
    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)
    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    # s += x[lp_dim - 1] <= TGZ
    s.objective = x[lp_dim - 1]

    nnz = np.count_nonzero(A)
    print (f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")

    s.primal()
    # outx = s.primalVariableSolution['x']
    k = list(s.primalConstraintSolution.keys())
    k2 =list(s.dualConstraintSolution.keys())
    q = s.dualConstraintSolution[k2[0]]
    print(f"{s.getStatusString()} objective: {s.objectiveValue}")
    print("nonzeros rhs:",np.count_nonzero(s.primalConstraintSolution[k[0]]))
    print("nonzeros dual:",np.count_nonzero(s.dualConstraintSolution[k2[0]]))

    return s.primalVariableSolution['x'], s.dualConstraintSolution[k2[0]]

def solve_simplex_big(A, rhs, logLevel=0):
    ones = np.ones(len(A)).reshape(-1,1)
    
    A_pos = np.hstack([A, ones])
    A_neg = np.hstack([-A, ones])

    A = np.vstack([A_pos,A_neg])
    rhs = np.hstack([rhs,-rhs])

    s = CyClpSimplex()
    s.logLevel = logLevel
    lp_dim = A.shape[1] 

   
    x = s.addVariable('x', lp_dim)
    A = np.matrix(A)
    rhs = CyLPArray(rhs)

    s += A * x >= rhs

    s += x[lp_dim - 1] >= 0
    # s += x[lp_dim - 1] <= TGZ
    s.objective = x[lp_dim - 1]

    nnz = np.count_nonzero(A)
    print (f"TASK SIZE XCOUNT: {lp_dim} GXCOUNT: {len(rhs)}")
    
    s.initialPrimalSolve()
    # outx = s.primalVariableSolution['x']
    k = list(s.primalConstraintSolution.keys())
    k2 =list(s.dualConstraintSolution.keys())
    q = s.dualConstraintSolution[k2[0]]

    print(f"{s.getStatusString()} objective: {s.objectiveValue}")
    print("nonzeros rhs:",np.count_nonzero(s.primalConstraintSolution[k[0]]))
    print("nonzeros dual:",np.count_nonzero(s.dualConstraintSolution[k2[0]]))

    return s.primalVariableSolution['x']
