
def count(num_cnst_add=None, eps=0.01):
    import os.path
    import sys
    # if os.path.isfile("test_cff"):
    #     pc = np.loadtxt("test_cff")
    # else:
    #     pc = None
    pc = None
    ofile = sys.argv[1]

    monos, rhs, ct = count_points(poly_coeff=pc)


    ct = np.hstack([ct,ct])

    lp_dim = monos.shape[1] + 1
    ones = np.ones((len(monos),1))


    A1 = np.hstack([monos, ones])
    A2 = np.hstack([-monos, ones])
    A = np.vstack([A1,A2])

    rhs = np.hstack([rhs,-rhs])

    m1 = lp_dim*1
    num_cnst_add = m1
    fix_idx = np.any(np.vstack([ct=="bnd_fnc",ct == "bnd_val",ct == "betw_blocks"]), axis=0)
    fixed_points = A[fix_idx][::2]
    nonfixed_points = A[~fix_idx]

    nfix_idx = np.random.choice(len(nonfixed_points), num_cnst_add, replace=False)
    nonfixed_points = nonfixed_points[nfix_idx]

    print(fixed_points.shape)
    print(nonfixed_points.shape)

    task_A = np.vstack([fixed_points,nonfixed_points])
    task_rhs = np.hstack([rhs[fix_idx][::2], rhs[nfix_idx]])

    run = True
    itnum = 0
    bs = None
    while run:
        stime = time.time()

        outx = solve_simplex(task_A, task_rhs, logLevel=1)
        t = time.time() - stime

        otkl = np.dot(A,outx) - rhs

        itnum += 1
        i = np.argmin(otkl)
        logging.info(f"iter {itnum}; {t:.2f} s")
        logging.debug(f"count otkl < 0: {len(otkl[otkl < 0])} / {len(otkl)}")
        logging.debug(f"count otkl < -{eps}: {len(otkl[otkl < -eps])} / {len(otkl)}")
        # logging.debug(f"count active constraints {len()}")
        logging.debug(f"fx: {outx[-1]} {otkl[i]}")
        logging.debug(f"type cnst {ct[i]}")

        if abs(np.min(otkl)) < eps:
            run = False
            break

        num_cnst_add = max(num_cnst_add, int(np.round(len(task_A)*0.15)))
        num_cnst_add = min(num_cnst_add, len(otkl[otkl < -eps]))
        logging.debug(f"num_cnst_add: {num_cnst_add}")


        worst_A = A[otkl.argsort()][:num_cnst_add]
        worst_rhs = rhs[otkl.argsort()][:num_cnst_add]

        otkl = np.dot(task_A,outx) - task_rhs

        nact_A = task_A[abs(otkl) >= eps]
        nact_rhs = task_rhs[abs(otkl) >= eps]
        act_A = task_A[abs(otkl) < eps]
        act_rhs = task_rhs[abs(otkl) < eps]
        print(nact_A.shape)
        print(act_A.shape)
        q = np.dot(nact_A[:,:-1], act_A[:,:-1].T)
        print(q.shape)
        w1 = np.all(q >= eps, axis = 1)
        w2 = np.all(q <= eps, axis = 1)

        w = w1 | w2
        w = ~w

        la = len(task_A)
        task_A = np.vstack([act_A,nact_A[w]])
        task_rhs = np.hstack([act_rhs,nact_rhs[w]])
        dl = la - len(task_A)
        logging.debug(f"filtered: {dl} constraints")


        task_A = np.vstack([task_A, worst_A])
        task_rhs = np.hstack([task_rhs, worst_rhs])

        # otkl = np.dot(task_A,outx) - task_rhs

    ofile += f"p{max_poly_degree}"
    np.savetxt(ofile, outx)
    print(outx)
