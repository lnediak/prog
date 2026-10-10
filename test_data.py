import os
import sys
import time

import numpy as np
import scipy.sparse as sps

import cvxpy as cp

import prog_cvxpy
import read_qps


def print_stuff(P, A, b, q, x, z, s, tt, ni, file):
    xn = np.linalg.norm(x)
    zn = np.linalg.norm(z)
    r1t = -(P @ x + A.T @ z + q)
    r2t = s + A @ x - b
    # alpha = max(sps.linalg.norm(A) ** 2, 1)
    # gamma = max(np.linalg.norm(q) ** 2, 1)
    # beta = max(np.linalg.norm(b) ** 2, 1)
    alpha = sps.linalg.norm(A) ** 2
    gamma = np.linalg.norm(q) ** 2
    beta = np.linalg.norm(b) ** 2
    r1term = np.linalg.norm(r1t) / np.sqrt(alpha * zn * zn + gamma)
    r2term = np.linalg.norm(r2t) / np.sqrt(alpha * xn * xn + beta)
    diffden = beta * gamma + alpha * (beta * zn * zn + gamma * xn * xn)
    diffterm = alpha * np.abs(r1t @ x + r2t @ z) / np.sqrt(diffden)
    print(f"ELAPSED: {tt}s")
    print(f"num_iters: {ni}")
    print(f"Dual residual term: {r1term}")
    print(f"Primal residual term: {r2term}")
    print(f"Residual diff term: {diffterm}")
    print(f"{alpha=},{gamma=},{beta=},{xn=},{zn=}")
    print()
    with open(os.path.join("out/", file), "w") as fw:
        fw.write(f"ELAPSED: {tt}s\n")
        fw.write(f"num_iters: {ni}\n")
        fw.write(f"{prob.status}\n{prob.value}\n")
        fw.write(f"{np.linalg.norm(r1t)}\n{r1t @ x}\n")
        fw.write(f"{np.linalg.norm(r2t)}\n{r2t @ z}\n")
        fw.write(f"{alpha}\n{beta}\n{gamma}\n")
        fw.write(f"{xn}\n{zn}\n")


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datadir = sys.argv[1]
    else:
        datadir = "data/"
    for file in os.listdir(datadir):
        if file in [
            "CONT-300.QPS",
            "BOYD2.QPS",
            "AUG2DC.QPS",
            "AUG2DCQP.QPS",
            "QFORPLAN.QPS",
        ]:
            continue
        # if file not in ["QSCAGR25.QPS"]:
        #     continue  # kek
        print(file)
        with open(os.path.join(datadir, file), "r") as f:
            obj, constraints, X = read_qps.read_qps(f)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        solver_opts = {
            # "max_iter": 200,
        }
        data, chain, inverse_data = prob.get_problem_data(
            prog_cvxpy.CustomQP(), solver_opts=solver_opts
        )
        start = time.perf_counter()
        soln = chain.solve_via_data(
            prob, data, verbose=False, solver_opts=solver_opts
        )
        kektime = time.perf_counter() - start
        prob.unpack_results(soln, chain, inverse_data)
        print(f"KEK: {file}: {prob.status=}, {prob.value=}")
        Aa = data[cp.settings.A]
        F = data[cp.settings.F]
        A = sps.vstack([Aa, F])
        bb = data[cp.settings.B]
        g = data[cp.settings.G]
        b = np.hstack([bb, g])
        q = data[cp.settings.Q]
        if cp.settings.P in data:
            P = data[cp.settings.P]
        else:
            P = 0
        x = soln[cp.settings.PRIMAL]
        z = soln["z"]
        s = soln["s"]
        ni = prob.solver_stats.num_iters
        print_stuff(P, A, b, q, x, z, s, kektime, ni, file + "-kek.txt")

        # clarabel version
        solver_opts = {
            "equilibrate_enable": False,
            "presolve_enable": False,
            "static_regularization_enable": False,
            "dynamic_regularization_enable": False,
            # "max_iter": 0,
        }
        prob = cp.Problem(cp.Minimize(obj), constraints)
        data, chain, inverse_data = prob.get_problem_data(
            cp.CLARABEL, solver_opts=solver_opts
        )
        start = time.perf_counter()
        soln = chain.solve_via_data(
            prob, data, verbose=False, solver_opts=solver_opts
        )
        clatime = time.perf_counter() - start
        prob.unpack_results(soln, chain, inverse_data)
        print(f"CLARABEL: {file}: {prob.status=}, {prob.value=}")
        A = data[cp.settings.A]
        b = data[cp.settings.B]
        q = data[cp.settings.C]
        if cp.settings.P in data:
            P = data[cp.settings.P]
        else:
            P = 0
        # the actual call is clarabel.DefaultSolver(P,q,A,b,...)
        kekx = x
        kekz = z
        keks = s
        x = soln.x
        z = soln.z
        s = soln.s
        ni = prob.solver_stats.num_iters
        print_stuff(P, A, b, q, x, z, s, clatime, ni, file + "-clarabel.txt")
