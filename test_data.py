import os
import sys

import numpy as np
import scipy.sparse as sps
import cvxpy as cp

import prog
import read_qps


if __name__ == "__main__":
    if len(sys.argv) > 1:
        datadir = sys.argv[1]
    else:
        datadir = "data/"
    for file in os.listdir(datadir):
        if file in ["CONT-300.QPS", "BOYD2.QPS", "AUG2DC.QPS", "AUG2DCQP.QPS"]:
            continue
        if file not in ["LISWET7.QPS"]:
            continue  # kek
        print(file)
        with open(os.path.join(datadir, file), "r") as f:
            obj, constraints, X = read_qps.read_qps(f)
        prob = cp.Problem(cp.Minimize(obj), constraints)
        # prob.solve(solver=prog.CustomQP())
        data, chain, inverse_data = prob.get_problem_data(
            cp.CLARABEL, solver_opts={"verbose": True}
        )
        soln = chain.solve_via_data(prob, data)
        prob.unpack_results(soln, chain, inverse_data)
        print(f"{file}: {prob.status=}, {prob.value=}")
        A = data[cp.settings.A]
        b = data[cp.settings.B]
        q = data[cp.settings.C]
        if cp.settings.P in data:
            P = data[cp.settings.P]
        else:
            P = 0
        # the actual call is clarabel.DefaultSolver(P,q,A,b,...)
        x = soln.x
        z = soln.z
        s = soln.s
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
        diffterm = alpha * np.abs(r1t @ x - r2t @ z) / np.sqrt(diffden)
        print(f"Dual residual term: {r1term}")
        print(f"Primal residual term: {r2term}")
        print(f"Residual diff term: {diffterm}")
        print(f"{alpha=},{gamma=},{beta=},{xn=},{zn=}")
        print()
