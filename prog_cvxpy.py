import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.reductions as cpr
import cvxpy.settings as s

import prog


class CustomQP(cpr.solvers.qp_solvers.qp_solver.QpSolver):
    def name(self):
        return "kek"

    def cite(self):
        return "kek"

    def import_solver(self):
        pass

    def invert(self, solution, inverse_data):
        # copied from copt_qpif.py
        status = solution[s.STATUS]
        attr = {s.NUM_ITERS: solution[s.NUM_ITERS]}

        primal_vars = None
        dual_vars = None
        if status in s.SOLUTION_PRESENT:
            opt_val = solution[s.VALUE] + inverse_data[s.OFFSET]
            primal_vars = {inverse_data[CustomQP.VAR_ID]: solution[s.PRIMAL]}
            # Build dual vars dict keyed by constraint IDs
            z = solution["z"]
            n_eq = self.nz
            eq_dual = cpr.solvers.utilities.get_dual_values(
                z[:n_eq],
                cpr.solvers.utilities.extract_dual_value,
                inverse_data[self.EQ_CONSTR],
            )
            ineq_dual = cpr.solvers.utilities.get_dual_values(
                z[n_eq:],
                cpr.solvers.utilities.extract_dual_value,
                inverse_data[self.NEQ_CONSTR],
            )
            dual_vars = {}
            dual_vars.update(eq_dual)
            dual_vars.update(ineq_dual)
            return cpr.solution.Solution(
                status, opt_val, primal_vars, dual_vars, attr
            )
        else:
            return cpr.solution.failure_solution(status, attr)

    def parse_data(self, data):
        # format here is (copy-paste from QpSolver source):
        """
        minimize      1/2 x' P x + q' x
        subject to    A x =  b
                      F x <= g
        """
        q = data[s.Q]
        A = data[s.A]
        b = data[s.B]
        F = data[s.F]
        g = data[s.G]
        if s.P in data:
            P = data[s.P]
        else:
            P = sp.csc_array((A.shape[1], A.shape[1]))
        # then we actually want A := [A^T F^T]^T, b := [b^T g^T]^T
        # so I took this thing from copt_qpif.py
        n = data["n_var"]
        if A.shape[0] > 0 and F.shape[0] == 0:
            Amat = A
            rhs = b
        elif A.shape[0] == 0 and F.shape[0] > 0:
            Amat = F
            rhs = g
        elif A.shape[0] > 0 and F.shape[0] > 0:
            Amat = sp.vstack([A, F])
            Amat = Amat.tocsc()
            rhs = np.hstack((b, g))
        else:
            Amat = sp.vstack([A, F])
            Amat = Amat.tocsc()
            rhs = None
        self.P = P
        self.q = q
        self.A = Amat
        self.b = rhs
        self.nz = A.shape[0]
        self.nnz = F.shape[0]

    def solve_via_data(
        self,
        data,
        warm_start: bool,
        verbose: bool,
        solver_opts,
        solver_cache=None,
    ):
        """
        warm_start and solver_cache are ignored
        """
        self.parse_data(data)
        solver = prog.KekQP()
        sol = solver.solve(
            self.P, self.q, self.A, self.b, self.nz, verbose, solver_opts
        )
        solution = {}
        solution[s.NUM_ITERS] = sol.num_iters
        # TODO: UPDATE WHEN prog.SStatus updates
        sstat_dict = {
            prog.SStatus.OPTIMAL: s.OPTIMAL,
            prog.SStatus.INFEASIBLE: s.INFEASIBLE,
            prog.SStatus.UNBOUNDED: s.UNBOUNDED,
            prog.SStatus.MAXITER: s.USER_LIMIT,
            prog.SStatus.NUMERICAL: s.SOLVER_ERROR,
            prog.SStatus.OTHER: s.SOLVER_ERROR,
        }
        if sol.status in sstat_dict:
            solution[s.STATUS] = sstat_dict[sol.status]
        else:
            solution[s.STATUS] = s.SOLVER_ERROR
        solution[s.VALUE] = sol.value
        solution[s.PRIMAL] = sol.x
        solution["z"] = sol.z
        solution["s"] = sol.s
        # TODO: MORE STUFF IN solution
        return solution


if __name__ == "__main__":
    x = cp.Variable(3)
    obj = cp.Minimize(cp.quad_form(x, np.eye(3)) + x[1])
    prob = cp.Problem(obj, [x[:2] >= -1, x[1] + x[2] == 1])
    prob.solve(solver=CustomQP())
    print(f"{prob.status=},{prob.value=},{x.value=}")
    print(f"{[c.dual_value for c in prob.constraints]=}")
    prob = cp.Problem(obj, [x[:2] >= 0, x[0] + x[1] <= -1])
    prob.solve(solver=CustomQP())
    print(f"{prob.status=}, should be infeasible")
    obj = cp.Minimize(cp.quad_form(x, np.diag([1, 0, 0])) + x[1])
    prob = cp.Problem(obj, [x <= 0])
    prob.solve(solver=CustomQP())
    print(f"{prob.status=}, should be unbounded")
