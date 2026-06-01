import numpy as np
import scipy.sparse as sp

import cvxpy as cp
import cvxpy.reductions as cpr
import cvxpy.settings as s


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
        P = data[s.P]
        q = data[s.Q]
        A = data[s.A]
        b = data[s.B]
        F = data[s.F]
        g = data[s.G]
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

    def solver_init(self, eps):
        P = self.P.tocoo()
        A = self.A.tocoo()
        # now to form block matrix
        pn = P.shape[0]
        assert pn == P.shape[1]
        ar, ac = A.shape
        N = pn + ar
        # we follow initialization given in clarabel paper
        data = np.hstack([P.data, A.data, A.data, np.ones(ar)])
        row = np.hstack([P.row, A.col, A.row + pn, np.arange(pn, N)])
        col = np.hstack([P.col, A.row + pn, A.col, np.arange(pn, N)])
        M = sp.coo_array((data, (row, col)), shape=(N, N))
        # TODO: IMPLEMENT P=0 case
        rhs = np.hstack([-self.q, self.b])
        xz = sp.linalg.spsolve(M.tocsc(), rhs)
        self.x = xz[:pn]
        z = xz[pn:]
        alphap = np.max(z[self.nz :])
        s0 = -z if alphap < -eps else -z + np.full_like(z, eps + alphap)
        self.s = s0[self.nz :]
        alphad = np.max(-z[self.nz :])
        self.z = z if alphad < -eps else z + np.full_like(z, eps + alphad)
        self.tau = 1.0
        self.kappa = 1.0

    def compute_res(self):
        P, q, A, b, nz, nnz = self.P, self.q, self.A, self.b, self.nz, self.nnz
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        self.dx = -(P @ x + A.T @ z + q * tau)
        self.dz = A @ x - b * tau
        self.dz[nz:] += s
        self.dtau = kappa + q @ x + b @ z + (x @ P @ x) / tau
        self.ds = np.hstack([np.zeros(nz), s])
        self.dkappa = kappa * tau

    def kktsolve(self, rhs):
        # TODO: TAKE FROM solver_opts
        P, q, A, b, nz, nnz = self.P, self.q, self.A, self.b, self.nz, self.nnz
        P = P.tocoo()
        A = A.tocoo()
        pn = P.shape[0]
        ar, ac = A.shape
        N = pn + ar
        Hd = self.s / self.z[self.nz :]
        data = np.hstack([P.data, A.data, A.data, -Hd])
        row = np.hstack([P.row, A.col, A.row + pn, np.arange(pn + nz, N)])
        col = np.hstack([P.col, A.row + pn, A.col, np.arange(pn + nz, N)])
        M = sp.coo_array((data, (row, col)), shape=(N, N))
        # TODO: CACHE STUFF
        return sp.linalg.spsolve(M.tocsc(), rhs)

    def full_kktsolve(self):
        P, q, A, b, nz, nnz = self.P, self.q, self.A, self.b, self.nz, self.nnz
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        dx, dz, dt, ds, dk = self.dx, self.dz, self.dtau, self.ds, self.dkappa
        pn = P.shape[0]
        Hd = s / z[nz:]
        xi = x / tau
        qbxz = self.kktsolve(np.hstack([-q, b]))
        v1, v2 = qbxz[:pn], qbxz[pn:]
        dif = xi - v1
        dtau_denom = kappa / tau + dif @ P @ dif + v2[nz:] @ (Hd * v2[nz:])
        # TODO: FIX NUMERATOR
        xz1 = self.kktsolve(np.hstack([dx, ds - dz]))
        x1, z1 = xz1[:pn], xz1[pn:]
        dtau_numer = dt - dk / tau + (P @ xi * 2 + q) @ x1 + b @ z1
        self.stau = dtau_numer / dtau_denom
        self.sx = x1 + self.stau * v1
        self.sz = z1 + self.stau * v2
        self.ss = -ds[nz:] - Hd * self.sz[nz:]
        self.skappa = -(dk + kappa * self.stau) / tau

    def ms_nn(self, v, sv):
        v = np.atleast_1d(v)
        sv = np.atleast_1d(sv)
        if not np.any(sv < 0):
            return np.inf
        return np.min(-v[sv < 0] / sv[sv < 0])

    def step(self):
        nz = self.nz
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        sx, sz, ss, st, sk = self.sx, self.sz, self.ss, self.stau, self.skappa
        alpha = min(self.ms_nn(s, ss), self.ms_nn(z[nz:], sz[nz:]))
        alpha = min(alpha, self.ms_nn(tau, st), self.ms_nn(kappa, sk))
        alpha = min(1, alpha * 0.99)  # as cvxopt does
        self.x = x + alpha * sx
        self.z = z + alpha * sz
        self.s = s + alpha * ss
        self.tau = tau + alpha * st
        self.kappa = kappa + alpha * sk

    def check_term(self, teps, ireps, iaeps):
        # TODO: RUIZ UNSCALING
        P, q, A, b, nz, nnz = self.P, self.q, self.A, self.b, self.nz, self.nnz
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        xb = x / tau
        self.primal = xb
        sb = np.hstack([np.zeros(nz), s / tau])
        zb = z / tau
        self.dual = zb
        rp = A @ xb + sb - b
        rd = P @ xb + A.T @ zb + q
        gp = (xb @ P @ xb) / 2 + q @ xb
        self.obj = gp
        gd = -(xb @ P @ xb) / 2 - b @ zb
        nrp = np.linalg.norm(rp)
        nrd = np.linalg.norm(rd)
        gap = np.abs(gp - gd)
        nx = np.linalg.norm(xb)
        ns = np.linalg.norm(sb)
        nz = np.linalg.norm(zb)
        nb = np.max(np.abs(b))
        nq = np.max(np.abs(q))
        if (
            nrp < teps * max(1, nb + nx + ns)
            and nrd < teps * max(1, nq + nx + nz)
            and gap < teps * max(1, min(np.abs(gp), np.abs(gd)))
        ):
            self.status = cp.settings.OPTIMAL
            return True
        # now looking at non-normalized variables
        nx = np.linalg.norm(x)
        nz = np.linalg.norm(z)
        nAz = np.linalg.norm(A.T @ z)
        bz = b @ z
        if bz < -iaeps and nAz < -ireps * bz * max(1, nx + nz):
            self.status = cp.settings.INFEASIBLE
            return True
        ns = np.linalg.norm(s)
        nPx = np.linalg.norm(P @ x)
        Axs = A @ x
        Axs[self.nz :] += s
        nAxs = np.linalg.norm(Axs)
        qx = q @ x
        if (
            qx < -iaeps
            and nPx < -ireps * qx * max(1, nx)
            and nAxs < -ireps * qx * max(1, nx + ns)
        ):
            self.status = cp.settings.UNBOUNDED
            return True
        return False

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
        ieps = solver_opts["init_eps"] if "init_eps" in solver_opts else 1e-8
        teps = solver_opts["term_eps"] if "term_eps" in solver_opts else 1e-8
        ireps = solver_opts["ir_eps"] if "ir_eps" in solver_opts else 1e-8
        iaeps = solver_opts["ia_eps"] if "ia_eps" in solver_opts else 1e-8
        self.parse_data(data)
        # TODO: RUIZ EQUILIBRATION
        self.solver_init(ieps)
        iters = 0
        while not self.check_term(teps, ireps, iaeps):
            self.compute_res()
            # TODO: MEHROTRA CORRECTION
            self.full_kktsolve()
            self.step()
            iters += 1
        solution = {}
        solution[s.NUM_ITERS] = iters
        solution[s.STATUS] = self.status
        solution[s.VALUE] = self.obj
        solution[s.PRIMAL] = self.primal
        solution["z"] = self.dual
        # TODO: MORE STUFF IN solution
        return solution


if __name__ == "__main__":
    x = cp.Variable(3)
    obj = cp.Minimize(cp.quad_form(x, np.eye(3)) + x[1])
    prob = cp.Problem(obj, [x[:2] >= -1, x[1] + x[2] == 1])
    prob.solve(solver=CustomQP())
    print(f"{prob.value=},{x.value=}")
    print(f"{[c.dual_value for c in prob.constraints]=}")
    prob = cp.Problem(obj, [x[:2] >= 0, x[0] + x[1] <= -1])
    prob.solve(solver=CustomQP())
    print(f"{prob.status=}, should be infeasible")
    obj = cp.Minimize(cp.quad_form(x, np.diag([1, 0, 0])) + x[1])
    prob = cp.Problem(obj, [x <= 0])
    prob.solve(solver=CustomQP())
    print(f"{prob.status=}, should be unbounded")
