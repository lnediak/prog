import enum
import numpy as np
import scipy.sparse as sp


# TODO: UPDATE THING
class SStatus(enum.Enum):
    NA = 0
    OPTIMAL = 1
    INFEASIBLE = 2
    UNBOUNDED = 3
    MAXITER = 4
    NUMERICAL = 5
    OTHER = 6


class QPSolution:
    num_iters = 0
    status = SStatus.NA
    value = np.nan
    x = None
    z = None


class DefaultQPKKTSolver:
    def __init__(self, P, A, ir_iter):
        P = P.tocoo()
        A = A.tocoo()
        self.P = P
        self.A = A
        self.ir_iter = ir_iter
        # now to form block matrix
        pn = P.shape[0]
        assert pn == P.shape[1]
        ar, ac = A.shape
        assert pn == ac
        N = pn + ar
        data = np.hstack([np.zeros(ar), P.data, A.data, A.data])
        row = np.hstack([np.arange(pn, N), P.row, A.col, A.row + pn])
        col = np.hstack([np.arange(pn, N), P.col, A.row + pn, A.col])
        self.M = sp.coo_array((data, (row, col)), shape=(N, N))

    def iterref(self, v, rhs):
        for i in range(self.ir_iter):
            res = rhs - self.M @ v
            c = self.lu.solve(res)
            v += c
        return v

    def kktsolve(self, diag, rhs):
        ar, ac = self.A.shape
        if diag is None:
            toret = self.lu.solve(rhs)
            return self.iterref(toret, rhs)
        if len(diag) < ar:
            self.M.data[: ar - len(diag)] = 0
        self.M.data[ar - len(diag) : ar] = diag
        self.lu = sp.linalg.splu(self.M.tocsc())
        toret = self.lu.solve(rhs)
        return self.iterref(toret, rhs)


class KekQP:
    def funny_shift(self, z, nz, eps):
        """WARNING: This mutates z"""
        # this thing isn't in the paper. it's from
        # Clarabel.rs/src/solver/implementations/default/variables.rs:
        # fn _shift_to_cone_interior
        # it also seems crazy arbitrary to me, but whatever
        znz = z[nz:]
        alphap = np.max(-znz, initial=-np.inf)
        posmargin = np.sum(znz[znz > 0], initial=0)
        if nz < len(z):
            target = max(1, eps * posmargin / len(znz))
        else:
            target = 1
        print(f"{alphap=},{posmargin=}")  # TODO: DEBUG
        if alphap >= 0:
            z[nz:] += alphap
            z[nz:] += target
        elif alphap > -target:
            z[nz:] += alphap + target
        return z

    def solver_init(self, eps):
        pn = self.P.shape[0]
        ar, ac = self.A.shape
        nz, nnz = self.nz, self.nnz
        # TODO: IMPLEMENT P=0 case
        self.rhs = np.hstack([-self.q, self.b])
        xz = self.kktsolver.kktsolve(-np.ones(nnz), self.rhs)
        self.x = xz[:pn]
        z = xz[pn:]
        s0 = self.funny_shift(-z, nz, eps)
        self.s = s0[nz:]
        self.z = self.funny_shift(z, nz, eps)
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

    def full_kktsolve(self, set_diag):
        P, q, A, b, nz, nnz = self.P, self.q, self.A, self.b, self.nz, self.nnz
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        dx, dz, dt, ds, dk = self.dx, self.dz, self.dtau, self.ds, self.dkappa
        r2 = ds - dz
        pn = P.shape[0]
        Hd = s / z[nz:]
        xi = x / tau
        self.rhs[:pn] = 2 * (P @ xi) + q
        self.rhs[pn:] = b
        qbxz = self.kktsolver.kktsolve(-Hd if set_diag else None, self.rhs)
        u1, u2 = qbxz[:pn], qbxz[pn:]
        dif = xi - u1
        dtau_denom = kappa / tau + dif @ P @ dif + u2[nz:] @ (Hd * u2[nz:])
        dtau_numer = dt - dk / tau + dx @ u1 + r2 @ u2
        self.stau = dtau_numer / dtau_denom
        self.rhs[:pn] = dx - self.stau * q
        self.rhs[pn:] = r2 + self.stau * b
        xz = self.kktsolver.kktsolve(None, self.rhs)
        self.sx, self.sz = xz[:pn], xz[pn:]
        self.ss = -ds[nz:] - Hd * self.sz[nz:]
        self.skappa = -(dk + kappa * self.stau) / tau

    def ms_nn(self, v, sv):
        """maximum scalar for step to enter/stay in nonnegative orthant"""
        v = np.atleast_1d(v)
        sv = np.atleast_1d(sv)
        if not np.any(sv < 0):
            return np.inf
        return np.min(-v[sv < 0] / sv[sv < 0])

    def get_stepsize(self):
        nz = self.nz
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        sx, sz, ss, st, sk = self.sx, self.sz, self.ss, self.stau, self.skappa
        alpha = min(self.ms_nn(s, ss), self.ms_nn(z[nz:], sz[nz:]))
        alpha = min(alpha, self.ms_nn(tau, st), self.ms_nn(kappa, sk))
        alpha = min(1, alpha)
        return alpha

    def step(self, alpha):
        x, z, s, tau, kappa = self.x, self.z, self.s, self.tau, self.kappa
        sx, sz, ss, st, sk = self.sx, self.sz, self.ss, self.stau, self.skappa
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
        self.slack = sb
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
            self.status = SStatus.OPTIMAL
            return True
        # now looking at non-normalized variables
        nx = np.linalg.norm(x)
        nz = np.linalg.norm(z)
        nAz = np.linalg.norm(A.T @ z)
        bz = b @ z
        if bz < -iaeps and nAz < -ireps * bz * max(1, nx + nz):
            self.status = SStatus.INFEASIBLE
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
            self.status = SStatus.UNBOUNDED
            return True
        return False

    def solve(self, P, q, A, b, nz, verbose, solver_opts):
        """
        minimize      (1/2)*x@P@x + q@x
        subject to    A[:nz,:]@x = b[:nz]
                      A[nz:,:]@x <= b[nz:]
        """
        self.P = P
        self.q = q
        self.A = A
        self.b = b
        self.nz = nz
        self.nnz = A.shape[0] - nz
        assert self.nnz >= 0
        ir_iter = solver_opts.get("ir_iter", 10)
        if "kktsolver" in solver_opts:
            # TODO: do ir_iter for this as well
            self.kktsolver = solver_opts["kktsolver"]
        else:
            self.kktsolver = DefaultQPKKTSolver(P, A, ir_iter)
        ieps = solver_opts.get("init_eps", 0.1)
        teps = solver_opts.get("term_eps", 1e-8)
        ireps = solver_opts.get("ir_eps", 1e-8)
        iaeps = solver_opts.get("ia_eps", 1e-8)
        max_step_frac = solver_opts.get("max_step_fraction", 0.99)
        maxit = solver_opts.get("max_iter", 200)
        maxit = np.inf if maxit < 0 else maxit
        # TODO: RUIZ EQUILIBRATION
        self.solver_init(ieps)
        iters = 0
        while not self.check_term(teps, ireps, iaeps):
            if iters >= maxit:
                self.status = SStatus.MAXITER
                break
            self.compute_res()
            self.full_kktsolve(set_diag=True)
            alpha = self.get_stepsize()
            sigma = (1 - alpha) ** 3
            mu = np.sum(self.s * self.z[self.nz :])
            mu = (mu + self.tau * self.kappa) / (self.nnz + 1)
            m = 1
            if iters == 0:
                # don't ask me why this is done, I found it in
                # Clarabel.rs/src/solver/implementations/default/variables.rs:
                # fn combined_step_rhs
                # condition from src/solver/core/solver.rs
                m = alpha
                self.sz *= m
            etae = self.ss * self.sz[self.nz :]
            etae -= sigma * mu
            self.dx *= 1 - sigma
            self.dz *= 1 - sigma
            self.dtau *= 1 - sigma
            self.dkappa += m * self.skappa * self.stau - sigma * mu
            self.ds[self.nz :] += etae / self.z[self.nz :]
            self.full_kktsolve(set_diag=False)
            alpha = self.get_stepsize() * max_step_frac
            self.step(alpha)
            iters += 1
        solution = QPSolution()
        solution.num_iters = iters
        solution.status = self.status
        solution.value = self.obj
        solution.x = self.primal
        solution.z = self.dual
        solution.s = self.slack
        # TODO: MORE STUFF IN solution
        return solution
