__author__ = "Alexander Klemps"

from lie import *


class BasicGeneralizedAlphaIntegrator(object):
    def __init__(self, model, h=1e-4, rho_inf=1.0, mode=0, tol=1e-6, max_iter=5):
        self.group_name = model.group_name
        assert self.group_name in lie_groups.keys(), "Group not implemented yet."

        self.model = model
        self.lie_group = lie_groups[self.group_name](name=self.group_name)
        self.exp = self.lie_group.exponential_map
        self.T = self.lie_group.tangent_operator
        self.h = h
        self.RHO_INF = rho_inf
        self.mode = mode
        assert self.mode in range(3), "Mode has to be 1, 2 or 3!"
        self.newton_max_iter = max_iter
        self.tolerance = tol

        ALPHA_F = self.RHO_INF / (self.RHO_INF + 1)
        ALPHA_M = (2 * self.RHO_INF - 1) / (self.RHO_INF + 1)
        BETA = 0.25 * (1 + ALPHA_F - ALPHA_M) ** 2
        GAMMA = 0.5 + ALPHA_F - ALPHA_M
        BETA_ = (1 - ALPHA_M) / (BETA * self.h ** 2 * (1 - ALPHA_F))
        GAMMA_ = GAMMA / (BETA * self.h)

        self.parameters = {
            "ALPHA_F": ALPHA_F,
            "ALPHA_M": ALPHA_M,
            "BETA": BETA,
            "GAMMA": GAMMA,
            "BETA_": BETA_,
            "GAMMA_": GAMMA_
        }

    def __iteration_matrix(self, new_q, xx, new_v, new_dv, new_lamb):
        M = self.model.M
        C = self.model.C_t
        K = self.model.K_t
        p = self.parameters
        T = self.lie_group.tangent_operator(xx)
        S = np.zeros((9, 9))
        S[:6, :6] = M * p["BETA_"] + C(new_v) * p["GAMMA_"] + K(new_q, new_lamb) @ T
        B = self.model.get_constraint_matrix(new_q)
        S[:6, 6:] = B.T
        S[6:, :6] = B @ T
        # S = np.hstack((S, B.T))
        # S_ = np.hstack((B @ T, np.zeros((3, 3))))
        # S = np.vstack((S, S_))

        return S

    def solve_time_step(self, q, v, dv, a):
        p = self.parameters
        new_dv = np.zeros(v.shape[0])
        new_lamb = np.zeros(3)
        new_a = (p["ALPHA_F"] * dv - p["ALPHA_M"] * a) / (1 - p["ALPHA_M"])
        new_v = v + self.h * (1 - p["GAMMA"]) * a + p["GAMMA"] * self.h * new_a

        qq = self.phi_h(q, v, q)
        xx = self.phi_x(v, a, new_a)

        new_q = qq

        for i in range(self.newton_max_iter):
            new_q = qq @ self.exp(xx)
            residual = self.model.evaluate(new_q, new_v, new_lamb, new_dv)

            err = np.linalg.norm(residual)
            if err < self.tolerance:
                break

            S = self.__iteration_matrix(new_q, xx, new_v, new_dv, new_lamb)
            delta = np.linalg.solve(S, -residual)
            delta_x = delta[:xx.shape[0]]
            delta_lamb = delta[xx.shape[0]:]
            xx += delta_x
            new_v += p["GAMMA_"] * delta_x
            new_dv += p["BETA_"] * delta_x
            new_lamb += delta_lamb
        new_a += (1 - p["ALPHA_F"]) / (1 - p["ALPHA_M"]) * new_dv

        return new_q, new_v, new_dv, new_a

    def phi_h(self, q, v, a):
        p = self.parameters
        qq = q
        if self.mode in [1, 2]:
            qq = qq @ self.exp(self.h * v)

        if self.mode == 2:
            qq = qq @ self.exp(self.h ** 2 * (0.5 - p["BETA"]) * a)

        return qq

    def phi_x(self, v, a, new_a):
        p = self.parameters
        xx = p["BETA"] * self.h ** 2 * new_a
        if self.mode in [0, 1]:
            xx += self.h ** 2 * (0.5 - p["BETA"]) * a
        if self.mode == 0:
            xx += self.h * v

        return xx


class ScaledGeneralizedAlphaIntegrator(object):
    def __init__(self, model, rho_inf=1.0, tols=(1e-6, 1e-6), max_iter=5):
        self.group_name = model.group_name
        assert self.group_name in lie_groups.keys(), "Group not implemented yet."

        self.model = model
        self.lie_group = lie_groups[self.group_name](name=self.group_name)
        self.exp = self.lie_group.exponential_map
        self.T = self.lie_group.tangent_operator
        self.RHO_INF = rho_inf
        self.newton_max_iter = max_iter
        self.tol_r, self.tol_phi = tols

        ALPHA_F = self.RHO_INF / (self.RHO_INF + 1)
        ALPHA_M = (2 * self.RHO_INF - 1) / (self.RHO_INF + 1)
        BETA = 0.25 * (1 + ALPHA_F - ALPHA_M) ** 2
        GAMMA = 0.5 + ALPHA_F - ALPHA_M
        BETA_ = (1 - ALPHA_M) / (BETA * (1 - ALPHA_F))
        GAMMA_ = GAMMA / BETA

        self.parameters = {
            "ALPHA_F": ALPHA_F,
            "ALPHA_M": ALPHA_M,
            "BETA": BETA,
            "GAMMA": GAMMA,
            "BETA_": BETA_,
            "GAMMA_": GAMMA_
        }

    def __iteration_matrix(self, new_q, dq, new_v, new_dv, new_lamb, h):
        M = self.model.M
        C = self.model.C_t
        K = self.model.K_t
        p = self.parameters
        T = self.lie_group.tangent_operator(h * dq)
        B = self.model.get_constraint_matrix(new_q)
        row1, col1 = M.shape
        row2, col2 = B.shape
        row, col = row1 + row2, col1 + row2
        S = np.zeros((row, col))
        S[:row1, :col1] = M * p["BETA_"] + h * p["GAMMA_"] * C(new_v) + h ** 2 * K(new_q, new_lamb) @ T
        S[row1 - col2:row1, col1:] = B.T
        if self.group_name != "so3":
            #S[row1:, :col1] = B @ T
            T_so3 = SO3.tangent_operator(h * dq[3:])
            S[row1:, 3:6] = -LieGroup.lie_algebra_element(self.model.X) @ T_so3
            S[row1:, :3] = -new_q.R.T
            if self.group_name == "se3":
                S[row1:, :3] = -np.eye(3)

        return S

    def solve_time_step(self, q, v, dv, a, lamb, h):
        p = self.parameters
        dq = v + 0.5 * h * a
        new_v = p["GAMMA_"] * dq + (1 - p["GAMMA_"]) * v + h * (1 - 0.5 * p["GAMMA_"]) * a
        new_dv = p["BETA_"] * ((dq - v) / h - 0.5 * a) + (a - p["ALPHA_F"] * dv) / (1 - p["ALPHA_F"])
        new_lamb = lamb
        new_q = q @ self.exp(h * dq)

        for i in range(self.newton_max_iter):
            new_q = q @ self.exp(h * dq)
            residual = self.model.evaluate(new_q, new_v, new_lamb, new_dv)
            residual[:len(v)] *= h
            residual[len(v):] /= h

            err_r = np.linalg.norm(residual[:6])
            err_phi = np.linalg.norm(residual[6:])

            if err_r < self.tol_r and err_phi < self.tol_phi:
                break
            if i == 0:
                S = self.__iteration_matrix(new_q, dq, new_v, new_dv, new_lamb, h)
            delta_xi = np.linalg.solve(S, -residual)
            delta_dq = delta_xi[:dq.shape[0]]
            delta_hlamb = delta_xi[dq.shape[0]:]
            dq += delta_dq
            new_v += p["GAMMA_"] * delta_dq
            new_dv += p["BETA_"] / h * delta_dq
            new_lamb += 1 / h * delta_hlamb
        new_a = ((1 - p["ALPHA_F"]) * new_dv + p["ALPHA_F"] * dv - p["ALPHA_M"] * a) / (1 - p["ALPHA_M"])

        return new_q, new_v, new_dv, new_a, new_lamb

    def get_initial_values(self, R0, omega0):
        q0 = self.lie_group.new_element(data=(R0 @ self.model.X, R0))
        B = self.model.get_constraint_matrix(q0)
        u0 = np.linalg.solve(B[:, :3], -B[:, 3:] @ omega0)
        v0 = np.hstack((u0, omega0))
        A = np.hstack((self.model.M, B.T))
        A = np.vstack((A, np.hstack((B, np.zeros([B.shape[0]] * 2)))))
        g = self.model._external_force(q0, v0) + self.model._internal_force(q0, v0)
        if self.group_name == "so3xr3":
            Z = -self.lie_group.lie_algebra_element(omega0).T @ R0.T @ u0
        else:
            Z = np.zeros(3)
        b = np.hstack((-g, -Z))
        res = np.linalg.solve(A, b)
        dv0 = res[:len(v0)]
        lamb0 = res[len(v0):]

        return q0, v0, dv0, lamb0
