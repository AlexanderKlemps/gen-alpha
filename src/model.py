__author__ = "Alexander Klemps"

# Implementation of classes to describe the physical properties of the
# vanilla heavy top model and one extended by the the presence of disspative
# forces modeled by a torsion spring. 

from lie import *


def my_arctan(y, x):
    if abs(x) < abs(y):
        return np.arctan(y / x)
    else:
        return np.pi / 2 - np.arctan(x / y)


class DAEBaseModel(object):
    def __init__(self):
        self.M = np.eye(6)
        self.B = None

    def evaluate(self, q, v, lamb, dv):
        if self.B is None:
            self.B = np.empty([lamb.shape[0]] * 2)
        residual = self.M @ dv + self.force(q, v) + self.B.T @ lamb
        return residual

    def force(self, q, v):
        return np.empty(v.shape[0])


class HeavyTopModel(DAEBaseModel):
    def __init__(self, group_name):
        super(HeavyTopModel, self).__init__()
        self.m = 15.0
        # self.X = np.array([0, 1, 0])
        # self.J = np.diag([0.234375, 0.46875, 0.234375])
        # self.g = np.array([0, 0, -9.81])
        #self.J = np.diag([15.234375, 0.46875, 15.234375])
        self.X = np.array([1, 0, 0])
        self.J = np.diag([0.46875, 0.234375, 0.234375])
        self.g = np.array([0, -9.81, 0])
        self.M = self.m * np.eye(6)
        self.M[3:, 3:] = self.J
        self.group_name = group_name

        if self.group_name == "so3":
            self.M = self.J

    def C_t(self, v):
        C = np.zeros((6, 6))
        if self.group_name == "so3xr3":
            omega = v[3:]
            C[3:, 3:] = LieGroup.lie_algebra_element(omega) @ self.J - LieGroup.lie_algebra_element(self.J @ omega)
        if self.group_name == "se3":
            u = v[:3]
            omega = v[3:]
            C[3:, 3:] = LieGroup.lie_algebra_element(omega) @ self.J - LieGroup.lie_algebra_element(self.J @ omega)
            C[:3, :3] = self.m * LieGroup.lie_algebra_element(omega)
            C[:3, 3:] = -self.m * LieGroup.lie_algebra_element(u)
        if self.group_name == "so3":
            C = LieGroup.lie_algebra_element(v) @ self.J - LieGroup.lie_algebra_element(self.J @ v)

        return C

    def K_t(self, q, lamb):
        K = np.zeros((6, 6))
        if self.group_name == "so3xr3":
            K[:3, 3:] = q.R @ LieGroup.lie_algebra_element(lamb)
        elif self.group_name == "se3":
            K[:3, 3:] = -LieGroup.lie_algebra_element(self.m * q.R.T @ self.g)
        elif self.group_name == "so3":
            K = -LieGroup.lie_algebra_element(self.X) @ LieGroup.lie_algebra_element(self.m * q.R.T @ self.g)

        return K

    def _external_force(self, q, v):
        F = np.zeros(6)
        if self.group_name == "so3xr3":
            F[:3] = -self.m * self.g
        elif self.group_name == "se3":
            F[:3] = -self.m * q.R.T @ self.g
        elif self.group_name == "so3":
            F = -self.m*LieGroup.lie_algebra_element(self.X) @ q.R.T @ self.g

        return F

    def _internal_force(self, q, v):
        return -self._hat(v).T @ (self.M @ v)

    def constraint(self, q):
        phi = np.empty(shape=(0,))
        if self.group_name != "so3":
            phi = -q.R.T @ q.x + self.X
        return phi

    def _hat(self, v):
        if self.group_name == "so3xr3":
            v_hat = np.zeros((6, 6))
            v_hat[3:, 3:] = LieGroup.lie_algebra_element(v[3:])
        elif self.group_name == "se3":
            v_hat = np.kron(np.eye(2), LieGroup.lie_algebra_element(v[3:]))
            v_hat[:3, 3:] = LieGroup.lie_algebra_element(v[:3])
        elif self.group_name == "so3":
            v_hat = LieGroup.lie_algebra_element(v)

        return v_hat

    def get_constraint_matrix(self, q):
        if self.group_name == "so3xr3":
            B = np.hstack((-q.R.T, -LieGroup.lie_algebra_element(self.X)))
        elif self.group_name == "se3":
            B = np.hstack((-np.eye(3), -LieGroup.lie_algebra_element(self.X)))
        elif self.group_name == "so3":
            B = np.empty(shape=(0, 0))

        return B

    def evaluate(self, q, v, lamb, dv):
        B = self.get_constraint_matrix(q)
        F_ext = self._external_force(q, v)
        F_int = self._internal_force(q, v)
        residual = self.M @ dv + F_int + F_ext
        if self.group_name != "so3":
            residual += B.T @ lamb
        phi = self.constraint(q)

        return np.hstack((residual, phi))

    def _kinetic_energy(self, v):
        e_kin = 0.5 * v.T @ self.M @ v

        return e_kin

    def _potential_energy(self, q):
        if self.group_name == "so3":
            e_pot = -self.m * (q.R @ self.X).T @ self.g
        else:
            e_pot = -self.m * q.x.T @ self.g

        return e_pot

    def total_energy(self, q, v):
        e_kin = self._kinetic_energy(v)
        e_pot = self._potential_energy(q)
        E = e_kin + e_pot

        return E


class HeavyTorsionTopModel(HeavyTopModel):
    def __init__(self, group_name, coeffs=(0.0, 4e3)):
        super(HeavyTorsionTopModel, self).__init__(group_name=group_name)
        self.friction_coeff, self.torsion_ceoff = coeffs
        self.g = np.array([0, -9.81, 0])

    @staticmethod
    def alpha(q):
        R = q.R
        roll = -np.arctan2(R[2, 1], R[2, 2])
        return roll

    @staticmethod
    def dalpha(R):
        r_31, r_32, r_33 = R[2, :]
        e1, e2, e3 = np.eye(3)
        tilde = LieGroup.lie_algebra_element
        dalpha = r_32/r_33**2 * tilde(e3) @ R.T @ e3 - tilde(e2) @ R.T @ e3 / r_33
        dalpha *= 1 / (1 + (r_32/r_33)**2)

        return dalpha

    def _external_force(self, q, v):
        F = super(HeavyTorsionTopModel, self)._external_force(q, v)
        if self.group_name != "so3":
            F[:3] += self.friction_coeff * v[:3]
            omega = v[3:]
        else:
            omega = v

        alpha = HeavyTorsionTopModel.alpha(q)
        dalpha = HeavyTorsionTopModel.dalpha(q.R)
        torsion_force = self.torsion_ceoff * alpha * dalpha
        dissipation_force = self.friction_coeff * omega

        if self.group_name != "so3":
            F[3:] = torsion_force + dissipation_force
        else:
            F += torsion_force + dissipation_force

        return F

    def K_t(self, q, lamb):
        K_t = super(HeavyTorsionTopModel, self).K_t(q, lamb)
        R = q.R
        r_31, r_32, r_33 = R[2, :]
        alpha = HeavyTorsionTopModel.alpha(q)
        d_alpha = HeavyTorsionTopModel.dalpha(R)
        n = r_32 ** 2 + r_33 ** 2

        c1 = -self.torsion_ceoff * alpha
        v1 = -self.torsion_ceoff * d_alpha
        v2 = -r_32*r_31/n * v1
        v2 += -c1/n * (r_31 * np.array([r_33, 0, -r_31]) + r_32 * np.array([0, -r_33, r_32]))
        v2 += 2*c1*r_32*r_31/n**2 * (r_32 * np.array([r_33, 0, -r_31]) + r_33 * np.array([-r_32, r_31, 0]))

        v3 = -r_33 * r_31 / n * v1
        v3 += -c1/n * (r_31 * np.array([-r_32, r_31, 0]) + np.array([0, -r_33, r_32]))
        v3 += 2*c1*r_33*r_31/n**2 * (r_32 * np.array([r_33, 0, -r_31]) + r_33 * np.array([-r_32, r_31, 0]))

        # first = d_alpha
        # n = r_32 ** 2 + r_33 ** 2
        # t = 1 / n
        # t *= np.array([0, r_33, -r_32]) - 2 * r_31 / n * (
        #         r_33 * np.array([-r_32, r_31, 0]) + r_32 * np.array([r_33, 0, -r_31]))
        # second = d_alpha * d_alpha[1] + alpha * (r_32 * t + r_31 / n * np.array([-r_33, 0, r_31]))
        # third = d_alpha * d_alpha[2] + alpha * (r_33 * t + r_31 / n * np.array([r_32, -r_31, 0]))

        #D = np.array([first, second, third])
        D = np.array([v1, v2, v3])
        if self.group_name != "so3":
            K_t[3:, 3:] += D
        else:
            K_t += D

        return K_t

    def C_t(self, v):
        C_t = super(HeavyTorsionTopModel, self).C_t(v)
        if self.group_name != "so3":
            C_t[:3, :3] += self.friction_coeff * np.ones(3)
            C_t[3:, 3:] += self.friction_coeff * np.ones(3)
        else:
            C_t += self.friction_coeff * np.ones(3)

        return C_t

    def _potential_energy(self, q):
        e_pot = super(HeavyTorsionTopModel, self)._potential_energy(q)
        alpha = HeavyTorsionTopModel.alpha(q)
        e_pot += self.torsion_ceoff * 0.5 * alpha ** 2

        return e_pot
