__author__ = "Alexander Klemps"

# General implementations for calculus within the Lie Groups
#   SO3, SE3 and the cartesian product S03xR3
# contained in utility classes of the same names.
# Each class provides the following operators
#   - exponential operator
#   - tangent operator
#   - the tilde-operator mapping identifying a vector v in R^3 with an element of the Lie Group
#
# In addition for every Lie group element is represented by an object of
# an element-class LieGroupElement, overloading certain basic operators to 
# facilitate calculus with these.

import numpy as np


class LieGroupElement(object):
    def __init__(self, name, data):
        self.name = name

        assert name in ["so3xr3", "se3", "so3"], "Group not implemented yet."

        if type(data) == tuple and len(data) == 2:
            self.x, self.R = data
            self.M = None
        elif type(data) == np.ndarray:
            self.M = data
            self.x = None
            self.R = None

        if self.M is None:
            if name == "so3xr3":
                self.M = np.eye(7)
            elif name == "se3":
                self.M = np.eye(4)

            if self.name != "so3":
                self.M[:3, :3] = self.R
                self.M[-4:-1, -1] = self.x

        elif self.x is None:
            self.R = self.M[:3, :3]
            if self.name != "so3":
                self.x = self.M[-4:-1, -1]

    def __matmul__(self, other):
        assert self.name == other.name, "You can not multiply elements of different groups."
        M = np.dot(self.M, other.M)
        return LieGroupElement(name=self.name, data=M)

    def __eq__(self, other):
        return np.all(self.x == other.x) and np.all(self.R == other.R)

    def __str__(self):
        return "({}{})".format(str(self.x), str(self.R))


class LieGroup(object):
    def __init__(self, name):
        self.name = name

    @staticmethod
    def lie_algebra_element(v):
        tilde = np.array([[0, -v[2], v[1]],
                          [v[2], 0, -v[0]],
                          [-v[1], v[0], 0]])

        return tilde

    @staticmethod
    def exponential_map(v):
        pass

    @staticmethod
    def tangent_operator(v):
        pass

    def new_element(self, data):
        return LieGroupElement(name=self.name, data=data)


class SO3(LieGroup):
    def __init__(self, name):
        super(SO3, self).__init__(name=name)

    @staticmethod
    def exponential_map(v):
        ang = np.linalg.norm(v)
        tilde_v = SO3xR3.lie_algebra_element(v)
        # Rodrigues formula
        E = np.eye(3) + np.sin(ang) / ang * tilde_v
        E += (1 - np.cos(ang)) / ang ** 2 * tilde_v @ tilde_v

        return LieGroupElement(name="so3", data=E)

    @staticmethod
    def tangent_operator(v):
        ang = np.linalg.norm(v)
        tilde_v = SO3xR3.lie_algebra_element(v)
        T = np.eye(3) + (np.cos(ang) - 1) / (ang ** 2) * tilde_v
        T += (1 - np.sin(ang) / ang) / ang ** 2 * tilde_v @ tilde_v

        return T


class SO3xR3(LieGroup):
    def __init__(self, name):
        super(SO3xR3, self).__init__(name=name)

    @staticmethod
    def exponential_map(v):
        x = v[:3]
        phi = v[3:]
        E_so3 = SO3.exponential_map(phi)

        E_so3xr3 = np.eye(v.shape[0] + 1)
        E_so3xr3[:3, :3] = E_so3.R
        E_so3xr3[-4:-1, -1] = x

        return LieGroupElement(name="so3xr3", data=E_so3xr3)

    @staticmethod
    def tangent_operator(v):
        omega = v[3:]
        T_so3 = SO3.exponential_map(omega)

        T_so3xr3 = np.eye(v.shape[0])
        T_so3xr3[3:, 3:] = T_so3.R

        return T_so3xr3


class SE3(LieGroup):
    def __init__(self, name):
        super(SE3, self).__init__(name=name)

    def exponential_map(self, v):
        U = v[:3]
        omega = v[3:]
        E = SO3.exponential_map(omega).R
        T = SO3.tangent_operator(omega)
        E_se3 = np.eye(4)
        E_se3[:3, :3] = E
        E_se3[-4:-1, -1] = T.T @ U

        return LieGroupElement(name="se3", data=E_se3)

    @staticmethod
    def tangent_operator(v):
        u = v[:3]
        omega = v[3:]
        ang = np.linalg.norm(omega)
        tilde_omega = SE3.lie_algebra_element(omega)
        tilde_u = SE3.lie_algebra_element(omega)
        T = SO3.tangent_operator(omega)

        T_se3 = np.kron(np.eye(2), T)
        a = 2 / ang * np.sin(ang / 2) * np.cos(ang / 2)
        b = (2 * np.sin(ang / 2) / ang) ** 2
        C = 0.5 * (1 - b) * tilde_u + (1 - a) / ang ** 2 * (tilde_u @ tilde_omega + tilde_omega @ tilde_u)
        C += -(a - b) / ang ** 2 * (tilde_omega.T @ u) @ tilde_omega
        C += 1 / ang ** 2 * (0.5 * b - 3 * (1 - a) / ang ** 2) * (tilde_omega.T @ u) @ (tilde_omega @ tilde_omega)

        T_se3[:3, 3:] = C - 0.5 * tilde_u

        return T_se3


lie_groups = {"so3xr3": SO3xR3, "se3": SE3, "so3": SO3}

if __name__ == '__main__':
    R = np.eye(3)
    x = np.array([1, 2, 3])
    M = np.eye(7)
    M[:3, :3] = R
    M[-4:-1, -1] = x

    lie_group = SO3xR3(name="so3xr3")
    q1 = lie_group.new_element(data=(x, R))
    q2 = lie_group.new_element(data=M)

    print(q1 == q2)
    print(q1)
    exp = lie_group.exponential_map
    e = exp(np.ones(6))
    print(q1 @ e)
