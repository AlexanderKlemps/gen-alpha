__author__ = "Alexander Klemps"

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from model import HeavyTopModel
from integrator import ScaledGeneralizedAlphaIntegrator
from cv2 import Rodrigues

def roll_pitch_yaw(R):
    alpha = np.arctan2(R[1, 0], R[0, 0])
    beta = np.arctan2(-R[-1, 0], np.sqrt(R[0, 0] ** 2 + R[1, 0] ** 2))
    gamma = np.arctan2(R[2, 1], R[2, 2])

    return alpha, beta, gamma


def cart2sph(x, y, z):
    hxy = np.hypot(x, y)
    r = np.hypot(hxy, z)
    el = np.arctan2(z, hxy)
    az = np.arctan2(y, x)
    return r, el, az


if __name__ == '__main__':
    group_name = "se3"
    visualize = True
    model = HeavyTopModel(group_name=group_name)
    integrator = ScaledGeneralizedAlphaIntegrator(model=model, rho_inf=0.9, tols=(1e-8, 1e-8))
    R_x = Rodrigues(np.array([np.pi / 2, 0, 0]))[0]
    R_z = Rodrigues(np.array([0, 0, np.pi / 2]))[0]

    h = 2e-2
    T = 20

    R0 = np.eye(3)
    omega0 = 1e-1 * np.array([150, -4.61538, 0])
    q, v, dv, lamb = integrator.get_initial_values(R0, omega0)
    a = dv

    if visualize:
        plt.ion()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.set_xlim3d([-1.5, 1.5])
        ax.set_xlabel('X')
        ax.set_ylim3d([-1.5, 1.5])
        ax.set_ylabel('Y')
        ax.set_zlim3d([-1.5, 1.5])
        ax.set_zlabel('Z')

        u = np.linspace(0, 2 * np.pi, 100)
        ys = 0.5 * np.cos(u)[..., None]
        xs = np.ones(ys.shape)
        zs = 0.5 * np.sin(u)[..., None]

        circle = np.hstack((xs, ys, zs))
        X2 = np.array([1, 0.5, 0])
    angles = []
    omegas = []
    xs = []

    N = int(T / h)
    for i in range(N):
        q, v, dv, a, lamb = integrator.solve_time_step(q, v, dv, a, lamb, h)
        e = model.total_energy(q, v)
        gamma = integrator.parameters["GAMMA"]
        omega = v[3:]
        e += 0.5 * h**2 * a.T @ model.M @ a
        #e += 0.5 *
        print(e)
        omegas.append(R_z @ R_x @ v[3:])
        xs.append((R_x @ q.x))

        if visualize:
            x = R_x @ q.x
            R = R_x @ q.R
            circle_rot = np.dot(R, circle.T).T
            point = ax.scatter([x[0]], [x[1]], [x[2]], color='r')
            line1, = ax.plot([0, x[0]], [0, x[1]], [0, x[2]], color='b')
            x2 = np.dot(R, X2)
            line2, = ax.plot([x2[0], x[0]], [x2[1], x[1]], [x2[2], x[2]], color='g')
            circ, = ax.plot(circle_rot[:, 0], circle_rot[:, 1], circle_rot[:, 2], color='g')

            plt.draw()
            plt.pause(0.0001)
            point.remove()
            line1.remove()
            line2.remove()
            circ.remove()

    omegas = np.array(omegas)
    angles = np.array(angles)
    time_line = np.linspace(0, T, N)
    xs = np.array(xs)

    plt.grid()
    plt.plot(time_line, xs[:, 2])
    plt.show()

    for i in range(omegas.shape[1]):
        plt.plot(time_line, omegas[:, i])

    plt.grid()
    plt.title("Heavy top with kinematic constraints")
    plt.xlabel("time t [s]")
    plt.ylabel("omega [rad/s]")
    plt.show()
