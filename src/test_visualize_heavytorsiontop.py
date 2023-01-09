__author__ = "Alexander Klemps"

import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D

from model import HeavyTorsionTopModel
from integrator import ScaledGeneralizedAlphaIntegrator
from cv2 import Rodrigues


if __name__ == '__main__':
    group_name = "se3"
    visualize = True
    model = HeavyTorsionTopModel(group_name=group_name, coeffs=(0.0, 5e3))
    integrator = ScaledGeneralizedAlphaIntegrator(model=model, rho_inf=0.6, tols=(1e-8, 1e-8))
    R_x = Rodrigues(np.array([np.pi/2, 0, 0]))[0]

    h = 1e-3
    R0 = np.eye(3)
    omega0 = 1e-1 * np.array([150, -4.61538, 0])

    q, v, dv, lamb = integrator.get_initial_values(R0, omega0)
    a = dv

    T = 2
    N = int(T / h)

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
    for i in range(N):
        q, v, dv, a, lamb = integrator.solve_time_step(q, v, dv, a, lamb, h)
        print(model.total_energy(q, v))
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
