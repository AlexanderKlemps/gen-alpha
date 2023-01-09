__author__ = "Alexander Klemps"

import matplotlib
matplotlib.rcParams['text.usetex'] = True
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
from matplotlib.widgets import Slider

from model import HeavyTopModel, HeavyTorsionTopModel
from integrator import ScaledGeneralizedAlphaIntegrator
from cv2 import Rodrigues
from lie import LieGroupElement

if __name__ == '__main__':
    group_name = "so3"
    visualize = False
    model = HeavyTorsionTopModel(group_name=group_name, coeffs=(3.0, 5e2))
    model.J = np.diag([0.46875, 15.234375, 15.234375])
    model.M = model.J
    integrator = ScaledGeneralizedAlphaIntegrator(model=model, rho_inf=0.8, tols=(1e-8, 1e-8))
    R_x = Rodrigues(np.array([np.pi/2, 0, 0]))[0]

    h = 1e-4
    # R0 = np.eye(3)
    roll = -np.pi/4
    yaw = -0.9*np.pi/2
    R0 = Rodrigues(np.array([0, 0, yaw]))[0] @ Rodrigues(np.array([roll, 0, 0]))[0]
    #R0 = np.eye(3)
    #omega0 = 1e-1 * np.array([150, -4.61538, 0])
    omega0 = np.zeros(3)
    q0 = LieGroupElement(group_name, data=R0)
    q = q0
    # q, v, dv, lamb = integrator.get_initial_values(R0, omega0)
    lamb = np.empty(shape=(0,))
    dv = np.linalg.solve(model.J, -np.cross(omega0, model.J @ omega0) - model._external_force(q, omega0))
    v = omega0
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

    energies = [model.total_energy(q, v)]
    ans = [a.T @ model.M @ a]
    omegas = [omega0]
    for i in range(N):
        q, v, dv, a, lamb = integrator.solve_time_step(q, v, dv, a, lamb, h)
        e = model.total_energy(q, v)
        energies.append(e)
        omegas.append(v)
        print(e)
        gamma = integrator.parameters["GAMMA"]
        beta = integrator.parameters["BETA"]
        alpha_m = integrator.parameters["ALPHA_M"]
        factor = (1 - gamma)**2 + (alpha_m * gamma / (1 - alpha_m))**2 - 2 * (1-gamma) * alpha_m * gamma / (1 - alpha_m)
        ans.append(a.T @ model.M @ a)

        if visualize:
            x = R_x @ (q.R @ model.X)
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

    time = np.linspace(0, T, N+1)
    energies = np.array(energies)
    ans = np.array(ans)
    omegas = np.array(omegas)
    plt.plot(time, energies)
    #plt.plot(time, ans / np.max(ans))
    #plt.plot(time, ans/ np.max(ans))
    # for i in range(3):
    #     plt.plot(time, omegas[:, i])
    #
    plt.grid()
    plt.title(r'$\alpha_0 =$ {}°, k={}, d={}'.format(roll / np.pi * 180, model.torsion_ceoff, model.friction_coeff))
    plt.xlabel("t [s]")
    plt.ylabel(r'total energy [J]')

    plt.show()