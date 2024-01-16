from scipy.optimize import curve_fit

import matplotlib.pyplot as plt
import numpy as np

from NbodySimEM import SimClass
from scipy.optimize import curve_fit

import sys

sys.path.append('../ProjectJefimenko_public')

from NbodySimEM import *

q1 = 1
q2 = 1

m1 = 20
m2 = 3
T = 10
k = 5
c = 5/3

q_arr = np.array([[q1],
                 [q2]])

m_arr = np.array([[m1],
                  [m2]])

pos0 = np.array([[0, 0, 0],
                 [5, 0, 0]])


vel0 = np.array([[0, 0, 0],
                 [0, 0, 0]])

acc0 = np.array([[0, 0, 0],
                 [0, 0, 0]])

internal_filter = np.array([[0, 0],
                            [1, 0]])

external_filter = np.array([[1],
                            [0]])

x2_0 = pos0[1][0]
v2_0 = 0

t_signal = x2_0 / c


E0 = 1
alp = (m1 * c ** 2) / (E0 * q1)

def x1(t, t0):
    return alp * (np.sqrt(1 + (c * (t + t0) / alp) ** 2) - 1)   # eq 18


def t_ret(t, x2, t0):
    return (x2 ** 2 + 2 * x2 * alp + c ** 2 *(t ** 2 - t0 ** 2) - 2 * c * t * (x2 + alp)) / (2 * c * (c * (t + t0) - x2 - alp))


def tau(t, x2, t0):
    tr = t_ret(t, x2, t0)
    return (np.sqrt(1 + (c * (t0 + tr) / alp) ** 2) - c * (t0 + tr) / alp) ** (-2)


def dudt(x2,t, t0):
    constants = (k * q1 * q2)/m2 #constants = (k * q1 * q2) /m2
    return constants / (x2 - x1(t_ret(t, x2, t0), t0)) ** 2 * tau(t, x2, t0)

def dxdt(u):
    return u / np.sqrt(1 + u **2 / c **2)


def y4(time):

    dt = np.diff(time)[0]

    x = x2_0
    u = 0

    ul = [u]
    xl = [x]

    c1 = c4 = 0.5 / (2 - 2 ** (1 / 3))
    c2 = c3 = 0.5 * (1 - 2 ** (1 / 3)) / (2 - 2 ** (1 / 3))
    d1 = d3 = 1 / (2 - 2 ** (1 / 3))
    d2 = - 2 ** (1 / 3) / (2 - 2 ** (1 / 3))

    for t in time[:-1]:

        '--------------------------------------------------------------------'

        x1_k = x + c1 * dxdt(u) * dt
        u1_k = u + d1 * dudt(x1_k,t+c1*dt,t_signal) * dt

        x2_k = x1_k + c2 * dxdt(u1_k) *dt
        u2_k = u1_k + d2 * dudt(x2_k,t + c1 * dt + c2 * dt,t_signal) *dt

        x3_k = x2_k + c3 * dxdt(u2_k) * dt
        u3_k = u2_k + d3 * dudt(x3_k,t + c1 *dt + c2 * dt + c3 * dt,t_signal) * dt

        x4_k = x3_k + c4 * dxdt(u3_k) * dt
        u4_k = u3_k

        x = x4_k

        u = u4_k
        ul.append(u)
        xl.append(x)


    xl = np.array(xl)
    ul = np.array(ul)

    return xl, ul




L2_last_x2 = []
L2_last_x1 = []
dts = []


for i in range(6):

    N_forward = 750 * 2**i

    dt = T/N_forward

    N_backward = round(t_signal/dt)

    mom_sim = SimClass(c, k, T, N_forward, N_backward, q_arr, m_arr,
                                    pos0, vel0, acc0, internal_filter=internal_filter, external_filter=external_filter,
                                    pusher="p-cycleRK2")
    mom_sim.run_simulation()
    sim_x1 = mom_sim.rx_tot[0]
    sim_x2 = mom_sim.rx_tot[1]

    t_arr_sim = mom_sim.total_time[:, 0, 0]
    x2_x0 = mom_sim.rx_tot[1,N_backward+1]
    x2_v0 = mom_sim.ux_tot[1,N_backward+1]
    u = np.array([[x2_v0,0,0]])
    x2_p0 = mom_sim.u2v(u)

    ## for ode

    t_arr = np.linspace(0, T, N_forward + 1)

    predicted_x2, predicted_p2 = y4(t_arr)

    """predicted_p2, predicted_x2 = predicted_p2[::upscale], predicted_x2[::upscale]"""

    predicted_x1 = x1(t_arr_sim, t_signal)

    predicted_x2 = np.concatenate((np.ones(N_backward) * x2_0, predicted_x2), axis=None)

    L2_x1 = np.abs(sim_x1-predicted_x1)/predicted_x1
    L2_x2 = np.abs(sim_x2-predicted_x2)/predicted_x2

    plt.plot(t_arr_sim,predicted_x2)
    plt.plot(t_arr_sim,sim_x2)

    L2_last_x1.append(L2_x1[-1])
    L2_last_x2.append(L2_x2[-1])
    dts.append(mom_sim.dt)


L2_last_x1 = np.array(L2_last_x1)
L2_last_x2 = np.array(L2_last_x2)
dts = np.array(dts)

def examp_func(x,a,b):
    return a*x**b

params, covs = curve_fit(examp_func,dts,L2_last_x2)
params1, covs1 = curve_fit(examp_func,dts,L2_last_x1)

examp_dts = np.linspace(min(dts),max(dts),1000)


fig, ax = plt.subplots()
ax.scatter(dts,L2_last_x1,color='green')
ax.scatter(dts,L2_last_x2,color='red')

ax.plot(examp_dts,examp_func(examp_dts,*params))
ax.plot(examp_dts,examp_func(examp_dts,*params1))
print(params)
print(params1)
plt.show()



