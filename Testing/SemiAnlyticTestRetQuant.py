import numpy as np
import matplotlib.pyplot as plt

from scipy.optimize import curve_fit
import sim_class

q1 = 1
q2 = 1

m1 = 20
m2 = 3
T = 10
k = 5
c = 2

v1 = 1.8
v2 = .5

N_backward = 3

q_arr = np.array([[q1],
                 [q2]])

m_arr = np.array([[m1],
                  [m2]])

vel0 = np.array([[v1, 0, 0],
                 [v2/np.sqrt(2), v2/np.sqrt(2), 0]])

acc0 = np.array([[0, 0, 0],
                 [0, 0, 0]])

internal_filter = np.array([[0, 0],
                            [0, 0]])

external_filter = np.array([[0],
                            [0]])

def t_r(v1, v2, t):
    prefactor = - 1 / (2*(c**2 -v1**2))
    sqrt_expr = np.sqrt(2) * t * np.sqrt(2 * c**2 * v2**2 - 2 * np.sqrt(2) * c**2 * v1 * v2 + v1**2 * (2 * c**2 - v2**2))
    other = - 2 * c**2 * t + np.sqrt(2) * t * v1 * v2
    return prefactor * (sqrt_expr + other)

def analytic_x1_ret(v1, v2, t):
    return v1 * t_r(v1, v2, t)


error = []
dts = []

for i in range(5):
    N_forward = 750 * 2 ** i
    dt = T / N_forward

    pos0 = np.array([[- v1 * N_backward * dt, 0, 0],
                     [-v2 / np.sqrt(2) * N_backward * dt, -v2 / np.sqrt(2) * N_backward * dt, 0]])

    mom_sim = sim_class.r_simulator(c, k, T, N_forward, N_backward, q_arr, m_arr,
                                pos0, vel0, acc0, internal_filt=internal_filter, external_filt=external_filter,
                                engine="p-cycle")
    mom_sim.run_simulation()

    sim_time = mom_sim.sim_time
    mom_ret_rx = mom_sim.list_ret_rx

    prediction = analytic_x1_ret(v1, v2, sim_time)

    L2 = np.abs(mom_ret_rx- prediction)/ np.abs(prediction)

    last_few_L2_avg = np.average(L2[-10:])

    error.append(last_few_L2_avg)
    dts.append(mom_sim.dt)



def examp_func(x,a,b):
    return a*x**b


examp_dts = np.linspace(min(dts),max(dts),1000)

params, covs = curve_fit(examp_func,dts,error)
params1, covs1 = curve_fit(examp_func,dts,error)


plt.scatter(dts,error,color='green')
plt.scatter(dts,error,color='red')

plt.plot(examp_dts,examp_func(examp_dts,*params))
plt.plot(examp_dts,examp_func(examp_dts,*params1))
print(params)
print(params1)
plt.show()



