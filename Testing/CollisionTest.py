"""
    Authors: M. A. Koszta, B. Szabo, T. A. Vaszary 

    This file is used to test the collision mechnism. 
"""

import sys

sys.path.append('../ProjectJefimenko_public')

from NbodySimEM import *

"""Test case"""

q1 = 1e-9
q2 = -1e-9

m1 = 1
m2 = 1
#
c = 1
k = 1
T = 3
N_forward = 1500
N_backward = 500

col_radius = 0.1

m_arr = np.array([[m1],
                  [m2]])

q_arr = np.array([[q1],
                  [q2]])

acc0 = np.array([[0.0, 0, 0],
                 [0, 0, 0]])

"""Head on collision test"""

# pos0HeadOn = np.array([[1, 0, 0],
#                  [-1, 0, 0]])
#
# mom0HeadOn = np.array([[-1, 0, 0],
#                  [1, 0, 0]])
#
# headOn = SimClass(c, k, T, N_forward, N_backward, q_arr, m_arr, pos0HeadOn, mom0HeadOn, acc0, collision_radius=0.1)
#
# headOn.run_simulation()
# headOn.visul()
#
# plt.show()
#
"""At an angle"""
v1 = 1
v2 = 2
x2 = 3
y2 = np.sqrt(2) * x2 / (2 * v2 + 1)

pos0Angle = np.array([[0, 0, 0],
                      [x2, y2, 0]])

vel0Angle = np.array([[v1 / np.sqrt(2), v1 / np.sqrt(2), 0],
                      [-v2, 0, 0]])

AtAngle = SimClass(c, k, T, N_forward, N_backward, q_arr, m_arr,
                   pos0Angle, vel0Angle * m_arr, acc0, collision_radius=0.1)

AtAngle.run_simulation()

AtAngle.visul()

plt.show()

"""3 Way collision"""

# dist = 1
# mommagn = 1
# m3 = 1
# q3 = 0
#
# m_arr = np.array([[m1],
#                   [m2],
#                   [m3]])
#
# q_arr = np.array([[q1],
#                   [q2],
#                   [q3]])
#
#
# acc0 = np.array([[0, 0, 0],
#                  [0, 0, 0],
#                  [0, 0, 0]])
#
# pos03Way = np.array([[dist, 0, 0],
#                      [-dist, 0, 0],
#                      [0, dist, 0]])
#
# mom03Way = np.array([[-mommagn, 0, 0],
#                      [mommagn, 0, 0],
#                      [0, -mommagn, 0]])
#
# ThreeWay = SimClass(c, k, T, N_forward, N_backward, q_arr, m_arr, pos03Way, mom03Way, acc0,
#                    collision_radius=0.1)
#
# ThreeWay.run_simulation()
# ThreeWay.visul()
#
# plt.show()