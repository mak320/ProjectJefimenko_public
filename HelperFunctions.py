import numpy as np
import numba as nb 

"""
If we notice weird numerical errors try to debug thru disabling fastmath
"""

@nb.njit()
def gamma(v, c):
    """
    :param v: velocity (N, 3) shaped array
    :param c: speed of light scalar
    :return: The Lorentz factor
    """
    vx = v[:, 0:1]
    vy = v[:, 1:2]
    vz = v[:, 2:3]
    v_mag = np.sqrt(vx ** 2 + vy ** 2 + vz ** 2)
    return 1 / np.sqrt(1 - (v_mag ** 2) / (c ** 2))

@nb.njit()
def split_cartesian_comps(s):
    """
    Splits commonly occurring (N, 3) arrays where each colum stands for a Cartesian direction into 3 arrays along each
    column (Cartesian direction)
    :param s:
    :return:
    """
    sx = s[:, 0:1]
    sy = s[:, 1:2]
    sz = s[:, 2:3]

    return sx, sy, sz

@nb.njit()
def u2v(u, c):
    """
    Converts normalized momentum to velocity
    :param u: normalized momentum, array of shape = (N, 3)
    :return: velocity, array of shape = (N, 3)
    """

    ux, uy, uz = split_cartesian_comps(u)

    u_magn_sq = ux ** 2 + uy ** 2 + uz ** 2

    vx = ux / np.sqrt(1 + u_magn_sq / c ** 2)
    vy = uy / np.sqrt(1 + u_magn_sq / c ** 2)
    vz = uz / np.sqrt(1 + u_magn_sq / c ** 2)

    return np.hstack((vx, vy, vz))

nb.njit(fastmath = True)
def F2a(u, F, c, mass):
    """
    Converts force to acceleration
    :param u: normalized momentum, array of shape = (N, 3)
    :param F: force,  array of shape = (N, 3)
    :return: acceleration, array of shape = (N, 3)
    """

    v_vec = u2v(u, c)
    
    vx, vy, vz = split_cartesian_comps(v_vec)

    Fx, Fy, Fz = split_cartesian_comps(F)

    v_DOT_F = vx * Fx + vy * Fy + vz * Fz
    gamma_loc = gamma(v_vec, c)

    """Construct relativistic accelerations from force, using inner products"""

    ax = (Fx - v_DOT_F * vx / c ** 2) / (gamma_loc * mass)
    ay = (Fy - v_DOT_F * vy / c ** 2) / (gamma_loc * mass)
    az = (Fz - v_DOT_F * vz / c ** 2) / (gamma_loc * mass)

    a_vec = np.hstack((ax, ay, az))
    return a_vec




