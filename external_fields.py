"""
User can specify arbitary functions in caretsion coordinates and time to specify the external 
electric and magnetic fields used by the Relatstic EM SImulator
Returns:
Array [E, N]: A single array of electric and magnetic fields 
"""


import numpy as np


def E_x(x, y, z, t):
    """can be set to any arbitrary function in 'normal' python convention - array magic does not happen here,
    but in ext_field"""
    return 0.0


def E_y(x, y, z, t):
    return 0


def E_z(x, y, z, t):
    return 1


def B_x(x, y, z, t):
    """can be set to any arbitrary function in 'normal' python convention - array magic does not happen here,
    but in ext_field"""
    return 0


def B_y(x, y, z, t):
    return 0


def B_z(x, y, z, t):
    return 0


def ext_field(x, y, z, t=0):
    """The external field generator:
    Inputs:
    x,y,z = (N,1) shaped np arrays
    Returns: [E,B] where E, B has numpy shapes (N,3)
    """
    N_charges = x.shape[0]
    updim = np.ones((N_charges, 1))
    E = np.hstack((E_x(x, y, z, t) * updim, E_y(x, y, z, t)
                  * updim, E_z(x, y, z, t) * updim))
    B = np.hstack((B_x(x, y, z, t) * updim, B_y(x, y, z, t)
                  * updim, B_z(x, y, z, t) * updim))
    total = [E, B]
    return total
