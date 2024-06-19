import numpy as np
import numba as nb


@nb.njit(fastmath=True, cache=True, parallel=True)
def ID_backend(r_current, r_past, previos_deepest_ID, dt, c):
    """This function generates an array that serves to index the kinemtic variables"""
    # rc == array shape - (3, N_obs, 1)
    # rp == array shape - (3, N_source, N_past_time)

    N_source = r_past.shape[0]
    N_obs = r_current.shape[0]
    N_t = r_past.shape[2]
    IDs_array_positive = np.empty(shape=(N_source, N_obs), dtype=np.uint32)
    IDs_array_negative = np.empty(shape=(N_source, N_obs), dtype=np.uint32)
    denom_sign = np.empty(shape=(N_source, N_obs))

    prev_omega_sign = -1
    for i in nb.prange(N_source):
        for j in range(N_obs):
            deepest_index = previos_deepest_ID[i, j]
            for t_idx in range(deepest_index, -1, -1):

                spatial_sep = np.linalg.norm(r_current[j, :, 0]-r_past[i, :, t_idx])

                temporal_sep = dt*c*t_idx
                omega = spatial_sep - temporal_sep

                if np.sign(omega) != prev_omega_sign:
                    IDs_array_negative[i, j] = t_idx + 1
                    IDs_array_positive[i, j] = t_idx
                    break
                prev_omega_sign = np.sign(omega)

    return IDs_array_negative, IDs_array_positive
