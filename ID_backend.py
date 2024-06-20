import numpy as np
import numba as nb

# np.random.seed(2)


@nb.njit(fastmath=True, cache=True, parallel=True)
def ID_backend(r_current, r_past, previos_deepest_ID, dt, c):
    """This function generates an array that serves to index the kinemtic variables"""
    # rc == array shape - (3, N_obs, 1)
    # rp == array shape - (3, N_source, N_past_time)

    N_source = r_past.shape[0]
    N_obs = r_current.shape[0]
    N_t = r_past.shape[2]
    numerical_delta = 1e-16
    IDs_array_positive = np.empty(shape=(N_obs, N_source), dtype=np.uint32)
    IDs_array_negative = np.empty(shape=(N_obs, N_source), dtype=np.uint32)
    interp_slope_sign = np.empty(shape=(N_obs, N_source))

    prev_omega = -np.inf
    for i in nb.prange(N_obs):
        for j in range(N_source):
            deepest_index = previos_deepest_ID[i, j]

            for t_idx in range(deepest_index, -1, -1):

                spatial_sep = np.linalg.norm(r_current[i, :, 0]-r_past[j, :, t_idx])

                temporal_sep = dt*c*t_idx
                omega = spatial_sep - temporal_sep

                if np.sign(omega) != np.sign(prev_omega):
                    IDs_array_negative[i, j] = t_idx + 1
                    IDs_array_positive[i, j] = t_idx
                    interp_slope_sign[i, j] = np.sign(np.abs(omega) - np.abs(prev_omega) + numerical_delta)

                    break
                prev_omega = omega

    return IDs_array_negative, IDs_array_positive, interp_slope_sign


# c = 1
# t = np.linspace(0, 5, 20)
# x1 = 0.7 * t
# x2 = -0.6 * t

# pos_past = np.zeros((2, 3, 20))
# pos_past[0, 0, :] = x1
# pos_past[1, 0, :] = x2
# pos_current = np.random.randn(30).reshape((10, 3, 1))

# deepest_id = np.ones(dtype=np.uint32, shape=(10, 2)) * 19

# dt = np.diff(t)[0]

# U, V, D = ID_backend(pos_current, pos_past, deepest_id, dt, c)

# print(U, V, D)
