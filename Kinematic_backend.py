import numpy as np


import ID_backend


def Extract_RetKinVar(r_tot, kinematic_var, ID_negative, ID_positive, ID_sign):
    N_obs = ID_positive.shape[0]
    N_source = ID_positive.shape[1]
    N_T = kinematic_var.shape[-1]

    x, y = np.meshgrid(range(3), range(N_source))

    # print(x.shape, y.shape)
    x = x[:, :, np.newaxis].repeat(N_obs, axis=-1)
    y = y[:, :, np.newaxis].repeat(N_obs, axis=-1)

    ID_positive = ID_positive.T[:, np.newaxis, :].repeat(3, axis=1)

    # print(x.shape)
    # print(y.shape)
    # print(ID_positive.shape)

    r_pos = r_tot[y, x, ID_positive]

    return r_pos


r_tot = kinematic_var = np.ones((10, 3, 100))
ID_pos = np.ones((20, 10), dtype=np.uint32)
# shape = (20, 10)
# ID_pos = np.arange(np.prod(shape), dtype=np.uint32).reshape(shape)
ID_neg = np.ones((20, 10), dtype=np.uint32)


def Extract_RetKinVar_gpt(r_tot, kinematic_var, ID_negative, ID_positive, ID_sign):
    N_obs = ID_positive.shape[0]
    N_source = ID_positive.shape[1]
    N_T = kinematic_var.shape[-1]

    expanded_ID_pos = np.expand_dims(ID_positive, axis=0).repeat(N_source, axis=0)
    print(expanded_ID_pos.shape)
    r_pos = np.take_along_axis(r_tot, expanded_ID_pos, axis=2)

    return r_pos


print(Extract_RetKinVar(r_tot, kinematic_var, ID_neg, ID_pos, 0))
