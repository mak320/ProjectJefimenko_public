import numpy as np
import ID_backend

def Extract_RetKinVar(r_tot, r_current, kinematic_var, ID_negative, ID_positive, ID_sign, dt, c, alpha=0.0):

    """r_current is assumed to (3, N_obs, 1) shaped"""

    N_obs = ID_positive.shape[0]
    N_source = ID_positive.shape[1]
    N_T = kinematic_var.shape[-1]

    x, y = np.meshgrid(range(3), range(N_source))

    
    x = x[:, :, np.newaxis].repeat(N_obs, axis=-1)
    y = y[:, :, np.newaxis].repeat(N_obs, axis=-1)

    ID_positive = ID_positive.T[:, np.newaxis, :].repeat(3, axis=1)
    ID_negative = ID_negative.T[:, np.newaxis, :].repeat(3, axis=1)

    """The resulting arrays of the retarded kinematic positions will have a shape of 
    (N_source, 3, N_obs)"""

    """These represent along the axis as:
    0th: ...Which source was at which positions
    1th: ...Which component of the retarded quantity
    2th: ...As seen by which observer"""


    """The ID sign array implicitly stores which of the two quanties of omega are closer to 0:
    If ID_sign[i,j] > 0 we infer that that ID_negative associated time moment is closer to the true retarded time moment
    IF ID_sign[i,j] < 0 we infer that that ID_positive associated time moment is closer to the true retarded time moment"""

    ID_sign_expanded = ID_sign.T[:, np.newaxis, :]
    
    r_current_expanded = np.swapaxes(np.swapaxes(r_current,0,1),0,2)

    """Thus the time index of smallest difference is:"""

    smallest_index = ((ID_sign_expanded+1)/2).astype(np.uint32) * ID_negative - ((ID_sign_expanded-1)/2).astype(np.uint32) * ID_positive

    R_aux = r_current_expanded - r_tot[y,x,smallest_index]

    time_differential = (alpha - smallest_index[:,0,:])*dt

    quasi_veloc = (r_tot[y,x,ID_positive] - r_tot[y,x,ID_positive]) / dt

    """CHECK TIME SCALING"""
    optim = 'greedy'

    R2 = np.einsum('ijk, ijk->ik', R_aux, R_aux, optimize=optim)

    RW_b = np.einsum('ijk, ijk->ik', R_aux, quasi_veloc, optimize=optim)

    """tau minus t_k"""
    tau_diff_denom = 2*RW_b - c**2 * (time_differential)
    """uber numba incomaptible"""
    tau_diff = np.divide(R2 - c**2 * (time_differential)**2, tau_diff_denom, out=np.zeros_like(tau_diff_denom), where=tau_diff_denom!=0)

    tau_diff = tau_diff[:,np.newaxis,:]

    retarded_quant_deriv = (kinematic_var[y,x,ID_positive] - kinematic_var[y,x,ID_negative]) / dt
    improved_retarded_quant = kinematic_var[y,x,smallest_index] + retarded_quant_deriv * tau_diff

    return improved_retarded_quant


r_tot = kinematic_var = np.ones((10, 3, 100))
ID_pos = np.ones((20, 10), dtype=np.uint32)

ID_neg = np.ones((20, 10), dtype=np.uint32)

ID_sign = np.ones((20, 10), dtype=np.uint32)


Nobs = 20 

r_curr = np.random.randn(Nobs*3).reshape(3, Nobs, 1)

print(Extract_RetKinVar(r_tot, r_curr, kinematic_var, ID_neg, ID_pos, ID_sign, 1.0, 0.1).shape)
