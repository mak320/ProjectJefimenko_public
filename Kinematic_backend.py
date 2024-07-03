import numpy as np
import numba as nb
from ID_backend import ID_backend
import matplotlib.pyplot as plt

@nb.njit(parallel = True, cache = True)
def lift_Indeces(smallest_index, kinematic_var, r_tot, ID_negative, ID_positive):
    N_obs = ID_positive.shape[0]
    N_source = ID_positive.shape[1]

    smallest_retarded_quant = np.zeros((N_obs,N_source,3))
    r_smallest = np.zeros((N_obs,N_source,3))

    r_pos = np.zeros((N_obs,N_source,3))
    r_neg = np.zeros((N_obs,N_source,3))

    pos_retarded_quant = np.zeros((N_obs,N_source,3))
    neg_retarded_quant = np.zeros((N_obs,N_source,3))


    for i in nb.prange(N_obs):
        for j in range(N_source):
            
            #shitty loop unrolling for the 3 cart directions 
            lindex = smallest_index[i,j] # local index
            smallest_retarded_quant[i,j,0] = kinematic_var[j,0,lindex]
            smallest_retarded_quant[i,j,1] = kinematic_var[j,1,lindex]
            smallest_retarded_quant[i,j,2] = kinematic_var[j,2,lindex]

            r_smallest[i,j,0] = r_tot[j,0,lindex]
            r_smallest[i,j,1] = r_tot[j,1,lindex]
            r_smallest[i,j,2] = r_tot[j,2,lindex]

            pindex = ID_positive[i,j] # positive index pairwise
            r_pos[i,j,0] = r_tot[j,0,pindex]
            r_pos[i,j,1] = r_tot[j,1,pindex]
            r_pos[i,j,2] = r_tot[j,2,pindex]

            pos_retarded_quant[i,j,0] = kinematic_var[j,0,pindex]
            pos_retarded_quant[i,j,1] = kinematic_var[j,1,pindex]
            pos_retarded_quant[i,j,2] = kinematic_var[j,2,pindex]

            nindex = ID_negative[i,j] # negative index pairwise
            r_neg[i,j,0] = r_tot[j,0,nindex]
            r_neg[i,j,1] = r_tot[j,1,nindex]
            r_neg[i,j,2] = r_tot[j,2,nindex]

            neg_retarded_quant[i,j,0] = kinematic_var[j,0,nindex]
            neg_retarded_quant[i,j,1] = kinematic_var[j,1,nindex]
            neg_retarded_quant[i,j,2] = kinematic_var[j,2,nindex]

    return smallest_retarded_quant, r_smallest, r_pos, r_neg, pos_retarded_quant, neg_retarded_quant



def Extract_RetKinVar(r_tot, r_current, kinematic_var, ID_negative, ID_positive, ID_sign, dt, c, alpha=0.0):

    """
    r_tot has a shape of (N_source, 3, N_t)
    r_current has a shape of (N_obs, 3, 1)
    """

    N_obs = ID_positive.shape[0]
    N_source = ID_positive.shape[1]
    N_T = kinematic_var.shape[-1]


    """The resulting arrays of the retarded kinematic positions will have a shape of 
    (N_obs, N_source, 3)"""

    """These represent along the axis as:
    0th: ...As seen by which observer...
    1th: ...Which source was at which positions...
    2th: ...Which component of the retarded quantity"""

    """The ID sign array implicitly stores which of the two quanties of omega are closer to 0:
    If ID_sign[i,j] > 0 we infer that that ID_negative associated time moment is closer to the true retarded time moment
    IF ID_sign[i,j] < 0 we infer that that ID_positive associated time moment is closer to the true retarded time moment"""

    """Thus the time index of smallest difference is:"""

    smallest_index = ((ID_sign+1)/2).astype(np.int32) * ID_negative - ((ID_sign-1)/2).astype(np.int32) * ID_positive

    smallest_retarded_quant, r_smallest, r_pos, r_neg, pos_retarded_quant, neg_retarded_quant = lift_Indeces(smallest_index, kinematic_var, r_tot, ID_negative, ID_positive) 

    #(N_obs, N_source, 3)
    r_current = np.swapaxes(r_current,1,2)
    R_aux = r_current - r_smallest # ok

    #(N_obs, N_soruce)
    time_differential = (alpha - smallest_index)*dt

    #(N_obs, N_soource, 3)
    quasi_veloc = (r_neg - r_pos) / dt

    """CHECK TIME SCALING"""
    optim = 'greedy'

    #(N_obs, N_source)
    R2 = np.einsum('ijk, ijk->ij', R_aux, R_aux, optimize=optim)
    #(N_obs, N_source)
    RW_b = np.einsum('ijk, ijk->ij', R_aux, quasi_veloc, optimize=optim)

    """tau minus t_k"""
    # (N_obs, N_source)
    tau_diff_denom = 2*RW_b - c**2 * (time_differential)
    """uber numba incomaptible"""
    #(N_obs, N_source)
    tau_diff = np.divide(R2 - c**2 * (time_differential)**2, tau_diff_denom, out=np.zeros_like(tau_diff_denom), where=tau_diff_denom!=0)
    #(N_obs, N_source, 1)
    tau_diff = tau_diff[:,:,np.newaxis]

    retarded_quant_deriv = (neg_retarded_quant - pos_retarded_quant) / dt
    improved_retarded_quant = smallest_retarded_quant + retarded_quant_deriv * tau_diff

    return improved_retarded_quant


lis = []
dts = np.linspace(0.00001,0.05,10)
for i in range(len(dts)):
    r_obs = np.array([0,0,0.0]).reshape((1,3,1))

    c = 1
    v0 = 0.5

    t = np.arange(0.0,-10,-dts[i])

    N_t = t.shape[0]
    print(N_t)

    dt = np.abs(np.diff(t)[0])


    x0 = 1
    r_past = np.zeros((2,3,N_t))
    r_past[0,0,:] = x0
    r_past[1,1,:] = v0*t+2.0
    deepest_index = (N_t - 1)*np.ones(dtype=np.uint32, shape=(1,2))


    U, V, D = ID_backend(r_obs,r_past,deepest_index,dt,c,0.0)
    imprved_r = Extract_RetKinVar(r_past,r_obs,r_past,U,V,D,dt,c,0.0)
    print(imprved_r.shape)
    y = imprved_r[0,1,1]
    lis.append(y)

print(lis)
lis = np.abs(np.asarray(lis)-4/3)

plt.plot(dts, lis,"o")
plt.show()
#
