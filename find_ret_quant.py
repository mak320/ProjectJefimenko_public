from HelperFunctions import *

 def get_ret_kin_variables_while(self, r, j, alpha):
        """
        This function outputs the same variables as the get_ret_kin_variables, but at considerably higher efficiency.
        :param r: Current position of particles: (N,3) shaped
        :param j: Current index of time
        :param alpha: Current sub-index of time: modifies actual calling time st. it adds +alpha*dt
        :return: Correct, interpolated kinematic variables
        """

        current_time = self.sim_time[j] + alpha * self.dt

        """Formatting the retarded kinematic variables in the same the vectorised get_kin_variables"""
        ret_rx_arr = np.zeros((self.NParticles, self.NParticles))
        ret_ry_arr = np.zeros((self.NParticles, self.NParticles))
        ret_rz_arr = np.zeros((self.NParticles, self.NParticles))

        ret_ux_arr = np.zeros((self.NParticles, self.NParticles))
        ret_uy_arr = np.zeros((self.NParticles, self.NParticles))
        ret_uz_arr = np.zeros((self.NParticles, self.NParticles))

        ret_ax_arr = np.zeros((self.NParticles, self.NParticles))
        ret_ay_arr = np.zeros((self.NParticles, self.NParticles))
        ret_az_arr = np.zeros((self.NParticles, self.NParticles))

        """This part can be reformatted to higher efficiency, by possibly vectorising it"""

        for A in range(self.NParticles):
            for B in range(self.NParticles):
                if B != A:
                    current_rx, current_ry, current_rz = r[A,0], r[A, 1], r[A, 2]
                    mID = int(self.min_IDs[A, B])
                    elapsed_time_length = self.rx_tot.shape[1] - 1

                    def CausalFunction(retarded_index):
                        """
                        This is the quantity denoted f in the paper
                        :param retarded_index:
                        :return:
                        """
                        spatial = np.sqrt((current_rx - self.rx_tot[B, retarded_index]) ** 2
                                          + (current_ry -
                                             self.ry_tot[B, retarded_index]) ** 2
                                          + (current_rz - self.rz_tot[B, retarded_index]) ** 2)

                        temporal = self.c * \
                            (current_time -
                             self.total_time[retarded_index, 0, 0])

                        return spatial - temporal



                    """
                    Cycle through casual quantities, 
                    until we find a smaller (more physical) one than the previously evaluated one, 
                    starting with the previous index. Only requires a few cycles of while loops typically.
                    """

                    closest_ID = mID

                    H1 = np.abs(CausalFunction(closest_ID))
                    try:
                        H2 = np.abs(CausalFunction(closest_ID + 1))
                        while H2 < H1:
                            try:
                                H1 = H2
                                closest_ID = closest_ID + 1
                                H2 = np.abs(CausalFunction(closest_ID + 1))
                            except:
                                """kills 'future' interactions"""
                                break
                    except:
                        id0 = mID

                    id0 = closest_ID
                    id1 = id0 - np.sign(CausalFunction(id0))

                    id1 = np.where(0 > id1, 0, id1).astype('int')
                    id1 = np.where(id1 > elapsed_time_length - 1,
                                   elapsed_time_length - 1, id1).astype("int")

                    """TODO: figure out where this should be called - I assume at earliest possible time, i.e. here."""

                    self.causal_corrector[A, B] = np.heaviside(
                        -np.sign(CausalFunction(self.min_IDs[A, B])), 1)

                    self.min_IDs[A, B] = max(min(id0, id1), 0)


                    """
                    Repeat same interpolation method, using the 'closest' quantity 
                    and the correctly identified neighbor ID.
                    """
                    closest_IDs_rx = self.rx_tot[B, id0]
                    closest_IDs_ry = self.ry_tot[B, id0]
                    closest_IDs_rz = self.rz_tot[B, id0]

                    closest_IDs_ux = self.ux_tot[B, id0]
                    closest_IDs_uy = self.uy_tot[B, id0]
                    closest_IDs_uz = self.uz_tot[B, id0]

                    closest_IDs_ax = self.ax_tot[B, id0]
                    closest_IDs_ay = self.ay_tot[B, id0]
                    closest_IDs_az = self.az_tot[B, id0]

                    IDs_time = self.total_time[id0, 0, 0]

                    neighbour_IDs_rx = self.rx_tot[B, id1]
                    neighbour_IDs_ry = self.ry_tot[B, id1]
                    neighbour_IDs_rz = self.rz_tot[B, id1]

                    neighbour_IDs_ux = self.ux_tot[B, id1]
                    neighbour_IDs_uy = self.uy_tot[B, id1]
                    neighbour_IDs_uz = self.uz_tot[B, id1]

                    neighbour_IDs_ax = self.ax_tot[B, id1]
                    neighbour_IDs_ay = self.ay_tot[B, id1]
                    neighbour_IDs_az = self.az_tot[B, id1]

                    """ Linear Interpolation of kinematic variables """

                    R2 = (current_rx - closest_IDs_rx) ** 2 + \
                         (current_ry - closest_IDs_ry) ** 2 + \
                         (current_rz - closest_IDs_rz) ** 2

                    plus_minus = -np.sign(CausalFunction(id0))

                    R_dot_wB = plus_minus / self.dt * (
                        (current_rx - closest_IDs_rx) * (neighbour_IDs_rx - closest_IDs_rx) +
                        (current_ry - closest_IDs_ry) * (neighbour_IDs_ry - closest_IDs_ry) +
                        (current_rz - closest_IDs_rz) * (neighbour_IDs_rz - closest_IDs_rz))

                    time_diff = current_time - IDs_time

                    numer_interp = R2 - (self.c ** 2 * time_diff ** 2)

                    denom_interp = 2 * (R_dot_wB - self.c ** 2 * time_diff)

                    dt_order_interp_quant = numer_interp * np.divide(1.0, denom_interp, out=np.zeros_like(denom_interp),
                                                                     where=denom_interp != 0.0)  # should be order h = order dt

                    """
                    Use same linear interpolation method. 
                    """

                    interp_correction = dt_order_interp_quant * \
                        plus_minus / self.dt  # should be order 1

                    ret_rx = closest_IDs_rx + \
                        ((neighbour_IDs_rx - closest_IDs_rx) * interp_correction)
                    ret_ry = closest_IDs_ry + \
                        ((neighbour_IDs_ry - closest_IDs_ry) * interp_correction)
                    ret_rz = closest_IDs_rz + \
                        ((neighbour_IDs_rz - closest_IDs_rz) * interp_correction)

                    ret_ux = closest_IDs_ux + \
                        ((neighbour_IDs_ux - closest_IDs_ux) * interp_correction)
                    ret_uy = closest_IDs_uy + \
                        ((neighbour_IDs_uy - closest_IDs_uy) * interp_correction)
                    ret_uz = closest_IDs_uz + \
                        ((neighbour_IDs_uz - closest_IDs_uz) * interp_correction)

                    ret_ax = closest_IDs_ax + \
                        ((neighbour_IDs_ax - closest_IDs_ax) * interp_correction)
                    ret_ay = closest_IDs_ay + \
                        ((neighbour_IDs_ay - closest_IDs_ay) * interp_correction)
                    ret_az = closest_IDs_az + \
                        ((neighbour_IDs_az - closest_IDs_az) * interp_correction)

                    ret_rx_arr[A, B] = ret_rx
                    ret_ry_arr[A, B] = ret_ry
                    ret_rz_arr[A, B] = ret_rz

                    ret_ux_arr[A, B] = ret_ux
                    ret_uy_arr[A, B] = ret_uy
                    ret_uz_arr[A, B] = ret_uz

                    ret_ax_arr[A, B] = ret_ax
                    ret_ay_arr[A, B] = ret_ay
                    ret_az_arr[A, B] = ret_az

        return ret_rx_arr, ret_ry_arr, ret_rz_arr, \
            ret_ux_arr, ret_uy_arr, ret_uz_arr, \
            ret_ax_arr, ret_ay_arr, ret_az_arr

@nb.njit()
def get_ret_kin_var(PastTimeArray, ret_pos, ret_vel, min_IDs,c):
    """
    PastTimeArray:: array of past times. Last element must be equal to current time, tn. Length of Nt.
    ret_pos: 3 x N x Nt array of past positions
    ret_val = 3 x N x Nt array of past velocities
    """
    current_time = PastTimeArray[-1]

    """Formatting the retarded kinematic variables in the same the vectorised get_kin_variables"""

    NParticles = ret_pos.shape[0]

    ret_rx_arr = np.zeros((NParticles, NParticles))
    ret_ry_arr = np.zeros((NParticles, NParticles))
    ret_rz_arr = np.zeros((NParticles, NParticles))

    ret_ux_arr = np.zeros((NParticles, NParticles))
    ret_uy_arr = np.zeros((NParticles, NParticles))
    ret_uz_arr = np.zeros((NParticles, NParticles))

    ret_ax_arr = np.zeros((NParticles, NParticles))
    ret_ay_arr = np.zeros((NParticles, NParticles))
    ret_az_arr = np.zeros((NParticles, NParticles))
    

    for A in range(NParticles):
        for B in range(NParticles):
            if A!=B:
                current_x, current_y, current_z = ret_pos[A,0,-1], ret_pos[A,1,-1], ret_pos[A,2,-1]
                local_minID = min_IDs[A, B]

                def f_AB(retarded_index):
                    """
                    This is the quantity denoted f in the paper
                    :param retarded_index:
                    :return:
                    """
                    spatial = np.sqrt((current_x - ret_pos[B, 0, retarded_index]) ** 2
                                        + (current_y -ret_pos[B, 1, retarded_index]) ** 2
                                        + (current_z - ret_pos[B, 2, retarded_index]) ** 2)

                    temporal = c * \
                        (current_time -
                            PastTimeArray[retarded_index])

                    return spatial - temporal
            

    
    
    
    



