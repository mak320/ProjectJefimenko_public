"""
Main script implementing the relativistic N-body solver algorithm for electromagnetic interactions through Liénard-Wiechert fields 

Authors: Mate Koszta, Bendeguz Szabo, Tamas Vaszary
This project licesed under the terms of the MIT license.
"""


import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt
import matplotlib.animation as ani

from external_fields import ext_field

np.set_printoptions(linewidth=150, precision=5)

"""Helper functions"""


def closest(a, arr):
    """
    :param a: scalar value
    :param arr: numpy array
    :return: finds the index of the value contained in arr that is closest to a.
    """
    idx = np.abs(arr - a).argmin(0)
    return idx


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


class SimClass:
    def __init__(self, c, K, T, NT, NP, charge, mass, pos0, mom0, acc0, collision_radius=None,
                 external_field=ext_field, internal_filter=1, external_filter=1, pusher="p-cycleRK2"):
        """
        In the following N is the number of particles

        :param c (float): speed of information propagation (speed of light)
        :param K (float): Coupling Constant (Coulomb constant = 1/(4 * pi * epsilon))
        :param T (float): Total duration of the simulation, not including the past trajectory generation
        :param NT (float): Forward Time-Steps Number of time-steps
        :param NP (float): Backward Time-Steps Total duration of the past trajectory generation
        :param charge (array, shape = (N, 1)): Charge array
        :param mass  (array, shape = (N, 1)): Mass array
        :param pos0 (array, shape = (N, 3)): Initial positions of all particles in Cartesian coordinates,
        each row representing one particle
        :param mom0 (array, shape = (N, 3)): Initial momentum of all particles in Cartesian coordinates,
        each row representing one particle
        :param acc0 (array, shape = (N, 3)): Initial acceleration of all particles in Cartesian coordinates,
        each row representing one particle
        :param external_field (func):  A function that returns the external field in the format [E, B] (func)
        :param internal_filter (array, shape = (N, N)): Filter array of 0s and 1s that limit particle-particle
        interactions
        if internal_filter[i, j] = 1 ==> particles i and j interact,
        otherwise internal_filter[i, j] = 0 ==> particles i and j do NOT interact
        :param external_filter ((array, shape = (N, 1))): Filter array of 0s and 1s that limit which particles interact
        with the external fields
         if internal_filter[i] = 1 ==> particles i interacts with external fields,
        otherwise internal_filter[i] = 0 ==> particles i does NOT interacts with external fields
        :param pusher: The choice of time integrator
        """

        """Physical Constants"""
        self.c = c
        self.K = K

        """Particle Parameters"""
        self.charge = charge
        self.mass = mass
        self.NParticles = self.mass.shape[0]

        """Dynamical Variables"""
        self.pos0 = pos0
        self.mom0 = mom0
        self.acc0 = acc0

        self.r = pos0.copy()
        self.u = mom0.copy() / self.mass
        self.a = acc0.copy()

        """Filter Matrices and Correctors"""
        self.internal_filter = internal_filter
        self.external_filter = external_filter

        self.corrector_self_interaction = 1 - np.eye(self.NParticles)
        self.causal_corrector = np.ones((self.NParticles, self.NParticles))

        """Time properties properties"""
        self.deepest_scan = 0
        self.NT = NT
        self.NP = NP
        self.T = T
        self.dt = self.T / self.NT

        self.time_length = max(self.NP - self.deepest_scan + 1, 3)

        """External Field"""
        self.external_field = external_field

        """Collision parameters"""
        self.collision_radius = collision_radius

        """Default assignment for total arrays"""
        self.rx_tot = None
        self.ry_tot = None
        self.rz_tot = None

        self.ux_tot = None
        self.uy_tot = None
        self.uz_tot = None

        self.ax_tot = None
        self.ay_tot = None
        self.az_tot = None

        """Default assignment for time arrays"""
        self.total_time = None
        self.sim_time = None
        self.past_time = None

        """Plotting"""
        self.colors = []

        for i in range(self.NParticles):
            if charge[i] > 0.0:
                self.colors.append('red')
            elif charge[i] < 0.0:
                self.colors.append('blue')
            else:
                self.colors.append('gold')

        """"Particle Pusher = Time Integrator"""
        self.pusher = pusher

        valid_engines = {"p-cycleRK2"}

        if not (self.pusher in valid_engines):
            raise ValueError("Invalid engine specified")

        """While loop implementation"""
        self.min_IDs = np.zeros(
            (self.NParticles, self.NParticles), dtype='int')

        """Testing"""

    """Conversion Functions"""

    def u2v(self, u):
        """
        Converts (normalized) momentum to velocity
        :param u: (normalized) momentum, array of shape = (N, 3)
        :return: velocity, array of shape = (N, 3)
        """

        ux, uy, uz = split_cartesian_comps(u)

        u_magn_sq = ux ** 2 + uy ** 2 + uz ** 2

        vx = ux / np.sqrt(1 + u_magn_sq / self.c ** 2)
        vy = uy / np.sqrt(1 + u_magn_sq / self.c ** 2)
        vz = uz / np.sqrt(1 + u_magn_sq / self.c ** 2)

        return np.hstack((vx, vy, vz))

    def F2a(self, u, F):
        """
        Converts force to acceleration
        :param u: (normalized) momentum, array of shape = (N, 3)
        :param F: force,  array of shape = (N, 3)
        :return: acceleration, array of shape = (N, 3)
        """

        v_vec = self.u2v(u)
        vx, vy, vz = split_cartesian_comps(v_vec)

        Fx, Fy, Fz = split_cartesian_comps(F)

        v_DOT_F = vx * Fx + vy * Fy + vz * Fz
        gamma_loc = gamma(v_vec, self.c)

        """Construct relativistic accelerations from force, using inner products"""

        ax = (Fx - v_DOT_F * vx / self.c ** 2) / (gamma_loc * self.mass)
        ay = (Fy - v_DOT_F * vy / self.c ** 2) / (gamma_loc * self.mass)
        az = (Fz - v_DOT_F * vz / self.c ** 2) / (gamma_loc * self.mass)

        a_vec = np.hstack((ax, ay, az))
        return a_vec

    def AppendCycleStep(self):
        """
        Use after each cycle of the time integrator to save the updated
        [position, (normalised) momentum, acceleration] to the corresponding total arrays
        """
        loc_rx = self.r[:, 0:1]
        loc_ry = self.r[:, 1:2]
        loc_rz = self.r[:, 2:3]

        loc_ux = self.u[:, 0:1]
        loc_uy = self.u[:, 1:2]
        loc_uz = self.u[:, 2:3]

        loc_ax = self.a[:, 0:1]
        loc_ay = self.a[:, 1:2]
        loc_az = self.a[:, 2:3]

        self.rx_tot = np.hstack((self.rx_tot, loc_rx))
        self.ry_tot = np.hstack((self.ry_tot, loc_ry))
        self.rz_tot = np.hstack((self.rz_tot, loc_rz))

        self.ux_tot = np.hstack((self.ux_tot, loc_ux))
        self.uy_tot = np.hstack((self.uy_tot, loc_uy))
        self.uz_tot = np.hstack((self.uz_tot, loc_uz))

        self.ax_tot = np.hstack((self.ax_tot, loc_ax))
        self.ay_tot = np.hstack((self.ay_tot, loc_ay))
        self.az_tot = np.hstack((self.az_tot, loc_az))

    def unpack_current(self, s):
        """
        :param s: Any array of shape (N,3) such as the position or momentum arrays used in the simulation
        :return: The retarded index finding need the current position (r) in a very particular format, this function
        formats it as such
        """

        """Used to unpack present variables into correctly vectorized and formatted arrays"""

        sx = s[:, 0:1]
        sx = np.repeat(sx, self.NParticles, axis=1)
        sx = np.expand_dims(sx, axis=0)
        sx = np.repeat(sx, self.time_length, axis=0)

        sy = s[:, 1:2]
        sy = np.repeat(sy, self.NParticles, axis=1)
        sy = np.expand_dims(sy, axis=0)
        sy = np.repeat(sy, self.time_length, axis=0)

        sz = s[:, 2:3]
        sz = np.repeat(sz, self.NParticles, axis=1)
        sz = np.expand_dims(sz, axis=0)
        sz = np.repeat(sz, self.time_length, axis=0)

        return sx, sy, sz

    def unpack_past(self, s_tot):
        """
        :param s_tot: Any array of shape (N,3) such as the position or momentum arrays used in the simulation
        :return: The retarded index finding need the record of past positions (rx_tot, ry_tot, rz_tot) and the
        record of past momenta (px_tot, py_tot, pz_tot) in a very particular format, this function formats them
        as such
        """

        s_past = s_tot[:, self.deepest_scan::].T
        s_past = np.expand_dims(s_past, axis=1)
        s_past = np.repeat(s_past, self.NParticles, axis=1)
        return s_past

    def generate_past(self):
        """
        Creates a past of kinematic variables such that the system
        is propagated FORWARD from time self.NP*self.dt to -self.dt, at which point it terminates and
        proper, interacting integration can begin.
        The equations of motions for this integration cycle is ONLY using the external fields, i.e. the particles
        are non-interacting in this regime.
        :return: N/A. Sets self.past variables in accordance with a non-interacting Lorentz Force.
        """

        """
        Create integration regime times.
        """

        past_time = np.linspace(-self.NP * self.dt, -self.dt, self.NP)
        sim_time = np.linspace(0, self.T - self.dt, self.NT)
        end_instance = np.array([self.T])

        self.total_time = np.hstack((past_time, sim_time, end_instance))
        self.total_time = self.total_time.reshape(
            (self.total_time.shape[0], 1, 1))
        self.total_time = self.total_time.repeat(self.NParticles, 1)
        self.total_time = self.total_time.repeat(self.NParticles, 2)

        self.past_time = past_time

        self.sim_time = sim_time

        """Vectorize and initalize past kinematic baribles"""

        self.rx_tot = self.r[:, 0:1]
        self.ry_tot = self.r[:, 1:2]
        self.rz_tot = self.r[:, 2:3]

        self.ux_tot = self.u[:, 0:1]
        self.uy_tot = self.u[:, 1:2]
        self.uz_tot = self.u[:, 2:3]

        self.ax_tot = self.a[:, 0:1]
        self.ay_tot = self.a[:, 1:2]
        self.az_tot = self.a[:, 2:3]

        def NonInteractingLorentzForce(r, v, j):
            """
            The way used to generate trajectories in the past needed to calculate retarded quantities during sim time
            is just evolve the particles in the external fields with no particle-particle interactions
            :param r: position
            :param v: velocity
            :param j: time index (important for time varying fields)
            :return: The force due to external fields ONLY for each particle
            """

            vx, vy, vz = split_cartesian_comps(v)

            rx, ry, rz = split_cartesian_comps(r)

            E_ext, B_ext = self.external_field(rx, ry, rz, j * self.dt)

            E_ext_x, E_ext_y, E_ext_z = split_cartesian_comps(E_ext)
            B_ext_x, B_ext_y, B_ext_z = split_cartesian_comps(B_ext)

            """Electric Part of the Lorentz Force"""
            FE_x = self.external_filter * E_ext_x * self.charge
            FE_y = self.external_filter * E_ext_y * self.charge
            FE_z = self.external_filter * E_ext_z * self.charge

            """Magnetic Part of the Lorentz Force"""
            FB_x = self.external_filter * \
                (vy * B_ext_z - vz * B_ext_y) * self.charge
            FB_y = self.external_filter * \
                (vz * B_ext_x - vx * B_ext_z) * self.charge
            FB_z = self.external_filter * \
                (vx * B_ext_y - vy * B_ext_x) * self.charge

            Fx = FE_x + FB_x
            Fy = FE_y + FB_y
            Fz = FE_z + FB_z

            F_vec = np.hstack((Fx, Fy, Fz))

            return F_vec

        """Integration time loop, using RK2, 
        used to integrate to to the starting,
        location of teh integration time loop"""

        for n in tqdm(range(self.NP)):
            Fn = NonInteractingLorentzForce(
                r=self.r, v=self.u2v(self.u), j=-(self.NP - n))
            an = self.F2a(u=self.u, F=Fn)

            u_mid = self.u + (Fn / self.mass) * self.dt / 2
            v_mid = self.u2v(u_mid)
            r_mid = self.r + self.u2v(self.u) * self.dt / 2

            F_mid = NonInteractingLorentzForce(
                r=r_mid, v=v_mid, j=-(self.NP - n))

            self.u = self.u + (F_mid / self.mass) * self.dt
            self.r = self.r + v_mid * self.dt
            self.a = an

            self.AppendCycleStep()

    def get_ret_kin_variables(self, r, j, alpha):
        """
        1. Identifies 3D time index closest to satisfying the defining equation to the retarded time
        2. Identifies the neighboring time index, the true retarded time lies between the closest index and the
        neighboring index
        3. Identifies the true retarded time by linear interpolation ==> t_r
        4. Find the value of retarded kinematic variables given t_r

        :param r: current position
        :param j: Cycle step
        :param alpha: time shift alpha = 0.0 for integer step, and alpha = 0.5 mid-steps
        :return: the retarded kinematic variables :
        """
        self.time_length = max(self.NP - self.deepest_scan + j + 1, 3)

        current_rx, current_ry, current_rz = self.unpack_current(r)
        current_time = self.sim_time[j] + alpha * self.dt

        past_rx = self.unpack_past(self.rx_tot)
        past_ry = self.unpack_past(self.ry_tot)
        past_rz = self.unpack_past(self.rz_tot)

        past_ux = self.unpack_past(self.ux_tot)
        past_uy = self.unpack_past(self.uy_tot)
        past_uz = self.unpack_past(self.uz_tot)

        past_ax = self.unpack_past(self.ax_tot)
        past_ay = self.unpack_past(self.ay_tot)
        past_az = self.unpack_past(self.az_tot)

        scan_time_regime = self.total_time[self.deepest_scan: self.NP + j + 1, :, :]

        """Backstop -- third order is overkill"""
        past_ax[-1, :, :] = 3 * past_ax[0, :, :] - \
            3 * past_ax[1, :, :] + past_ax[2, :, :]
        past_ay[-1, :, :] = 3 * past_ay[0, :, :] - \
            3 * past_ay[1, :, :] + past_ay[2, :, :]
        past_az[-1, :, :] = 3 * past_az[0, :, :] - \
            3 * past_az[1, :, :] + past_az[2, :, :]

        """Construction of the Omega array (defining equation of retarded time)"""

        retarded_space_diff = np.sqrt((current_rx - past_rx) ** 2 +
                                      (current_ry - past_ry) ** 2 +
                                      (current_rz - past_rz) ** 2)

        Omega = retarded_space_diff - self.c * \
            (current_time - scan_time_regime)

        """Causal Corrector-sets interactions which would come for from outside the light cone of a particle to zero"""
        oldest_rx = self.rx_tot[:, 0:1].T
        oldest_rx = np.expand_dims(oldest_rx, axis=1)
        oldest_rx = np.repeat(oldest_rx, self.NParticles, axis=1)

        oldest_ry = self.ry_tot[:, 0:1].T
        oldest_ry = np.expand_dims(oldest_ry, axis=1)
        oldest_ry = np.repeat(oldest_ry, self.NParticles, axis=1)

        oldest_rz = self.rz_tot[:, 0:1].T
        oldest_rz = np.expand_dims(oldest_rz, axis=1)
        oldest_rz = np.repeat(oldest_rz, self.NParticles, axis=1)

        oldest_space_diff = np.sqrt((current_rx - oldest_rx) ** 2 +
                                    (current_ry - oldest_ry) ** 2 +
                                    (current_rz - oldest_rz) ** 2)

        oldest_time = self.total_time[0, :, :]

        causal_Omega = oldest_space_diff - \
            self.c * (current_time - oldest_time)

        oldest_time_omega_slice = causal_Omega[0, :, :]

        self.causal_corrector = np.heaviside(
            -np.sign(oldest_time_omega_slice), 1)

        """ The time IDs closest to satisfying the defining equation of t_r are contained in closest_IDs 

        The convention for IDs type and retarded quantities 
        --> First index corresponds to current dynamical variable particle indices
        --> Second index corresponds to past particle indices at the time of signal emission
        """

        closest_IDs = closest(0, np.abs(Omega))

        # helper indices index_x -- current time slice, index_y -- signal emission time slice
        index_x, index_y = np.indices(closest_IDs.shape)

        """ Finding the neighbour to closest_IDs, which brackets the true retarded time slice with closest_IDs """

        neighbour_IDs = closest_IDs - \
            np.sign(Omega[closest_IDs[:], index_x, index_y])

        neighbour_IDs = np.where(0 > neighbour_IDs, 0, neighbour_IDs)

        neighbour_IDs = np.where(neighbour_IDs > self.time_length - 1,
                                 self.time_length - 1, neighbour_IDs).astype("int")

        """
        closest_IDs_(kinematic quantity) means the past_(kinematic quantity) called at the closest_IDs time slice, 
        similarly neighbour_IDs_(kinematic quantity) is the past_(kinematic quantity) called at the neighbour_IDs.
        """

        closest_IDs_rx = past_rx[closest_IDs[:], index_x, index_y]
        closest_IDs_ry = past_ry[closest_IDs[:], index_x, index_y]
        closest_IDs_rz = past_rz[closest_IDs[:], index_x, index_y]

        closest_IDs_ux = past_ux[closest_IDs[:], index_x, index_y]
        closest_IDs_uy = past_uy[closest_IDs[:], index_x, index_y]
        closest_IDs_uz = past_uz[closest_IDs[:], index_x, index_y]

        closest_IDs_ax = past_ax[closest_IDs[:], index_x, index_y]
        closest_IDs_ay = past_ay[closest_IDs[:], index_x, index_y]
        closest_IDs_az = past_az[closest_IDs[:], index_x, index_y]


        """
        Repeat selection of past kinematic variables for neighboring variables, for future interpolation.
        """
        IDs_time = scan_time_regime[closest_IDs[:], index_x, index_y]

        neighbour_IDs_rx = past_rx[neighbour_IDs[:], index_x, index_y]
        neighbour_IDs_ry = past_ry[neighbour_IDs[:], index_x, index_y]
        neighbour_IDs_rz = past_rz[neighbour_IDs[:], index_x, index_y]

        neighbour_IDs_ux = past_ux[neighbour_IDs[:], index_x, index_y]
        neighbour_IDs_uy = past_uy[neighbour_IDs[:], index_x, index_y]
        neighbour_IDs_uz = past_uz[neighbour_IDs[:], index_x, index_y]

        neighbour_IDs_ax = past_ax[neighbour_IDs[:], index_x, index_y]
        neighbour_IDs_ay = past_ay[neighbour_IDs[:], index_x, index_y]
        neighbour_IDs_az = past_az[neighbour_IDs[:], index_x, index_y]

        """ Linear Interpolation of kinematic variables """

        R2 = (current_rx[0] - closest_IDs_rx) ** 2 + \
             (current_ry[0] - closest_IDs_ry) ** 2 + \
             (current_rz[0] - closest_IDs_rz) ** 2

        plus_minus = -np.sign(Omega[closest_IDs[:], index_x, index_y])

        """For dedicated derivation of this formulae, see paper draft."""

        R_dot_wB = plus_minus / self.dt * ((current_rx[0] - closest_IDs_rx) * (neighbour_IDs_rx - closest_IDs_rx) +
                                           (current_ry[0] - closest_IDs_ry) * (neighbour_IDs_ry - closest_IDs_ry) +
                                           (current_rz[0] - closest_IDs_rz) * (neighbour_IDs_rz - closest_IDs_rz))

        time_diff = current_time - IDs_time

        numer_interp = R2 - (self.c ** 2 * time_diff ** 2)

        denom_interp = 2 * (R_dot_wB - self.c ** 2 * time_diff)

        """Remove numerical artefacts"""

        dt_order_interp_quant = numer_interp * np.divide(1.0, denom_interp, out=np.zeros_like(denom_interp),
                                                         where=denom_interp != 0.0)  # should be order h = order dt

        """This variable can further be used to check for numerical instability."""

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

        self.deepest_scan = min(np.amin(closest_IDs), self.rx_tot.shape[1] - 3)

        """"""

        return ret_rx, ret_ry, ret_rz, ret_ux, ret_uy, ret_uz, ret_ax, ret_ay, ret_az

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
                    current_rx, current_ry, current_rz = r[A,
                                                           0], r[A, 1], r[A, 2]
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

    def EM_fields(self, r, j, alpha):
        """
        Electro-Magnetic field expressions based on Liénard–Wiechert fields
        https://en.wikipedia.org/wiki/Liénard–Wiechert_potential

        :param r: current position
        :param j: Cycle step
        :param alpha: variable time shift -- 0 if integer step, 0.5 if in half-integer step.
        :return: Electric and magnetic fields
        """

        ret_rx, ret_ry, ret_rz, \
            ret_ux, ret_uy, ret_uz, \
            ret_ax, ret_ay, ret_az = self.get_ret_kin_variables(
                r=r, j=j, alpha=alpha)

        # retarded velocity
        ret_u_magn_sq = ret_ux ** 2 + ret_uy ** 2 + ret_uz ** 2

        ret_vx = ret_ux / np.sqrt(1 + ret_u_magn_sq / self.c ** 2)
        ret_vy = ret_uy / np.sqrt(1 + ret_u_magn_sq / self.c ** 2)
        ret_vz = ret_uz / np.sqrt(1 + ret_u_magn_sq / self.c ** 2)

        ret_v_magn_sq = ret_vx ** 2 + ret_vy ** 2 + ret_vz ** 2

        current_rx, current_ry, current_rz = self.unpack_current(r)

        delta_rx = current_rx[0, :, :] - ret_rx
        delta_ry = current_ry[0, :, :] - ret_ry
        delta_rz = current_rz[0, :, :] - ret_rz

        delta_r = np.sqrt(delta_rx ** 2 + delta_ry ** 2 + delta_rz ** 2)
        inv_delta_r = np.divide(
            1.0, delta_r, out=np.zeros_like(delta_r), where=delta_r != 0)

        nu_x = (self.c * delta_rx * inv_delta_r) - ret_vx
        nu_y = (self.c * delta_ry * inv_delta_r) - ret_vy
        nu_z = (self.c * delta_rz * inv_delta_r) - ret_vz

        delta_r_DOT_ret_a = delta_rx * ret_ax + delta_ry * ret_ay + delta_rz * ret_az
        delta_r_DOT_nu = delta_rx * nu_x + delta_ry * nu_y + delta_rz * nu_z

        inv_delta_r_DOT_nu_cubed = np.divide(1, delta_r_DOT_nu ** 3,
                                             out=np.zeros_like(delta_r_DOT_nu),
                                             where=delta_r_DOT_nu != 0)

        prefactor = self.internal_filter * self.corrector_self_interaction * self.causal_corrector * \
            self.K * self.charge

        # prefactor = self.internal_filter * self.causal_corrector * \
        #             self.K * self.charge

        """Electric field"""
        E_x = prefactor * delta_r * inv_delta_r_DOT_nu_cubed * ((self.c ** 2 - ret_v_magn_sq) * nu_x +
                                                                delta_r_DOT_ret_a * nu_x - delta_r_DOT_nu * ret_ax)

        E_y = prefactor * delta_r * inv_delta_r_DOT_nu_cubed * ((self.c ** 2 - ret_v_magn_sq) * nu_y +
                                                                delta_r_DOT_ret_a * nu_y - delta_r_DOT_nu * ret_ay)

        E_z = prefactor * delta_r * inv_delta_r_DOT_nu_cubed * ((self.c ** 2 - ret_v_magn_sq) * nu_z +
                                                                delta_r_DOT_ret_a * nu_z - delta_r_DOT_nu * ret_az)

        """Magnetic field"""
        B_x = 1 / self.c * inv_delta_r * (delta_ry * E_z - delta_rz * E_y)

        B_y = 1 / self.c * inv_delta_r * (delta_rz * E_x - delta_rx * E_z)

        B_z = 1 / self.c * inv_delta_r * (delta_rx * E_y - delta_ry * E_x)

        return E_x, E_y, E_z, B_x, B_y, B_z

    """Regen Get_force resze"""

    def Lorentz_force(self, r, v, j, alpha):
        """
        :param r: current position
        :param v: current velocity
        :param j: time index
        :param alpha: variable time shift -- 0 if integer step, 0.5 if in half-integer step.
        :return: the Lorentz force calculated based on the electric and magnetic fields returned by EM_fields method
        """

        E_x, E_y, E_z, B_x, B_y, B_z = self.EM_fields(r=r, j=j, alpha=alpha)

        current_rx, current_ry, current_rz = split_cartesian_comps(r)
        current_vx, current_vy, current_vz = split_cartesian_comps(v)

        E_ext, B_ext = self.external_field(current_rx, current_ry, current_rz,
                                           j * self.dt + alpha * self.dt)

        E_ext_x, E_ext_y, E_ext_z = split_cartesian_comps(E_ext)
        B_ext_x, B_ext_y, B_ext_z = split_cartesian_comps(B_ext)

        F_x = E_x @ self.charge + (current_vy * B_z - current_vz * B_y) @ self.charge + \
            self.external_filter * self.charge * \
            (E_ext_x + (current_vy * B_ext_z - current_vz * B_ext_y))

        F_y = E_y @ self.charge + (current_vz * B_x - current_vx * B_z) @ self.charge + \
            self.external_filter * self.charge * \
            (E_ext_y + (current_vz * B_ext_x - current_vx * B_ext_z))

        F_z = E_z @ self.charge + (current_vx * B_y - current_vy * B_x) @ self.charge + \
            self.external_filter * self.charge * \
            (E_ext_z + (current_vx * B_ext_y - current_vy * B_ext_x))

        F_vec = np.hstack((F_x, F_y, F_z))

        return F_vec

    """Ezt kulon kell ellenorizni minden ami collision, befolyasolja a cyclet (particlepushert is) is"""

    def collision_condition(self, pos_new):

        rx_n = self.r[:, 0:1]
        ry_n = self.r[:, 1:2]
        rz_n = self.r[:, 2:3]

        Dx_tn = rx_n - rx_n.T
        Dy_tn = ry_n - ry_n.T
        Dz_tn = rz_n - rz_n.T

        rx_tnp1 = pos_new[:, 0:1]
        ry_tnp1 = pos_new[:, 1:2]
        rz_tnp1 = pos_new[:, 2:3]

        Dx_tnp1 = rx_tnp1 - rx_tnp1.T
        Dy_tnp1 = ry_tnp1 - ry_tnp1.T
        Dz_tnp1 = rz_tnp1 - rz_tnp1.T

        wx = (Dx_tnp1 - Dx_tn)
        wy = (Dy_tnp1 - Dy_tn)
        wz = (Dz_tnp1 - Dz_tn)

        DR_n_squared = Dx_tn ** 2 + Dy_tn ** 2 + Dz_tn ** 2

        w_squared = wx ** 2 + wy ** 2 + wz ** 2
        DR_dot_w = Dx_tn * wx + Dy_tn * wy + Dz_tn * wz

        det = DR_dot_w ** 2 - w_squared * \
            (DR_n_squared - self.collision_radius ** 2)

        """Conditions"""
        """ 1.Real solution -- Non-negative determinant """
        det_cond = det >= 0.0

        """ 2.Bounding the solution -- at least one of the solutions (+ or -) is between t_n and t_{n+1} """
        bound_plus_low = np.divide(-DR_dot_w + np.sqrt(det), w_squared, out=-np.ones_like(
            w_squared, dtype="float"), where=w_squared != 0.0) >= 0

        bound_plus_high = np.divide(-DR_dot_w + np.sqrt(det), w_squared, out=-np.ones_like(
            w_squared, dtype="float"), where=w_squared != 0.0) <= 1

        bound_minus_low = np.divide(-DR_dot_w - np.sqrt(det), w_squared, out=-np.ones_like(
            w_squared, dtype="float"), where=w_squared != 0.0) >= 0

        bound_minus_high = np.divide(-DR_dot_w - np.sqrt(det), w_squared, out=-np.ones_like(
            w_squared, dtype="float"), where=w_squared != 0.0) <= 1

        """ At least one of the real solutions is caught between t_n and t_{n+1} """
        collision_selector = np.where(
            det_cond & ((bound_plus_low & bound_plus_high) | (bound_minus_low & bound_minus_high)), 1, 0)

        collision_selector[np.diag_indices_from(collision_selector)] = 0

        return collision_selector

    def collision(self, pos_new, u):
        """
        :param pos_new: position at t_{n+1} were there to be no collisions
        :param u: normalised momentum u = p/m at t_{n}
        :return:
        """
        collision_condition = self.collision_condition(pos_new)
        collision_number = np.sum(collision_condition) / 2

        px, py, pz = split_cartesian_comps(u * self.mass)

        post_col_mom = np.hstack((px, py, pz))
        """why are the indices shifted by one here?"""
        for A in range(1, self.NParticles + 1, 1):
            for B in range(1, A + 1, 1):
                if collision_condition[A - 1, B - 1] == 1:
                    # szebb lenne a kod ha 0-nal kezdodik a range
                    pA_x = px[A - 1]
                    pA_y = py[A - 1]
                    pA_z = pz[A - 1]

                    pB_x = px[B - 1]
                    pB_y = py[B - 1]
                    pB_z = pz[B - 1]

                    p_comb_x = pA_x + pB_x
                    p_comb_y = pA_y + pB_y
                    p_comb_z = pA_z + pB_z

                    mA = self.mass[A - 1]
                    mB = self.mass[B - 1]

                    EA = np.sqrt((pA_x ** 2 + pA_y ** 2 + pA_z ** 2)
                                 * self.c ** 2 + mA ** 2 * self.c ** 4)
                    EB = np.sqrt((pB_x ** 2 + pB_y ** 2 + pB_z ** 2)
                                 * self.c ** 2 + mB ** 2 * self.c ** 4)

                    E_comb = EA + EB

                    v_bar_x = p_comb_x * self.c ** 2 / E_comb
                    v_bar_y = p_comb_y * self.c ** 2 / E_comb
                    v_bar_z = p_comb_z * self.c ** 2 / E_comb

                    v_bar_magn = np.sqrt(
                        v_bar_x ** 2 + v_bar_y ** 2 + v_bar_z ** 2)

                    norm_x = np.divide(v_bar_x, v_bar_magn, out=np.zeros_like(
                        v_bar_magn), where=v_bar_magn != 0)
                    norm_y = np.divide(v_bar_y, v_bar_magn, out=np.zeros_like(
                        v_bar_magn), where=v_bar_magn != 0)
                    norm_z = np.divide(v_bar_z, v_bar_magn, out=np.zeros_like(
                        v_bar_magn), where=v_bar_magn != 0)

                    gamma_bar = 1 / np.sqrt(1 - v_bar_magn ** 2 / self.c ** 2)

                    pA_DOT_norm = pA_x * norm_x + pA_y * norm_y + pA_z * norm_z

                    pA_prime_x = pA_x + (gamma_bar - 1) * pA_DOT_norm * norm_x - (
                        gamma_bar * EA * v_bar_x) / self.c ** 2
                    pA_prime_y = pA_y + (gamma_bar - 1) * pA_DOT_norm * norm_y - (
                        gamma_bar * EA * v_bar_y) / self.c ** 2
                    pA_prime_z = pA_z + (gamma_bar - 1) * pA_DOT_norm * norm_z - (
                        gamma_bar * EA * v_bar_z) / self.c ** 2

                    EA_prime = np.sqrt(
                        (pA_prime_x ** 2 + pA_prime_y ** 2 + pA_prime_z ** 2) * self.c ** 2 + mA ** 2 * self.c ** 4)
                    EB_prime = np.sqrt(
                        (pA_prime_x ** 2 + pA_prime_y ** 2 + pA_prime_z ** 2) * self.c ** 2 + mB ** 2 * self.c ** 4)

                    E_comb_prime = EA_prime + EB_prime

                    q_prime_magn = np.sqrt((E_comb_prime ** 2 - (mA ** 2 + mB ** 2) * self.c ** 4) ** 2 -
                                           4 * mA ** 2 * mB ** 2 * self.c ** 8) / (2 * E_comb_prime * self.c)

                    pA_prime_magn = np.sqrt(
                        pA_prime_x ** 2 + pA_prime_y ** 2 + pA_prime_z ** 2)

                    qA_prime_x = - q_prime_magn * pA_prime_x / pA_prime_magn
                    qA_prime_y = - q_prime_magn * pA_prime_y / pA_prime_magn
                    qA_prime_z = - q_prime_magn * pA_prime_z / pA_prime_magn

                    qB_prime_x = -qA_prime_x
                    qB_prime_y = -qA_prime_y
                    qB_prime_z = -qA_prime_z

                    """ Inverse Lorentz transform"""
                    qA_prime_DOT_norm = qA_prime_x * norm_x + \
                        qA_prime_y * norm_y + qA_prime_z * norm_z
                    qB_prime_DOT_norm = - qA_prime_DOT_norm

                    qA_x = qA_prime_x + (
                        gamma_bar - 1) * qA_prime_DOT_norm * norm_x + gamma_bar * EA_prime * v_bar_x / self.c ** 2
                    qA_y = qA_prime_y + (
                        gamma_bar - 1) * qA_prime_DOT_norm * norm_y + gamma_bar * EA_prime * v_bar_y / self.c ** 2
                    qA_z = qA_prime_z + (
                        gamma_bar - 1) * qA_prime_DOT_norm * norm_z + gamma_bar * EA_prime * v_bar_z / self.c ** 2

                    qB_x = qB_prime_x + (
                        gamma_bar - 1) * qB_prime_DOT_norm * norm_x + gamma_bar * EB_prime * v_bar_x / self.c ** 2
                    qB_y = qB_prime_y + (
                        gamma_bar - 1) * qB_prime_DOT_norm * norm_y + gamma_bar * EB_prime * v_bar_y / self.c ** 2
                    qB_z = qB_prime_z + (
                        gamma_bar - 1) * qB_prime_DOT_norm * norm_z + gamma_bar * EB_prime * v_bar_z / self.c ** 2

                    px[A - 1] = qA_x
                    py[A - 1] = qA_y
                    pz[A - 1] = qA_z

                    px[B - 1] = qB_x
                    py[B - 1] = qB_y
                    pz[B - 1] = qB_z

                    post_col_mom = np.hstack((px, py, pz))

        return collision_number, post_col_mom

    """Ez a cycle most jelenleg csak rk2 momentum van es csak momentum alpu cyclet akrunk kesobbiekben beepiteni alladora"""

    def p_cycleRK2(self):
        """
        (Normalised) Momentum based RK-2 time integrator (particle pusher)
        :return: the propagated positions of all particles based on the force that was calculated
        """

        if self.collision_radius is not None:
            for n in tqdm(range(self.NT)):

                # Trying to avid the scope problem
                r_np1 = self.r  # Initialize outside the loop
                u_np1 = self.u  # Initialize outside the loop
                an = self.a  # Initialize outside the loop

                # egyszer biztos fut a while loop
                col_num = 1
                while col_num > 0:
                    vn = self.u2v(self.u)
                    Fn = self.Lorentz_force(r=self.r, v=vn, j=n, alpha=0.0)
                    an = self.F2a(u=self.u, F=Fn)

                    u_mid = self.u + (Fn / self.mass) * self.dt / 2
                    v_mid = self.u2v(u_mid)
                    r_mid = self.r + self.u2v(self.u) * self.dt / 2

                    F_mid = self.Lorentz_force(
                        r=r_mid, v=v_mid, j=n, alpha=0.5)

                    """Push position and normalised momentum"""
                    r_np1 = self.r + v_mid * self.dt
                    u_np1 = self.u + (F_mid / self.mass) * self.dt

                    """Check for collisions that happened between t_n and t_{n+1}, update the momenta where 
                    appropriate"""
                    number_of_collisions, post_collision_p = self.collision(
                        pos_new=r_np1, u=self.u)

                    col_num = number_of_collisions
                    self.u = post_collision_p / self.mass

                self.r = r_np1
                self.u = u_np1
                self.a = an

                self.AppendCycleStep()

        else:
            for n in tqdm(range(self.NT)):
                vn = self.u2v(self.u)
                Fn = self.Lorentz_force(r=self.r, v=vn, j=n, alpha=0.0)
                an = self.F2a(u=self.u, F=Fn)

                u_mid = self.u + (Fn / self.mass) * self.dt / 2
                v_mid = self.u2v(u_mid)
                r_mid = self.r + self.u2v(self.u) * self.dt / 2

                F_mid = self.Lorentz_force(r=r_mid, v=v_mid, j=n, alpha=0.5)

                self.u = self.u + (F_mid / self.mass) * self.dt
                self.r = self.r + v_mid * self.dt
                self.a = an

                self.AppendCycleStep()

    def run_simulation(self):
        """Just executes both intergation regimes"""
        self.generate_past()
        self.p_cycleRK2()

    def visul(self, rate=10, norm=True, track=True, s1=10, s2=0.2, alpha_track=0.2, range=50,
              show=True):
        """
        :param rate: rate of frame display: higher value: more internal frames skipped
        :param norm: Whether to normalize acceleration vector displays or not.
        :param track: Whether to plot particle tracks.
        :param s1: Size of particles.
        :param s2: Size of tracks.
        :param alpha_track: Alpha value of tracks. Recommended to be around 0.2
        :param range: Range of display in all dimensions, 2x of value is the total display domain
        :param show: whether to show plot or not.
        :return:
        No return value. Make visualization visible within a separate window.
        """

        self.fig = plt.figure()
        self.ax = self.fig.add_subplot(111, projection='3d', proj_type='ortho')
        self.track = track
        self.norm = norm
        self.s1, self.s2 = s1, s2
        self.alpha_track = alpha_track

        self.ax.set(xlim=[-range, range],
                    ylim=[-range, range], zlim=[-range, range])

        self.fr = self.ax.scatter(self.rx_tot[:, 0], self.ry_tot[:, 0], self.rz_tot[:, 0], color=self.colors, s=s1,
                                  alpha=1)

        if self.track:
            self.frt = self.ax.scatter(self.rx_tot[:, 0], self.ry_tot[:, 0], self.rz_tot[:, 0], color='grey', s=s2,
                                       alpha=0.5)

        def updater(i):
            self.fr.remove()

            if self.track:
                self.frt.remove()

            self.fr = self.ax.scatter(self.rx_tot[:, i], self.ry_tot[:, i], self.rz_tot[:, i], color=self.colors, s=s1,
                                      alpha=1)
            if self.track:
                self.frt = self.ax.scatter(self.rx_tot[:, 0:i], self.ry_tot[:, 0:i], self.rz_tot[:, 0:i], color='grey',
                                           s=s2, alpha=alpha_track)

        self.length = self.total_time.shape[0]

        self.ani = ani.FuncAnimation(self.fig, updater, frames=np.arange(
            0, self.length, 1)[::rate], interval=1)

        if show:
            self.fig.show()


if __name__ == '__main__':

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
