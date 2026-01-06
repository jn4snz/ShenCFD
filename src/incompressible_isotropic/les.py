# © 2025. Triad National Security, LLC. All rights reserved.

# This program was produced under U.S. Government contract 89233218CNA000001 for
# Los Alamos National Laboratory (LANL), which is operated by Triad National
# Security, LLC for the U.S. Department of Energy/National Nuclear Security
# Administration. All rights in the program are reserved by Triad National
# Security, LLC, and the U.S. Department of Energy/National Nuclear Security
# Administration. The Government is granted for itself and others acting on its
# behalf a nonexclusive, paid-up, irrevocable worldwide license in this material
# to reproduce, prepare. derivative works, distribute copies to the public,
# perform publicly and display publicly, and to permit others to do so.

from mpi4py import MPI  # must always be imported first

from copy import deepcopy
from math import pi, sqrt
import numpy as np
from shenfun import Function

from .dns import DNS
from ..fourier_analysis import fft3d
from ..file_io import h5FileIO
from ..maths import dot, smooth_bandpass
from ..statistics import allsum, allmax

comm = MPI.COMM_WORLD


class LES(DNS):

    # import all turbulence model functions as bound methods
    from .les_models import (update_smag_viscosity,
                             update_dyn_smag_coeff,
                             rhs_smagorinsky)

    parser = deepcopy(DNS.parser)

    parser.add_argument(
        '--model', type=str.lower, default='dyn_smag',
        choices=['smag', 'dyn_smag', 'ode_fsm'],
        help="Choose a specific LES turbulence model")

    parser.add_argument(
        '--filter_type', type=str.lower, default='gaussian',
        choices=['compact', 'gaussian', 'tophat'],
        help="test filter type")

    parser.add_argument(
        '--ic_ratio', type=float, default=4/3,
        help="initial condition filter ratio")

    parser.add_argument(
        '--test_ratio', type=float, default=8/3,
        help="test filter ratio")

    for action in parser._actions:
        if action.dest == 'init_cond':
            action.choices.append('les')
            action.default = 'les'

    @property
    def nvar(self):
        return 3

    def configure_rhs(self):
        """

        """
        config = self.config
        ratio = config.ic_ratio

        # --------------------------------------------------------------
        # Configure the turbulence model
        # --------------------------------------------------------------
        self.Cs = 0.01  # default hard-code value
        self.Delta2 = (ratio * pi / self.k_dealias)**2

        # more "work" arrays needed, but overall less extra memory than Hybrid
        self.work = np.empty([6, *self.r.shape], self.r.dtype)
        self.nuT = self.work[5]  # just an alias

        if config.model == 'dyn_smag':
            if config.filter_type == "compact":
                kf = self.k_dealias / config.test_ratio
                self.Ktest = self.analytic_compact_filter(kf)

            elif config.filter_type == 'gaussian':
                delta = config.test_ratio * pi / self.k_dealias
                self.Ktest = self.analytic_gaussian_filter(delta)

            elif config.filter_type == "tophat":
                delta = config.test_ratio * pi / self.k_dealias
                self.Ktest = self.analytic_tophat_filter(delta)

            self.Ktest *= self.Kdealias
            self.step_updates.append(self.update_dyn_smag_coeff)

        elif config.model == 'smag':
            self.step_updates.append(self.update_smag_viscosity)

        else:  # 'ode_fsm', do nothing here.
            pass

        # --------------------------------------------------------------
        # Configure the Right-Hand Side
        # --------------------------------------------------------------
        self.rhs_terms.append(self.rhs_rotation_convection)
        if config.model in ['smag', 'dyn_smag']:
            self.rhs_terms.append(self.rhs_smagorinsky)
        else:  # == 'ode_fsm':
            self.soft_abort('ERROR: ode_fsm model not yet implemented!')
            # self.rhs_terms.append(self.rhs_ode_fsm)

        # --------------------------------------------------------------
        # Dealias and project pressure from nonlinear terms
        self.rhs_terms.append(self.rhs_dealias)
        self.rhs_terms.append(self.rhs_pressure_poisson)

        # --------------------------------------------------------------
        # Add linear terms
        self.rhs_terms.append(self.rhs_linear_diffusion)
        if config.forcing:
            nmod = self.num_wavemodes
            kd = self.k_dealias
            kfL = config.kfLow
            kfH = config.kfHigh
            self._bandpass = smooth_bandpass(nmod, kfL, kfH, kd)
            self.step_updates.append(self.update_spectral_forcing)
            self.rhs_terms.append(self.rhs_spectral_forcing)

        return

    def initialize_simulation(self):
        config = self.config

        if config.init_cond == 'random':
            energy = config.KE_init
            exponent = config.kExp_init
            rseed = self.comm.rank  # for dev testing only
            self.initialize_random_spectrum(energy, exponent, rseed)

        elif config.init_cond == 'taylor_green':
            self.initialize_taylor_green_vortex()

        elif config.init_cond == 'les':
            ratio = config.ic_ratio
            frame = config.frame
            kf = int(self.k_dealias / ratio)

            filename = f'hybrid_IC_K1024_k{kf}.{frame:03d}.h5'
            config.init_file = f'{config.idir}/{filename}'
            self.initialize_les_from_dns(config.init_file)

        else:
            self.soft_abort('ERROR: incorrect initial condition')

        for update in self.step_updates:
            update(self.U_hat)

        return

    def step_update(self, U_hat, t_sim, dU, dt):
        """

        """
        new_dt = super().step_update(U_hat, t_sim, dU, dt)

        if new_dt < 1.0e-8:
            self.finalize()
            self.soft_abort("Error: tiny time step!")

        return new_dt

    def compute_dt(self, old_dt, *ignore):
        """

        """
        dx = 2 * pi / self.nx[0]
        # nuT = work[5]

        buffer = np.empty(2)
        buffer[0] = np.max(np.sum(np.abs(self.u), axis=0))
        buffer[1] = np.max(np.abs(self.nuT))
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)

        u_max = max(1e-20, buffer[0])
        nuT_max = max(1e-20, buffer[1])

        self._dt_hydro = self.config.cfl * dx / u_max
        self._dt_diff = self.config.Co_diff * dx**2 / (6 * (self.nu + nuT_max))

        dt = min(2.0 * old_dt, self._dt_hydro, self._dt_diff)

        return dt

    def initialize_les_from_dns(self, filename):
        config = self.config
        N = config.N

        # --------------------------------------------------------------
        # get the proper Fourier analysis context for the DNS data
        fh = h5FileIO(filename, 'r', self.comm)
        Np = fh["U_hat0"].shape[0]
        self.all_assert(Np >= N)

        iofft = fft3d([Np]*3, comm=self.comm)  # file IO
        io_hat = Function(iofft)

        # --------------------------------------------------------------
        # Read in and truncate the pre-processed DNS data
        for i in range(3):
            fh.read(f'U_hat{i}', io_hat)
            io_hat.refine(N, self.u_hat[i])

        self.u_hat *= self.Kdealias
        self.vfft.backward(self.u_hat, self.u)

        return

    def write_statistics_to_file(self, U_hat, t_sim):
        istat = self.config.istat
        iK = self.iK
        work = self.work[0]
        nuT = self.nuT  # nuT is just an alias to work[5]

        grp = None
        if self.comm.rank == 0:
            grp = self.fh_stat.create_group(f"{istat:03d}")
            grp.attrs['t_sim'] = t_sim

        Ek1d = self.compute_energy_spectrum(U_hat[:3])
        Tk1d = self.compute_transfer_spectrum(U_hat[:3])
        if self.comm.rank == 0:
            grp['Ek'] = Ek1d
            grp['Tk'] = Tk1d

        work[:] = 0.5 * dot(self.u, self.u) + 1e-99
        K_r = self.write_histogram(grp, work, 'k_r')[0]

        self.curl(U_hat[:3], out=self.w)
        work[:] = dot(self.w, self.w) + 1e-99
        Enst = self.write_histogram(grp, work, 'enst_r')[0]

        for i in range(3):
            Sii = self.fft.backward(iK[i] * U_hat[i])
            self.w[i] = Sii

        S2, S3 = self.write_histogram(grp, self.w, 'S_diag')[1:3]
        Skew = S3 / S2**1.5

        work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
        self.write_histogram(grp, work, 'Ssq')

        work *= self.nu
        eps_r = self.write_histogram(grp, work, 'eps_r')[0]

        # LES models will have already computed nuT in their
        # respective `update_step()` method
        NuT = self.write_histogram(grp, nuT, 'nuT')[0]

        work *= nuT / self.nu  # convert eps_r to Prod
        Prod = self.write_histogram(grp, work, 'Prod')[0]

        self.w_hat[:] = 0.0
        k = 0
        for i in range(2):
            for j in range(i+1, 3):
                Sij = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])

                self.w[k] = nuT * Sij
                Rij_hat = self.fft.forward(self.w[k])
                self.w_hat[i] += iK[j] * Rij_hat
                self.w_hat[j] += iK[i] * Rij_hat
                k += 1

        self.write_histogram(grp, self.w, 'tau_triu')

        for i in range(3):
            Sii = self.fft.backward(iK[i] * U_hat[i])
            self.w[i] = nuT * Sii
            Rii_hat = self.fft.forward(self.w[i])
            self.w_hat[i] += iK[i] * Rii_hat

        self.write_histogram(grp, self.w, 'tau_diag')

        self.vfft.backward(self.w_hat, self.w)
        self.write_histogram(grp, self.w, 'force')

        eps_t = eps_r
        K_t = K_r
        Re = sqrt(20/3) * K_t / sqrt(self.nu * eps_t)
        Lu = K_t**1.5 / (eps_t * 2*pi)

        vel = allsum(self.u)/(3 * self.num_points)
        div = allmax(np.abs(self.div(U_hat[:3], out=work)))

        history = (f"{t_sim:< 15.7e} "
                   f"{K_r:< 15.7e} "
                   f"{eps_r:< 15.7e} "
                   f"{0.0:< 15.7e} "
                   f"{0.0:< 15.7e} "
                   f"{self.Cs:< 15.7e} "
                   f"{NuT:< 15.7e} "
                   f"{Prod:< 15.7e} "
                   f"{Enst:< 15.7e} "
                   f"{Skew:< 15.7e} "
                   f"{Re:< 15.7e} "
                   f"{Lu:< 15.7e} "
                   f"{self._dt_hydro:< 15.7e} "
                   f"{self._dt_diff:< 15.7e} "
                   f"{0.0:< 15.7e} "
                   f"{vel:< 15.7e} "
                   f"{div:< 15.7e} "
                   f"{str(self.config.forcing):6s}"
                   "\n")

        if self.comm.rank == 0:
            self.fh_hist.write(history)

        self.config.istat += 1

        return

    histfile_header = (
        "# Simulation Global Statistics\n"
        "# ------------------------------------------------\n"
        "#  0 = simulation time\n"
        "#  1 = Resolved Kinetic Energy \n"
        "#  2 = Resolved KE Dissipation \n"
        "#  3 = Modeled Kinetic Energy \n"
        "#  4 = Modeled KE Dissipation \n"
        "#  5 = Modeled nuT coefficient (<C_m>)\n"
        "#  6 = Modeled eddy viscosity (<nu_T>)\n"
        "#  7 = Modeled SGS dissipation (<Pi>)\n"
        "#  8 = Resolved Enstrophy \n"
        "#  9 = Longitudinal Resolved Velocity Gradient Skewness \n"
        "# 10 = Taylor-scale Reynolds number (Re_lam)\n"
        "# 11 = Integral length scale (Lu/L = k_t**1.5 / (eps_t * L))\n"
        "# 12 = CFL timestep (dt_hydro)\n"
        "# 13 = von Neumann timestep (dt_diff)\n"
        "# 14 = growth/decay limit for k-eps (dt_keps)\n"
        "# 15 = Mean velocity (<U>)\n"
        "# 16 = Maximum magnitude of divergence [max(abs(div(u))]\n"
        "# 17 = Forcing On/Off (True/False)\n"
        "# ------------------------------------------------\n"
        )
