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
from math import pi, sqrt, ceil
import numpy as np

from shenfun import Array, Function

from .dns import DNS
from ..fourier_analysis import fft3d
from ..file_io import h5FileIO
from ..maths import dot, smooth_bandpass
from ..statistics import allsum, allmin, allmax

comm = MPI.COMM_WORLD


class KSGS(DNS):

    # import all turbulence model functions as bound methods
    from .les_models import (update_ksgs_viscosity,
                             update_dynk_coeff,
                             update_ldmk_coeff,
                             rhs_k_sgs,
                             rhs_logk_sgs)

    parser = deepcopy(DNS.parser)

    parser.add_argument(
        '--Co_keps', type=float, default=0.1,
        help='k-eps equation maximum-percent-change limit.')

    parser.add_argument(
        '--cycle_limit', type=int, default=10000,
        help='maximum allowed time steps.')

    parser.add_argument(
        '--model', type=str.lower, default='dyn_ksgs',
        choices=['ksgs', 'dynk', 'ldmk', 'ode_blend'],
        help="Choose a specific one-equation LES turbulence model")

    parser.add_argument(
        '--Ce', type=float, default=1.0,
        help='k-SGS dissipation coefficient.')

    parser.add_argument(
        '--log_ke', action='store_true', default=True,
        help="Use logarthmic k-epsilon variables")

    parser.add_argument(
        '--no-log_ke', dest='log_ke', action='store_false',
        help="Do not use logarthmic k-epsilon variables")

    parser.add_argument(
        '--smooth_ic', action='store_true', default=True,
        help="Smooth k-epsilon initial conditions before subsampling")

    parser.add_argument(
        '--no-smooth_ic', dest='smooth_ic', action='store_false',
        help="Do not smooth k-epsilon initial conditions before subsampling")

    parser.add_argument(
        '--smooth_ke', action='store_true', default=True,
        help="Smooth dealias k-epsilon variables")

    parser.add_argument(
        '--no-smooth_ke', dest='smooth_ke', action='store_false',
        help="Do not smooth dealias k-epsilon variables")

    parser.add_argument(
        '--ic_ratio', type=float, default=2.0,
        help="initial condition filter ratio")

    parser.add_argument(
        '--test_ratio', type=float, default=4.0,
        help="test filter ratio")

    for action in parser._actions:
        if action.dest == 'init_cond':
            action.choices.append('ksgs')
            action.default = 'ksgs'

        elif action.dest == 'dealiasing':
            action.choices = ['nyquist', 'partial', 'iso2/3', '2/3']
            action.default = '2/3'

    @property
    def nvar(self):
        return 4

    def configure_rhs(self):
        """

        """
        config = self.config
        ratio = config.ic_ratio
        config.Delta = ratio * pi / ceil(self.N[0] / 3)
        config.Cs = 0.055
        config.sigk_inv = 1.0

        # --------------------------------------------------------------
        # Configure the turbulence model
        # --------------------------------------------------------------
        self.Delta = config.Delta
        self.Cs = config.Cs
        self.Ce = config.Ce
        self.sigk_inv = config.sigk_inv

        self.work = np.empty_like(self.r)
        self.work2 = np.empty([3, *self.r.shape], self.r.dtype)
        self.nuT = np.empty_like(self.r)

        self._dt_keps = 0.0

        if config.smooth_ke:
            delta = 3 * pi / self.kmax[0]
            self.Ksmooth = self.analytic_gaussian_filter(delta)

        else:
            self.Ksmooth = 1.0

        if config.model in ['dynk', 'ldmk']:
            delta = config.Delta * config.test_ratio / config.ic_ratio
            self.Ktest = self.analytic_gaussian_filter(delta)
            # self.Ktest *= self.Kdealias

        if config.model == 'dynk':
            self.step_updates.append(self.update_dynk_coeff)
            self.step_updates.append(self.update_ksgs_viscosity)

        elif config.model == 'ldmk':
            self.Cs = np.empty_like(self.r)
            self.step_updates.append(self.update_ldmk_coeff)
            self.step_updates.append(self.update_ksgs_viscosity)

        elif config.model == 'ksgs':
            self.step_updates.append(self.update_ksgs_viscosity)

        else:  # 'ode_blend', do nothing here.
            self.soft_abort('ERROR: ode_blend model not yet implemented!')

        # --------------------------------------------------------------
        # Configure the Right-Hand Side
        # --------------------------------------------------------------
        self.rhs_terms.append(self.rhs_rotation_convection)
        if config.log_ke:
            self.rhs_terms.append(self.rhs_logk_sgs)
        else:
            self.rhs_terms.append(self.rhs_k_sgs)

        # --------------------------------------------------------------
        # Dealias and project pressure from momentum equation
        self.rhs_terms.append(self.rhs_dealias)
        self.rhs_terms.append(self.rhs_pressure_poisson)

        # --------------------------------------------------------------
        # Add linear terms (do not contribute to pressure or aliasing)
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

        if config.init_cond == 'ksgs':
            ratio = config.ic_ratio
            frame = config.frame
            kf = round(self.N[0] / (3 * ratio))

            filename = f'hybrid_IC_K1024_k{kf}.{frame:03d}.h5'
            config.init_file = f'{config.idir}/{filename}'
            self.initialize_ksgs_from_dns(config.init_file)

        elif config.init_cond == 'random':
            energy = config.KE_init
            exponent = config.kExp_init
            rseed = self.comm.rank  # for dev testing only
            self.initialize_random_spectrum(energy, exponent, rseed)

        elif config.init_cond == 'taylor_green':
            self.initialize_taylor_green_vortex()

        else:
            self.soft_abort('ERROR: incorrect initial condition')

        for update in self.step_updates:
            update(self.U_hat)

        return

    def total_energy(self):
        if self.config.log_ke:
            k_m = self.W[3] = np.exp(self.U[3])
        else:
            k_m = self.U[3]

        self.W[0] = 0.5 * dot(self.u, self.u) + k_m

        return allsum(self.W[0]) / self.num_points

    def step_update(self, U_hat, t_sim, dU, dt):
        """

        """
        config = self.config
        max_km = getattr(self, 'max_km', np.inf)

        buffer = np.empty(2, dtype=np.int)
        buffer[0] = np.sum(self.U[3] > max_km)
        buffer[1] = np.sum(self.U[3] < 1e-15)
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

        if buffer[0] > 0:
            self.print(f'Warning: km > max! n={buffer[0]}')
            # self.clip_km(U_hat, t_sim)

        if not self.config.log_ke and (buffer[1] > 0):
            self.finalize(t_sim)
            self.soft_abort("Error: negative k or eps value!")

        new_dt = super().step_update(U_hat, t_sim, dU, dt)

        if new_dt < 1.0e-8:
            self.finalize(t_sim)
            self.soft_abort("Error: tiny time step!")

        if self.istep > config.cycle_limit:
            self.finalize(t_sim)
            self.soft_abort("Error: hit cycle limit!")

        return new_dt

    def clip_km(self, U_hat, t_sim):
        self.U[3] = np.where(self.U[3] <= self.max_km, self.U[3], self.max_km)
        U_hat[3] = self.fft.forward(self.U[3])

        return U_hat

    def compute_dt(self, old_dt, U_hat, dU):
        """
        k-epsilon limiter:
        We want |dY|/Y < C (limit fractional change in Y over whole time step)
        where Y = [k, eps]. Noting that
            |dY| / Y = |dlnY/dt| * dt
        and rearranging,
            dt < C / |dlnY/dt|,
        therefore
            dt = min(C / |dlnY/dt|) = C / max|dlnY/dt|

        """
        config = self.config
        Co_keps = config.Co_keps
        Co_diff = config.Co_diff
        CFL = config.cfl
        sig_max = max(1, self.sigk_inv)
        dx = 2 * pi / self.nx[0]

        work = self.work

        self.fft.backward(dU[3], self.W[3])

        buffer = np.empty(3)

        work[:] = np.sum(np.abs(self.u), axis=0)
        buffer[0] = np.max(work)
        buffer[1] = sig_max * np.max(np.abs(self.nuT))

        if config.log_ke:
            buffer[2] = np.max(np.abs(self.W[3:]))
        else:
            buffer[2] = np.max(np.abs(self.W[3:]) / self.U[3:])

        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)

        u_max = max(1e-20, buffer[0])
        nuT_max = max(1e-20, buffer[1])
        dlnYdt_max = max(1e-20, buffer[2])

        self._dt_hydro = CFL * dx / u_max
        self._dt_diff = Co_diff * dx**2 / (6 * (self.nu + nuT_max))
        self._dt_keps = Co_keps / dlnYdt_max

        dt = min(self._dt_hydro, self._dt_diff, self._dt_keps)
        dt = min(2.0 * old_dt, dt)

        return dt

    def initialize_ksgs_from_dns(self, filename):
        """Generates a k-SGS LES initial condition from DNS data

        """
        config = self.config
        ratio = config.ic_ratio
        N = config.N

        # --------------------------------------------------------------
        # get the proper Fourier analysis context for the DNS data
        fh = h5FileIO(filename, 'r', self.comm)
        Nio = fh["U_hat0"].shape[0]
        Nds = max(1, Nio // N) * N
        skip = Nds // N
        sample = (slice(None, None, skip), )*3

        iofft = fft3d([Nio]*3, comm=self.comm)  # file IO
        dsfft = fft3d([Nds]*3, comm=self.comm)  # "DownSized", but could be up
        io_hat = Function(iofft)
        ds_hat = Function(dsfft)
        data = Array(dsfft)

        if config.smooth_ic:
            delta = 3 * pi / self.kmax[0]
            K = iofft.local_wavenumbers()
            Ksq = K[0]**2 + K[1]**2 + K[2]**2
            IC_smooth = np.exp(-(1/24) * delta**2 * Ksq)
        else:
            IC_smooth = 1.0

        # --------------------------------------------------------------
        # Read in and truncate the pre-processed DNS data
        K_m = fh.attrs['K_m']
        K_r = fh.attrs['K_r']
        Fk = K_m / (K_m + K_r)

        #  -- spectral truncation of u preserves divergence-free condition
        for i in range(3):
            fh.read(f'U_hat{i}', io_hat)
            io_hat.refine(N, self.u_hat[i])

        self.u_hat *= self.Kdealias
        self.vfft.backward(self.u_hat, self.u)

        # -- physical-space sub-sampling of k preserves positivity
        fh.read("U_hat3", io_hat)

        self.max_km = 1.01 * allmax(iofft.backward(io_hat))

        io_hat *= IC_smooth
        io_hat.refine(Nds, ds_hat)
        ds_hat.mask_nyquist()
        dsfft.backward(ds_hat, data)
        self.U[3] = data[sample]

        fh.close()

        # --------------------------------------------------------------
        # test the quality of the truncated initial condition
        self.work[:] = dot(self.u, self.u)
        k_r = 0.5 * allsum(self.work) / self.num_points

        k_m = allsum(self.U[3]) / np.prod(self.N)
        k_min = allmin(self.U[3])

        fk = k_m / (k_m + k_r)
        err0 = abs(Fk - fk) / Fk

        # --------------------------------------------------------------
        # get the dealiased k-eps values and check quality again
        if config.log_ke:
            self.U[3] = np.log(self.U[3])
            self.max_km = np.log(self.max_km)

        self.fft.forward(self.U[3], self.U_hat[3])
        self.U_hat[3] *= self.Kdealias
        self.fft.backward(self.U_hat[3], self.U[3])

        if config.log_ke:
            self.W[3] = np.exp(self.U[3])
            km2 = allsum(self.W[3]) / self.num_points
        else:
            km2 = allsum(self.U[3]) / self.num_points

        fk = km2 / (km2 + k_r)
        err2 = abs(Fk - fk) / Fk

        err_str = (f'{N}, {ratio:0.2f}, {Fk:0.3e}, '
                   f'{err0:0.3e}, {err2:0.3e}, {k_min:0.3e}')
        self.print(err_str, True)

        return

    def write_statistics_to_file(self, U_hat, t_sim):
        istat = self.config.istat
        iK = self.iK
        Cs = self.Cs
        Df = self.Delta

        nuT = self.nuT
        work = self.work

        if self.config.log_ke:
            lnk = self.U[3]
            k_m = self.W[3] = np.exp(self.U[3])

        else:
            k_m = self.U[3]

        grp = None
        if self.comm.rank == 0:
            grp = self.fh_stat.create_group(f"{istat:03d}")
            grp.attrs['t_sim'] = t_sim

        K_m = self.write_histogram(grp, k_m, 'k_m')[0]

        work[:] = (self.Ce / Df) * k_m**1.5
        Eps_m = self.write_histogram(grp, work, 'eps_m')[0]

        NuT = self.write_histogram(grp, nuT, 'nuT')[0]

        C_m = self.Cs
        if np.size(self.Cs) > 1:
            C_m = self.write_histogram(grp, self.Cs, 'C_m')[0]

        work[:] = 0.5 * dot(self.u, self.u) + 1e-99
        K_r = self.write_histogram(grp, work, 'k_r')[0]

        self.curl(U_hat[:3], out=self.w)
        work[:] = dot(self.w, self.w) + 1e-99
        Enst = self.write_histogram(grp, work, 'enst_r')[0]

        work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
        self.write_histogram(grp, work, 'Ssq')

        work *= self.nu
        Eps_r = self.write_histogram(grp, work, 'eps_r')[0]

        work *= (1/self.nu) * nuT   # convert eps_r to Prod
        Prod = self.write_histogram(grp, work, 'Prod')[0]

        for i in range(3):
            Sii = self.fft.backward(iK[i] * U_hat[i])
            self.w[i] = Sii

        S2, S3 = self.write_histogram(grp, self.w, 'S_diag')[1:3]
        Skew = S3 / S2**1.5

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

        E1d = self.compute_energy_spectrum(U_hat[:3])
        T1d = self.compute_transfer_spectrum(U_hat[:3])
        Km1d = self.compute_energy_spectrum(U_hat[3])

        if self.comm.rank == 0:
            grp['E1d'] = E1d
            grp['T1d'] = T1d
            grp['Km1d'] = Km1d

        if self.config.log_ke is False:
            work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
            self.write_histogram(grp, work, 'D_Prod')

        else:
            work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
            work *= Cs * Df * np.exp(-0.5*lnk)
            self.write_histogram(grp, work, 'P_k')

            work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
            self.write_histogram(grp, work, 'D_Pk')

            work[:] = (self.Ce / Df) * np.exp(0.5*lnk)
            self.write_histogram(grp, work, 'eps_k')

            work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
            self.write_histogram(grp, work, 'D_eps_k')

            # compute grad(lnk)
            self.w_hat[:] = iK * U_hat[3]
            self.vfft.backward(self.w_hat, self.w)

            # compute lnk source terms
            work[:] = dot(self.w, self.w) + 1e-99
            work[:] *= self.sigk_inv * nuT + self.nu
            self.write_histogram(grp, work, 'nu_Dk_Dk')

            work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
            self.write_histogram(grp, work, 'D_nuDkDk')

        Eps_t = Eps_r + Eps_m
        K_t = K_r + K_m
        Re = sqrt(20/3) * K_t / sqrt(self.nu * Eps_t)
        Lu = K_t**1.5 / (Eps_t * 2*pi)

        vel = allsum(self.u)/(3 * self.num_points)
        div = allmax(np.abs(self.div(U_hat[:3], out=work)))

        history = (f"{t_sim:< 15.7e} "
                   f"{K_r:< 15.7e} "
                   f"{Eps_r:< 15.7e} "
                   f"{K_m:< 15.7e} "
                   f"{Eps_m:< 15.7e} "
                   f"{C_m:< 15.7e} "
                   f"{NuT:< 15.7e} "
                   f"{Prod:< 15.7e} "
                   f"{Enst:< 15.7e} "
                   f"{Skew:< 15.7e} "
                   f"{Re:< 15.7e} "
                   f"{Lu:< 15.7e} "
                   f"{self._dt_hydro:< 15.7e} "
                   f"{self._dt_diff:< 15.7e} "
                   f"{self._dt_keps:< 15.7e} "
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
