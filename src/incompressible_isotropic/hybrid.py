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
from math import pi, sqrt, fsum, ceil
import numpy as np

from shenfun import Array, Function
from shenfun.fourier import energy_fourier

from .dns import DNS
from ..fourier_analysis import FourierAnalysis, fft3d
from ..file_io import h5FileIO
from ..maths import dot, smooth_bandpass
from ..statistics import allsum, allmin, allmax

comm = MPI.COMM_WORLD


class Hybrid(DNS):

    # import all turbulence model functions as bound methods
    from .hybrid_models import (update_baseline_ce2,
                                update_keps_viscosity,
                                update_fsm_coeffs,
                                update_vkfsm_coeffs,
                                update_fkfsm_coeffs,
                                update_xfsm_coeffs,
                                update_rgfsm_coeffs,
                                update_dynamic_cmu,
                                update_pfsm_coeffs,
                                update_perot_coeffs,
                                update_des_coeffs,
                                update_xles_coeffs,
                                update_rgtau_coeffs,
                                update_pans_coeffs,
                                update_pitm_coeffs,
                                update_opans_coeffs,
                                update_opitm_coeffs,
                                update_ocess_coeffs,
                                update_cess_coeffs,
                                update_cesk_coeffs,
                                update_cesu_coeffs,
                                update_cesx_coeffs,
                                rhs_k_epsilon,
                                rhs_log_k_eps)

    parser = deepcopy(DNS.parser)

    parser.add_argument(
        '--Co_keps', type=float, default=0.1,
        help='k-eps equation maximum-percent-change limit.')

    parser.add_argument(
        '--cycle_limit', type=int, default=6000,
        help='maximum allowed time steps.')

    parser.add_argument(
        '--model', type=str.lower, default='raw',
        choices=['raw', 'fsm', 'vkfsm', 'fkfsm', 'xfsm', 'rgfsm', 'dynamic',
                 'des', 'xles', 'rgtau', 'perot', 'pfsm',
                 'pans', 'pitm', 'opans', 'opitm', 'ocess',
                 'cess', 'cesk', 'cesu', 'cesx'],
        help="Choose a specific hybrid turbulence model")

    parser.add_argument(
        '--base_ce2', dest='Ce2', type=float, default=1.714,
        help="Set the baseline k-epsilon Ce2 coefficient value")

    parser.add_argument(
        '--alt_ce2', action='store_true', default=False,
        help="Use Reynolds-dependent Ce2 coefficient")

    parser.add_argument(
        '--Cdes', type=float, default=0.61,
        help="Set the DES $C_\\Delta$ coefficient value")

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
            action.choices = ['hybrid', 'urans']
            action.default = 'hybrid'
        elif action.dest == 'dealiasing':
            action.choices = ['nyquist', 'partial', 'iso2/3', '2/3']
            action.default = '2/3'

    @property
    def nvar(self):
        return 5

    def configure_rhs(self):
        """

        """
        config = self.config

        # --------------------------------------------------------------
        # Configure the turbulence model
        # --------------------------------------------------------------
        self.nuT = np.empty_like(self.r)
        self.work = np.empty_like(self.r)
        self.alpha = np.array([1.0])
        self._dt_keps = 0.0

        if config.smooth_ke:
            delta = sqrt(12) * pi / self.kmax[0]  # sqrt(12) = 3.4641
            self.Ksmooth = self.analytic_gaussian_filter(delta)

        else:
            self.Ksmooth = 1.0

        # --------------------------------------------------------------
        # Baseline k-eps coefficients
        config.Cmu = 0.09
        config.sigk_inv = 1.0
        config.sige_inv = 1/1.3
        config.Ce1 = 1.44
        config.beta = 0.008

        ratio = config.ic_ratio
        config.Delta = ratio * pi / ceil(self.N[0] / 3)

        if config.alt_ce2:
            config.p = 4.0
            self.step_updates.append(self.update_baseline_ce2)

        self.Cmu = config.Cmu
        self.sigk_inv = config.sigk_inv
        self.sige_inv = config.sige_inv
        self.Ce1 = config.Ce1
        self.Ce2 = config.Ce2
        self.Ck2 = 1.0
        self.Delta = config.Cdes * config.Delta

        # --------------------------------------------------------------
        # Configure hybrid model coefficients
        if config.model == 'fsm':
            self.step_updates.append(self.update_fsm_coeffs)

        if config.model == 'vkfsm':
            self.step_updates.append(self.update_vkfsm_coeffs)

        if config.model == 'fkfsm':
            self.step_updates.append(self.update_fkfsm_coeffs)

        if config.model == 'xfsm':
            self.Cmu = np.ones_like(self.r)
            self.step_updates.append(self.update_xfsm_coeffs)

        if config.model == 'rgfsm':
            self.Cmu = np.ones_like(self.r)
            self.step_updates.append(self.update_rgfsm_coeffs)

        elif config.model == 'des':
            self.Ck2 = np.ones_like(self.r)
            self.step_updates.append(self.update_des_coeffs)

        elif config.model == 'xles':
            self.Cmu = np.ones_like(self.r)
            self.Ck2 = np.ones_like(self.r)
            self.step_updates.append(self.update_xles_coeffs)

        elif config.model == 'rgtau':
            self.Cmu = np.empty_like(self.r)
            self.Ck2 = np.ones_like(self.r)
            self.Ce1 = np.empty_like(self.r)
            self.Ce2 = np.empty_like(self.r)
            self.step_updates.append(self.update_rgtau_coeffs)

        elif config.model == 'pfsm':
            self.Cmu = np.empty_like(self.r)
            self.step_updates.append(self.update_pfsm_coeffs)

        elif config.model == 'perot':
            self.alpha = np.empty_like(self.r)
            self.Cmu = np.empty_like(self.r)
            self.step_updates.append(self.update_perot_coeffs)

        elif config.model == 'pans':
            self._computed = False
            self.step_updates.append(self.update_pans_coeffs)

        elif config.model == 'pitm':
            self.step_updates.append(self.update_pitm_coeffs)

        elif config.model == 'opans':
            self.step_updates.append(self.update_opans_coeffs)

        elif config.model == 'opitm':
            self.step_updates.append(self.update_opitm_coeffs)

        elif config.model == 'ocess':
            self.step_updates.append(self.update_ocess_coeffs)

        elif config.model == 'cess':
            self.step_updates.append(self.update_cess_coeffs)

        elif config.model == 'cesk':
            self.step_updates.append(self.update_cesk_coeffs)

        elif config.model == 'cesu':
            self.step_updates.append(self.update_cesu_coeffs)

        elif config.model == 'cesx':
            self.step_updates.append(self.update_cesx_coeffs)

        elif config.model == 'dynamic':
            self.work2 = np.empty([2, *self.r.shape], self.r.dtype)

            delta = config.Delta * config.test_ratio / config.ic_ratio
            self.Ktest = self.analytic_gaussian_filter(delta)
            # self.Ktest *= self.Kdealias

            self.step_updates.append(self.update_dynamic_cmu)

        else:  # "raw", do nothing!
            pass

        # --------------------------------------------------------------
        # Update nuT after the model coefficients
        self.step_updates.append(self.update_keps_viscosity)

        # --------------------------------------------------------------
        # Configure the Right-Hand Side
        # --------------------------------------------------------------
        self.rhs_terms.append(self.rhs_rotation_convection)
        if config.log_ke:
            self.rhs_terms.append(self.rhs_log_k_eps)
        else:
            self.rhs_terms.append(self.rhs_k_epsilon)

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
        """

        """
        config = self.config

        if config.init_cond == 'hybrid':
            ratio = config.ic_ratio
            frame = config.frame
            kf = round(self.N[0] / (3 * ratio))

            filename = f'hybrid_IC_K1024_k{kf}.{frame:03d}.h5'
            config.init_file = f'{config.idir}/{filename}'
            self.initialize_hybrid_from_dns(config.init_file)

        elif config.init_cond == 'urans':
            self.initialize_urans_from_dns(config.init_file)

        else:
            self.soft_abort('ERROR: incorrect initial condition')

        for update in self.step_updates:
            update(self.U_hat)

        # if config.model in ['cesk', 'cesx', 'cesu']:
        #     if config.log_ke:
        #         self.U[4] -= np.log(self.Ck2)
        #     else:
        #         self.U[4] /= self.Ck2
        #
        #     self.fft.forward(self.U[4], self.U_hat[4])

        return

    def total_energy(self):
        """

        """
        if self.config.log_ke:
            k_m = self.W[3] = np.exp(self.U[3])
        else:
            k_m = self.U[3]

        self.W[0] = 0.5 * dot(self.u, self.u) + k_m

        return allsum(self.W[0]) / self.num_points

    def integral_length(self, U_hat):
        """

        """
        eps = self.W[4]
        if self.config.log_ke:
            k_m = self.W[3] = np.exp(self.U[3])
            eps[:] = self.Ck2 * np.exp(self.U[4])
        else:
            k_m = self.U[3]
            eps[:] = self.Ck2 * self.U[4]

        self.curl(U_hat[:3], out=self.w)
        self.w[1] = self.nu * dot(self.w, self.w) + eps
        self.w[0] = 0.5 * dot(self.u, self.u) + k_m

        buffer = np.empty(2)
        buffer[0] = fsum(self.w[0].flat)
        buffer[1] = fsum(self.w[1].flat)
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
        buffer /= self.num_points

        K_t, Eps_t = buffer

        return K_t**1.5 / Eps_t

    def step_update(self, U_hat, t_sim, dU, dt):
        """

        """
        config = self.config
        max_km = getattr(self, 'max_km', np.inf)

        buffer = np.empty(3, dtype=np.int)
        buffer[0] = np.sum(self.U[3] > max_km)
        buffer[1] = np.sum(self.U[4] < 1e-15)
        buffer[2] = np.sum(self.U[3] < 1e-15)
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

        if buffer[0] > 0:
            # self.print(f'Warning: km > max! n={buffer[0]}')
            # self.clip_km(U_hat, t_sim)
            if buffer[0] > 64:
                self.finalize(t_sim)
                self.soft_abort("Error: hit kmax limit!")

        if not self.config.log_ke and (buffer[1] > 0 or buffer[2] > 0):
            self.finalize(t_sim)
            self.soft_abort("Error: negative k or eps value!")

        new_dt = super().step_update(U_hat, t_sim, dU, dt)

        if new_dt < 5.0e-7:
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
        sig_max = max(1, self.sigk_inv, self.sige_inv)
        dx = 2 * pi / self.nx[0]

        work = self.work

        self.fft.backward(dU[3], self.W[3])
        self.fft.backward(dU[4], self.W[4])

        buffer = np.empty(4)

        work[:] = np.sum(np.abs(self.u), axis=0)
        buffer[0] = np.max(work)
        buffer[1] = sig_max * np.max(self.nuT)

        work[:] = np.abs(self.alpha) * self.nuT
        buffer[2] = np.max(work)

        if config.log_ke:
            buffer[3] = np.max(np.abs(self.W[3:]))
        else:
            buffer[3] = np.max(np.abs(self.W[3:]) / self.U[3:])

        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.MAX)

        u_max = max(1e-20, buffer[0])
        nuT_max = max(1e-20, buffer[1], buffer[2])
        dlnYdt_max = max(1e-20, buffer[3])

        self._dt_hydro = CFL * dx / u_max
        self._dt_diff = Co_diff * dx**2 / (6 * (self.nu + nuT_max))
        self._dt_keps = Co_keps / dlnYdt_max

        dt = min(self._dt_hydro, self._dt_diff, self._dt_keps)
        dt = min(2.0 * old_dt, dt)

        return dt

    def initialize_urans_from_dns(self, filename):
        """Generates a k-epsilon URANS initial condition from DNS data

        """
        config = self.config

        fh = h5FileIO(filename, 'r', self.comm)
        Np = fh.attrs['N']

        if fh.attrs['dealiasing'] in ['3/2', 'iso3/2']:
            padding = 1.5
        else:
            padding = 1

        dns = FourierAnalysis([Np]*3, padding=padding, comm=self.comm)

        v_hat = Function(dns.vfft)
        v = Array(dns.vfft)
        w_hat = Function(dns.fft)

        # --------------------------------------------------------------
        # read in the DNS data, maintaining backwards compatibility
        if "/U_hat0" in fh:
            fh.read("U_hat", v_hat)
            dns.vfft.backward(v_hat, v)

        else:
            fh.read("U", v)
            dns.vfft.forward(v, v_hat)

        fh.close()

        # --------------------------------------------------------------
        # compute u_bar, k_t, and eps_t
        if np.sum(dns.Ksq == 0) == 1:
            u_bar = v_hat[:, dns.Ksq == 0].reshape((3,)).astype('f8')
        else:
            u_bar = np.zeros(3)
        self.comm.Allreduce(MPI.IN_PLACE, u_bar, op=MPI.SUM)

        self.U[0] = allsum(v[0]) / dns.num_points
        self.U[1] = allsum(v[1]) / dns.num_points
        self.U[2] = allsum(v[2]) / dns.num_points

        self.print(f'testing getting DC mode as mean: u_bar = {u_bar}, '
                   f'allsum(v) = {self.U[:, 0, 0, 0]}')

        # k
        self.U[3] = 0.5 * energy_fourier(v_hat, dns.fft)

        # eps
        w_hat[0] = dns.iK[1] * v_hat[2] - dns.iK[2] * v_hat[1]
        self.U[4] = self.nu * energy_fourier(w_hat, dns.fft)

        w_hat[1] = dns.iK[2] * v_hat[0] - dns.iK[0] * v_hat[2]
        self.U[4] += self.nu * energy_fourier(w_hat, dns.fft)

        w_hat[2] = dns.iK[0] * v_hat[1] - dns.iK[1] * v_hat[0]
        self.U[4] += self.nu * energy_fourier(w_hat, dns.fft)

        # --------------------------------------------------------------
        # Ensure both U and U_hat arrays are properly initialized.
        if config.log_ke:
            self.U[3:] = np.log(self.U[3:])

        self.Ufft.forward(self.U, self.U_hat)
        self.max_km = np.inf

        return

    def initialize_hybrid_from_dns(self, filename):
        """Generates a k-epsilon hybrid LES initial condition from DNS data

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
        E_m = fh.attrs['E_m']
        E_r = fh.attrs['E_r']
        Fk = K_m / (K_m + K_r)
        Fe = E_m / (E_m + E_r)

        #  -- spectral truncation of u preserves divergence-free condition
        for i in range(3):
            fh.read(f'U_hat{i}', io_hat)
            io_hat.refine(N, self.u_hat[i])

        self.u_hat *= self.Kdealias
        self.vfft.backward(self.u_hat, self.u)

        # -- physical-space sub-sampling of k-eps preserves positivity
        fh.read("U_hat3", io_hat)

        self.max_km = 1.01 * allmax(iofft.backward(io_hat))

        io_hat *= IC_smooth
        io_hat.refine(Nds, ds_hat)
        ds_hat.mask_nyquist()
        dsfft.backward(ds_hat, data)
        self.U[3] = data[sample]

        fh.read("U_hat4", io_hat)

        io_hat *= IC_smooth
        io_hat.refine(Nds, ds_hat)
        ds_hat.mask_nyquist()
        dsfft.backward(ds_hat, data)
        self.U[4] = data[sample]

        fh.close()

        # --------------------------------------------------------------
        # test the quality of the truncated initial condition
        self.work[:] = dot(self.u, self.u)
        k_r = 0.5 * allsum(self.work) / self.num_points

        self.work[:] = self.strain_squared(self.u_hat, out=self.work)
        e_r = self.nu * allsum(self.work) / self.num_points

        k_m = allsum(self.U[3]) / np.prod(self.N)
        e_m = allsum(self.U[4]) / np.prod(self.N)
        k_min = allmin(self.U[3])
        e_min = allmin(self.U[4])

        fk = k_m / (k_m + k_r)
        fe = e_m / (e_m + e_r)
        err0 = abs(Fk - fk) / Fk
        err1 = abs(Fe - fe) / Fe

        # --------------------------------------------------------------
        # get the dealiased k-eps values and check quality again
        if config.log_ke:
            self.U[3:] = np.log(self.U[3:])
            self.max_km = np.log(self.max_km)

        for i in [3, 4]:
            self.fft.forward(self.U[i], self.U_hat[i])
            self.U_hat[i] *= self.Kdealias
            self.fft.backward(self.U_hat[i], self.U[i])

        if config.log_ke:
            self.W[3:] = np.exp(self.U[3:])
            km2 = allsum(self.W[3]) / self.num_points
            em2 = allsum(self.W[4]) / self.num_points
        else:
            km2 = allsum(self.U[3]) / self.num_points
            em2 = allsum(self.U[4]) / self.num_points

        fk = km2 / (km2 + k_r)
        fe = em2 / (em2 + e_r)
        err2 = abs(Fk - fk) / Fk
        err3 = abs(Fe - fe) / Fe

        err_str = (f'{N}, {ratio:0.2f}, {Fk:0.3e}, {Fe:0.3e}, '
                   f'{err0:0.3e}, {err1:0.3e}, {err2:0.3e}, {err3:0.3e}, '
                   f'{k_min:0.3e}, {e_min:0.3e},')
        self.print(err_str, True)

        return

    def write_statistics_to_file(self, U_hat, t_sim):
        config = self.config
        istat = config.istat
        iK = self.iK
        alpha = self.alpha
        nuT = self.nuT
        work = self.work
        eps = self.W[4]
        phi = self.W[2]

        if config.log_ke:
            # lnk = self.U[3]
            # lne = self.U[4]
            k_m = self.W[3] = np.exp(self.U[3])
            eps[:] = phi[:] = np.exp(self.U[4])
        else:
            k_m = self.U[3]
            eps[:] = phi[:] = self.U[4]
        eps *= self.Ck2

        grp = None
        if self.comm.rank == 0:
            grp = self.fh_stat.create_group(f"{istat:03d}")
            grp.attrs['t_sim'] = t_sim

        K_m = self.write_histogram(grp, k_m, 'k_m')[0]
        phi_m = self.write_histogram(grp, phi, 'phi_m')[0]
        eps_m = self.write_histogram(grp, eps, 'eps_m')[0]

        self.write_histogram(grp, nuT, 'nuT')
        C_m = self.Cmu
        if np.size(self.Cmu) > 1:
            C_m = self.write_histogram(grp, self.Cmu, 'C_m')[0]

            if alpha.size > 1:
                self.write_histogram(grp, alpha, 'alpha')
                work[:] = alpha * self.Cmu
                self.write_histogram(grp, work, 'a.C_m')

        # f_Delta = C*Delta/L_m = C * Delta * eps_m / k_m**(3/2)
        if config.log_ke:
            work[:] = self.Delta * np.exp(self.U[4] - 1.5*self.U[3])
        else:
            work[:] = self.Delta * self.U[4] * self.U[3]**(-1.5)
        f_D = self.write_histogram(grp, work, 'f_Delta')[0]

        work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
        self.write_histogram(grp, work, 'Ssq')

        work *= self.nu
        eps_r = self.write_histogram(grp, work, 'eps_r')[0]

        work *= (1/self.nu) * alpha * nuT   # convert eps_r to Prod
        Prod = self.write_histogram(grp, work, 'Prod')[0]

        work *= self.Ce1 * phi / k_m
        P_eps = self.write_histogram(grp, work, 'P_eps')[0]

        work[:] = self.Ce2 * phi**2 / k_m
        D_eps = self.write_histogram(grp, work, 'D_eps')[0]

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

        self.w_hat[:] = 0.0
        k = 0
        for i in range(2):
            for j in range(i+1, 3):
                Sij = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])

                self.w[k] = alpha * nuT * Sij
                Rij_hat = self.fft.forward(self.w[k])
                self.w_hat[i] += iK[j] * Rij_hat
                self.w_hat[j] += iK[i] * Rij_hat
                k += 1

        self.write_histogram(grp, self.w, 'tau_triu')

        for i in range(3):
            Sii = self.fft.backward(iK[i] * U_hat[i])
            self.w[i] = alpha * nuT * Sii
            Rii_hat = self.fft.forward(self.w[i])
            self.w_hat[i] += iK[i] * Rii_hat

        self.write_histogram(grp, self.w, 'tau_diag')

        self.vfft.backward(self.w_hat, self.w)
        self.write_histogram(grp, self.w, 'force')

        E1d = self.compute_energy_spectrum(U_hat[:3])
        T1d = self.compute_transfer_spectrum(U_hat[:3])
        Km1d = self.compute_energy_spectrum(U_hat[3])
        Dm1d = self.compute_energy_spectrum(U_hat[4])

        if self.comm.rank == 0:
            grp['E1d'] = E1d
            grp['T1d'] = T1d
            grp['Km1d'] = Km1d
            grp['Dm1d'] = Dm1d

        # if config.log_ke is False:
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_Prod')
        #
        #     work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
        #     work *= nuT * self.U[4] / k_m
        #     self.write_histogram(grp, work, 'Pe_k')
        #
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_Pe_k')
        #
        #     work[:] = self.U[4]**2 / k_m
        #     self.write_histogram(grp, work, 'eps2_k')
        #
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_eps2_k')
        #
        # else:
        #     work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
        #     work *= alpha * self.Cmu * np.exp(lnk - lne)
        #     self.write_histogram(grp, work, 'P_k')
        #
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_Pk')
        #
        #     work[:] = self.Ck2 * np.exp(lne - lnk)
        #     self.write_histogram(grp, work, 'eps_k')
        #
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_eps_k')
        #
        #     # compute grad(lnk)
        #     self.w_hat[:] = iK * U_hat[3]
        #     self.vfft.backward(self.w_hat, self.w)
        #
        #     # compute lnk source terms
        #     work[:] = dot(self.w, self.w) + 1e-99
        #     work *= self.sigk_inv * nuT + self.nu
        #     self.write_histogram(grp, work, 'nu_Dk_Dk')
        #
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_nuDkDk')
        #
        #     # compute grad(lne)
        #     self.w_hat[:] = iK * U_hat[4]
        #     self.vfft.backward(self.w_hat, self.w)
        #
        #     # compute lne source terms
        #     work[:] = dot(self.w, self.w) + 1e-99
        #     work *= self.sige_inv * nuT + self.nu
        #     self.write_histogram(grp, work, 'nu_De_De')
        #
        #     work[:] = self.filter(work, self.Ksmooth * self.Kdealias)
        #     self.write_histogram(grp, work, 'D_nuDeDe')

        eps_t = eps_r + eps_m
        K_t = K_r + K_m
        Re = sqrt(20/3) * K_t / sqrt(self.nu * eps_t)
        Lu = K_t**1.5 / (eps_t * 2*pi)
        Ck2 = eps_m / phi_m

        vel = allsum(self.u)/(3 * self.num_points)
        div = allmax(np.abs(self.div(U_hat[:3], out=work)))

        history = (f"{t_sim:< 15.7e} "
                   f"{K_r:< 15.7e} "
                   f"{eps_r:< 15.7e} "
                   f"{K_m:< 15.7e} "
                   f"{eps_m:< 15.7e} "
                   f"{Prod:< 15.7e} "
                   f"{P_eps:< 15.7e} "
                   f"{D_eps:< 15.7e} "
                   f"{C_m:< 15.7e} "
                   f"{f_D:< 15.7e}"
                   f"{Ck2:< 15.7e}"
                   f"{Re:< 15.7e} "
                   f"{Lu:< 15.7e} "
                   f"{Enst:< 15.7e} "
                   f"{Skew:< 15.7e} "
                   f"{self._dt_hydro:< 15.7e} "
                   f"{self._dt_diff:< 15.7e} "
                   f"{self._dt_keps:< 15.7e} "
                   f"{vel:< 15.7e} "
                   f"{div:< 15.7e} "
                   "\n")

        if self.comm.rank == 0:
            self.fh_hist.write(history)

        config.istat += 1

        return

    histfile_header = (
        "# Simulation Global Statistics\n"
        "# ------------------------------------------------\n"
        "#  0 = t \n"
        "#  1 = <k_r> \n"
        "#  2 = <eps_r> \n"
        "#  3 = <k_m> \n"
        "#  4 = <eps_sgs> = <Ck2 * eps_m> \n"
        "#  5 = <Pi> \n"
        "#  6 = Production of eps_m \n"
        "#  7 = Destruction of eps_m \n"
        "#  8 = <C_mu> \n"
        "#  9 = <f_Delta> \n"
        "# 10 = {Ck2} \n"
        "# 11 = Re_lam \n"
        "# 12 = Integral length scale (Lu/L = k_t**1.5 / (eps_t * L)) \n"
        "# 13 = Enstrophy \n"
        "# 14 = Skewness \n"
        "# 15 = CFL timestep (dt_hydro) \n"
        "# 16 = von Neumann timestep (dt_diff) \n"
        "# 17 = growth/decay limit for k-eps (dt_keps) \n"
        "# 18 = Mean velocity (<U>) \n"
        "# 19 = Maximum magnitude of divergence [max(abs(div(u))] \n"
        "# ------------------------------------------------\n"
        )
