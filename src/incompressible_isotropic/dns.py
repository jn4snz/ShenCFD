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

import os
from copy import deepcopy
from shutil import copyfile
from math import sqrt, pi, fsum

import numpy as np
import h5py

from .navierstokes import NavierStokes
from ..maths import dot
from ..statistics import allsum, allmax, moments, histogram1

comm = MPI.COMM_WORLD


class DNS(NavierStokes):

    parser = deepcopy(NavierStokes.parser)

    parser.add_argument(
        '--cfl', type=float, default=0.9,
        help='Advection CFL limit.')

    parser.add_argument(
        '--Co_diff', type=float, default=1.0,
        help='Diffusion Courant limit.')

    parser.add_argument(
        '--dt_stat', type=float, default=np.inf,
        help="time between detailed statistics and 1D spectra outputs")

    parser.add_argument(
        '--dry_run', action='store_true', default=False,
        help="Exit program immediately after configuring the simulation")

    parser.add_argument(
        '--init_only', action='store_true', default=False,
        help="Exit program after generating the initial conditions")

    parser.add_argument(
        '--no-init_only', dest='init_only', action='store_false',
        help="Disable init_only if it got into a restart file")

    def __init__(self, config=None, comm=comm):

        # Configure the simulation
        # -------------------------------------------------------------------
        super().__init__(config, self.nvar, comm)
        config = self.config

        if config.dry_run:
            lines = ['# Simulation Configuration ',
                     '# -------------------------']
            for k, v in vars(config).items():
                lines.append(f"{k:15s} = {v}")
            lines.append('# -------------------------\n')
            self.print('\n'.join(lines), True)

            return

        # Initialize the simulation and its outputs
        # -------------------------------------------------------------------
        self._dt_diff = 0.0
        self._dt_hydro = 0.0
        self.istep = 0

        self.fh_hist = None
        self.fh_stat = None

        if config.restart:
            # config already fully set from restart file by get_config(),
            # any necessary value overwrites take place in restart_simulation()
            self.restart_simulation()

        else:

            self.initialize_simulation()

            # config variables not set by get_config():
            config.t_rst = config.t_init
            config.t_stat = config.t_init
            config.istat = 0
            config.frame = 0  # overwrites existing get_config() value

            comm.Barrier()

            hist_file = f"{config.odir}/{config.pid}-history.txt"
            stat_file = f"{config.odir}/{config.pid}-statistics.h5"
            MiB = 1048576  # 1 mibibyte (Lustre stripe size)

            if comm.rank == 0:
                self.fh_stat = h5py.File(stat_file, 'w', driver='core',
                                         block_size=MiB)
                self.fh_hist = open(hist_file, 'wt', buffering=MiB)
                self.fh_hist.write(self.histfile_header)

            self.write_statistics_to_file(self.U_hat, config.t_init)
            self.write_restart_to_file(self.U_hat)

            config.t_rst += config.dt_rst
            config.t_stat += config.dt_stat

        # print simulation configuration to screen
        # -------------------------------------------------------------------

        lines = ['\n# Simulation Configuration ',
                 '# -------------------------']
        for k, v in vars(config).items():
            lines.append(f"{k:15s} = {v}")

        KE = self.total_energy()
        lines.append('# -------------------------\n'
                     f"cycle = {0:4d}, time = {config.t_init:< 14.6e}, "
                     f"dt = N/A           , KE = {KE:< 14.6e}")

        self.print('\n'.join(lines), True)

        return

    @property
    def nvar(self):
        return 3

    def restart_simulation(self):

        config = self.config
        hist_file = f"{config.odir}/{config.pid}-history.txt"
        stat_file = f"{config.odir}/{config.pid}-statistics.h5"
        MiB = 1048576  # 1 mibibyte (Lustre stripe size)

        self.initialize_from_restart(config.init_file)
        config.frame += 1
        config.t_rst = (config.t_init//config.dt_rst + 1) * config.dt_rst

        for update in self.step_updates:
            update(self.U_hat)

        # check for an existing history log
        old_history = f"{config.idir}/{config.init_pid}-history.txt"
        if os.path.isfile(old_history):
            comm.Barrier()
            if comm.rank == 0:
                if old_history != hist_file:
                    copyfile(old_history, hist_file)
                self.fh_hist = open(hist_file, 'at', buffering=MiB)

        else:  # old file does not exist for some reason
            comm.Barrier()
            if comm.rank == 0:
                self.fh_hist = open(hist_file, 'wt', buffering=MiB)
                self.fh_hist.write(self.histfile_header)

        # check for existing statistics outputs
        old_stats = f"{config.idir}/{config.init_pid}-statistics.h5"
        if os.path.isfile(old_stats):
            fh = h5py.File(old_stats, 'r')
            istat = max(map(int, fh.keys()))
            tstat = fh[f"{istat:03d}"].attrs["t_sim"]
            fh.close()

            config.istat = istat + 1
            config.t_stat = tstat + config.dt_stat

            comm.Barrier()
            if comm.rank == 0:
                if old_stats != stat_file:
                    copyfile(old_stats, stat_file)
                self.fh_stat = h5py.File(stat_file, 'a', driver='core',
                                         block_size=MiB)

        else:  # old file does not exist for some reason
            config.istat = 0
            config.t_stat = config.t_init

            comm.Barrier()
            if comm.rank == 0:
                self.fh_stat = h5py.File(stat_file, 'w', driver='core',
                                         block_size=MiB)
            self.write_statistics_to_file(self.U_hat, config.t_init)

        return

    def initialize_simulation(self):
        config = self.config

        if config.init_cond == 'random':
            energy = config.KE_init
            exponent = config.kExp_init
            rseed = self.comm.rank  # for dev testing only
            self.initialize_random_spectrum(energy, exponent, rseed)

        else:  # init_cond == 'taylor_green':
            self.initialize_taylor_green_vortex()

        for update in self.step_updates:
            update(self.U_hat)

        return

    def step_update(self, U_hat, t_sim, dU, dt):
        """

        """
        config = self.config

        for update in self.step_updates:
            update(U_hat, t_sim, dU, dt)

        new_dt = self.compute_dt(dt, U_hat, dU)
        self.istep += 1

        t_next = t_sim + new_dt

        KE = self.total_energy()
        self.print(f"cycle = {self.istep:4d}, time = {t_sim:< 14.6e}, "
                   f"dt = {dt:< 14.6e}, KE = {KE:< 14.6e}",
                   flush=(self.istep % 25 == 0))

        if t_sim + 1e-8 >= config.t_stat:
            config.t_stat += max(config.dt_stat, t_next - config.t_stat)
            self.write_statistics_to_file(U_hat, t_sim)

        rst_output = False
        if t_sim + 1e-8 >= config.t_rst:
            config.t_init = t_sim
            config.dt_init = new_dt
            config.t_rst += max(config.dt_rst, t_next - config.t_rst)
            self.write_restart_to_file(U_hat)
            rst_output = True

        flush_buffers = rst_output or (self.istep % 200 == 0)
        if flush_buffers and self.comm.rank == 0:
            self.fh_hist.flush()
            self.fh_stat.flush()
            print(' *** satistics flushed to disk '
                  f'({config.istat} entries)', flush=True)

        return new_dt

    def compute_dt(self, old_dt, *ignore):
        """Includes CFL conditions

        """
        dx = 2 * pi / self.nx[0]
        self._dt_diff = self.config.Co_diff * dx**2 / (6 * self.nu)

        u_max = allmax(np.sum(np.abs(self.u), axis=0))
        self._dt_hydro = self.config.cfl * dx / u_max

        dt = min(2.0 * old_dt, self._dt_hydro, self._dt_diff)

        return dt

    def finalize(self, t_sim=None):
        config = self.config
        config.t_init = t_sim or config.tlimit

        # output statistics without updating config.t_stat
        self.write_statistics_to_file(self.U_hat, config.t_init)

        # output checkpoint without updating config.t_rst
        self.write_restart_to_file(self.U_hat, checkpoint=True)

        if self.comm.rank == 0:
            self.fh_hist.flush()
            self.fh_stat.flush()
            print(' *** satistics flushed to disk '
                  f'({config.istat} entries)', flush=True)

            self.fh_hist.close()
            self.fh_stat.close()

        return

    def compute_transfer_spectrum(self, u_hat):
        r"""Compute the 1D (shell-averaged) spectral kinetic energy transfer
        :math:`\widehat{T}_K(\kappa)`.

        """
        iK = self.iK
        u = self.u
        w = self.w
        w_hat = self.w_hat

        # take curl of velocity and inverse transform to get vorticity
        w[0] = self.fft.backward(iK[1] * u_hat[2] - iK[2] * u_hat[1])
        w[1] = self.fft.backward(iK[2] * u_hat[0] - iK[0] * u_hat[2])
        w[2] = self.fft.backward(iK[0] * u_hat[1] - iK[1] * u_hat[0])

        # take cross-product of vorticity and velocity and transform back
        w_hat[0] = self.fft.forward(u[1] * w[2] - u[2] * w[1])
        w_hat[1] = self.fft.forward(u[2] * w[0] - u[0] * w[2])
        w_hat[2] = self.fft.forward(u[0] * w[1] - u[1] * w[0])

        self.project_divergence_free(w_hat)

        T3d = np.sum(np.real(w_hat * np.conj(u_hat)), axis=0)
        T3d[:, :, self.K[2].reshape(-1) == 0] *= 0.5
        T3d[:, :, self.K[2].reshape(-1) == self.kmax[2]] *= 0.5

        T1d = np.zeros(self.num_wavemodes, dtype=T3d.dtype)
        for k in range(self.num_wavemodes):
            T1d[k] = fsum(T3d[self.wavemodes == k].flat)

        self.comm.Allreduce(MPI.IN_PLACE, T1d, op=MPI.SUM)

        return T1d

    def write_histogram(self, fh, data, key):
        # get moments before taking log
        m = moments(data)
        xrange = m[-2:]

        # take log for strictly positive data
        if xrange[0] > 0.0:
            data = np.log10(data)
            xrange = tuple(np.log10(m[-2:]))
            log = True
            bins = 50
        else:
            log = False
            bins = 100

        hist, edges = histogram1(data, bins, xrange)

        if self.comm.rank == 0:
            grp = fh.create_group(key)
            grp['hist'] = hist
            grp['edges'] = edges
            grp.attrs['log'] = log
            grp.attrs['moments'] = m[:-2]
            grp.attrs['range'] = m[-2:]  # note I store non-log range here

        self.comm.Barrier()

        return m  # return raw 1st-6th moments

    def write_statistics_to_file(self, U_hat, t_sim):
        istat = self.config.istat
        iK = self.iK
        work = np.empty_like(self.r)

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
        k_r = self.write_histogram(grp, work, 'k_r')[0]

        work[:] = self.strain_squared(U_hat[:3], out=work) + 1e-99
        self.write_histogram(grp, work, 'SijSij')

        work *= self.nu
        eps_r = self.write_histogram(grp, work, 'eps_r')[0]

        self.curl(U_hat[:3], out=self.w)
        work[:] = dot(self.w, self.w) + 1e-99
        Enst = self.write_histogram(grp, work, 'Enst')[0]

        for i in range(3):
            Sii = self.fft.backward(iK[i] * U_hat[i])
            self.w[i] = Sii

        S2, S3 = self.write_histogram(grp, self.w, 'S_diag')[1:3]
        Skew = S3 / S2**1.5

        Re = sqrt(20/3) * k_r / sqrt(self.nu * eps_r)
        Lu = k_r**1.5 / (eps_r * 2*pi)

        vel = allsum(self.u)/(3 * self.num_points)
        div = allmax(np.abs(self.div(self.U_hat, out=work)))

        history = (f"{t_sim:< 15.7e} "
                   f"{k_r:< 15.7e} "
                   f"{eps_r:< 15.7e} "
                   f"{0.0:< 15.7e} "
                   f"{0.0:< 15.7e} "
                   f"{Enst:< 15.7e} "
                   f"{Skew:< 15.7e} "
                   f"{Re:< 15.7e} "
                   f"{Lu:< 15.7e} "
                   f"{self._dt_hydro:< 15.7e} "
                   f"{self._dt_diff:< 15.7e} "
                   f"{vel:< 15.7e} "
                   f"{div:< 15.7e} "
                   f"{self.nu:< 15.7e} "
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
        "#  5 = Resolved Enstrophy \n"
        "#  6 = Longitudinal Resolved Velocity Gradient Skewness \n"
        "#  7 = Taylor-scale Reynolds number (Re_lam)\n"
        "#  8 = Integral length scale (Lu/L = k_t**1.5 / (eps_t * L))\n"
        "#  9 = CFL timestep (dt_hydro)\n"
        "# 10 = von Neumann timestep (dt_diff)\n"
        "# 11 = Mean velocity (<U>)\n"
        "# 12 = Maximum magnitude of divergence [max(abs(div(u))]\n"
        "# 13 = Kinematic viscosity (nu)\n"
        "# 14 = Forcing On/Off (True/False)\n"
        "# ------------------------------------------------\n"
        )
