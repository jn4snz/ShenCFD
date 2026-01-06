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

from mpi4py import MPI

import os
import argparse
from copy import deepcopy

from math import sqrt, pi, ceil
import numpy as np
from shenfun import Array, Function, CompositeSpace
from shenfun.fourier import energy_fourier

from ..fourier_analysis import FourierAnalysis, fft3d
from ..file_io import h5FileIO
from ..utils import FileArgParser
from ..statistics import allsum, allmax  # , allmin
from ..maths import dot, smooth_bandpass

comm = MPI.COMM_WORLD


class NavierStokes(FourierAnalysis):

    def __init__(self, config=None, nvar=3, comm=comm):
        self.comm = comm

        if config is None:
            config = self.get_config()
        self.config = config

        self._nu = config.nu

        if config.dealiasing in ['3/2', 'iso3/2']:
            padding = 1.5
        else:
            padding = 1

        super().__init__([config.N]*3, padding, comm)

        if comm.rank == 0:
            os.makedirs(f'{config.odir}/restarts', exist_ok=True)

        self._Ufft = CompositeSpace([self.fft, ]*nvar)

        self.U_hat = Function(self.Ufft)   # fourier-space solution vector
        self.u_hat = Function(self.vfft, buffer=self.U_hat[:3])

        self.U = Array(self.Ufft)          # physical-space solution vector
        self.u = Array(self.vfft, buffer=self.U[:3])

        self.W_hat = Function(self.Ufft)   # fourier-space work vector
        self.w_hat = Function(self.vfft, buffer=self.W_hat[:3])

        self.W = Array(self.Ufft)          # physical-space work vector
        self.w = Array(self.vfft, buffer=self.W[:3])

        self._rhs_terms = []
        self._stage_updates = []
        self._step_updates = []

        self.set_dealiasing(config.dealiasing)
        self.configure_rhs()

        return

    @property
    def Ufft(self):
        return self._Ufft

    @property
    def nu(self):
        return self._nu

    def total_energy(self):
        self.w[0] = dot(self.u, self.u)
        return 0.5 * allsum(self.w[0]) / self.num_points

    @property
    def rhs_terms(self):
        return self._rhs_terms

    @property
    def stage_updates(self):
        return self._stage_updates

    @property
    def step_updates(self):
        return self._step_updates

    def compute_rhs(self, U_hat, dU):
        """Black box approach to RHS terms (e.g. convection, diffusion)

        Parameters
        ----------
        U_hat : type
            Description of parameter `U_hat`.
        dU : type
            Description of parameter `dU`.

        """
        dU[:] = 0.0
        for rhs_term in self.rhs_terms:
            rhs_term(U_hat, dU)

        return

    def stage_update(self, U_hat):
        """"Black box approach to non-RHS updates after each Runge-Kutta stage

        Parameters
        ----------
        U_hat : type
            Description of parameter `U_hat`.

        """
        self.Ufft.backward(U_hat, self.U)

        for update in self.stage_updates:
            update(U_hat)

        return

    def step_update(self, U_hat, t_sim, dU, dt):
        """Black box approach to non-RHS updates after each full time step

        Parameters
        ----------
        U_hat : type
            Description of parameter `U_hat`.
        dU : type
            Description of parameter `dU`.
        t_sim : type
            Description of parameter `t_sim`.
        dt : type
            Description of parameter `dt`.

        Returns
        -------
        new_dt
            Returns dynamic CFL-based time step

        """
        for update in self.step_updates:
            update(U_hat, t_sim, dU, dt)

        new_dt = self.compute_dt(dt, U_hat, dU)
        self.istep += 1

        return new_dt

    def configure_rhs(self):
        """Never forget that order matters when overloading `configure_rhs()`!

        """
        config = self.config

        # add nonlinear terms and dealiasing
        self.rhs_terms.append(self.rhs_rotation_convection)
        self.rhs_terms.append(self.rhs_dealias)

        # project off pressure from all nonlinear terms
        self.rhs_terms.append(self.rhs_pressure_poisson)

        # add linear terms
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

    def set_dealiasing(self, method):
        """

        """
        if method == 'nyquist':
            self.k_dealias = self.kmax[0]

        elif method == 'partial':
            self.k_dealias = int(sqrt(2) * self.N[0] / 3)

        elif method in ['2/3', 'iso2/3']:
            self.k_dealias = ceil(self.N[0] / 3)

        elif method in ['3/2', 'iso3/2']:
            self.k_dealias = self.kmax[0]

        else:
            self.k_dealias = int(method)

        if method in ['partial', 'iso2/3', 'iso3/2']:
            self.Kdealias = self.Ksq < self.k_dealias**2

        else:  # dealiasing in ['nyquist', '2/3', '3/2']:
            self.Kdealias = ((np.abs(self.K[0]) < self.k_dealias)
                             * (np.abs(self.K[1]) < self.k_dealias)
                             * (np.abs(self.K[2]) < self.k_dealias))

        return

    def rhs_dealias(self, U_hat, dU):
        dU *= self.Kdealias

        return dU

    def rhs_rotation_convection(self, U_hat, dU):
        """Incompressible Navier-Stokes advection in rotation form.

        Solution vectors can be any length, but only applies to first 3 terms,
        which must be the velocities.

        """
        iK = self.iK
        u = self.u
        w = self.w

        # take curl of velocity and inverse transform to get vorticity
        w[0] = self.fft.backward(iK[1] * U_hat[2] - iK[2] * U_hat[1])
        w[1] = self.fft.backward(iK[2] * U_hat[0] - iK[0] * U_hat[2])
        w[2] = self.fft.backward(iK[0] * U_hat[1] - iK[1] * U_hat[0])

        # take cross-product of vorticity and velocity and transform back
        # NOTE: u x w = - w x u (Since Du/Dt = du/dt + w x u)
        dU[0] += self.fft.forward(u[1] * w[2] - u[2] * w[1])
        dU[1] += self.fft.forward(u[2] * w[0] - u[0] * w[2])
        dU[2] += self.fft.forward(u[0] * w[1] - u[1] * w[0])

        return dU

    def rhs_linear_diffusion(self, U_hat, dU):
        """

        """
        dU[:3] -= self.nu * self.Ksq * U_hat[:3]

        return dU

    def rhs_pressure_poisson(self, U_hat, dU):
        """

        """
        dU[:3] += 1j * self.iK * dot(dU[:3], self._K_Ksq)

        return dU

    def rhs_spectral_forcing(self, U_hat, dU):
        """

        """
        u_hat = U_hat[:3]

        Ek_actual = self.compute_energy_spectrum(u_hat)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ek_forcing = self._fscale * np.sqrt(self._bandpass / Ek_actual)

        # EDGE CASE: on resized startup there may be more zeros in Ek_actual
        # than dealiased modes, and therefore Ek_forcing will still contain
        # infs or NaNs
        idx = np.where(~np.isfinite(Ek_forcing))
        Ek_forcing[idx] = 0.0

        dU[:3] += Ek_forcing[self.wavemodes] * u_hat

        return dU

    def compute_dt(self, old_dt, *ignore):
        """Unity CFL

        """
        dx = 2 * pi / self.nx[0]
        self._dt_diff = dx**2 / (6 * self.nu)

        u_max = allmax(np.sum(np.abs(self.u), axis=0))
        self._dt_hydro = dx / u_max

        dt = min(2.0 * old_dt, self._dt_hydro, self._dt_diff)

        return dt

    def update_spectral_forcing(self, U_hat, *ignore):
        """

        """
        u_hat = U_hat[:3]
        u = self.u
        w_hat = self.w_hat
        w = self.w

        Ek_actual = self.compute_energy_spectrum(u_hat)
        Ek_forcing = np.zeros_like(Ek_actual)
        with np.errstate(divide='ignore', invalid='ignore'):
            Ek_forcing = np.sqrt(self._bandpass / Ek_actual)

        # EDGE CASE: on resized startup there may be more zeros in Ek_actual
        # than dealiased modes, and therefore Ek_forcing will still contain
        # infs or NaNs
        idx = np.where(~np.isfinite(Ek_forcing))
        Ek_forcing[idx] = 0.0

        w_hat[:] = u_hat * Ek_forcing[self.wavemodes]
        self.vfft.backward(w_hat, w)

        size = self.num_points
        self._fscale = self.config.epsilon * size / allsum(u * w)

        return self._fscale

    def write_restart_to_file(self, U_hat, checkpoint=False):
        config = self.config

        if not checkpoint:
            f = f"{config.odir}/restarts/{config.pid}.{config.frame:03d}.h5"
            message = f" *** restart solution saved #{config.frame:03d}"

        else:
            f = f"{config.odir}/restarts/{config.pid}.checkpoint.h5"
            message = (" *** restart solution saved to checkpoint, "
                       "set frame = -2 in order to use.")

        with h5FileIO(f, 'w', self.comm) as fh:
            fh.write("U_hat", U_hat, self.Ufft, kwargs=vars(config))

        self.print(message)
        config.frame += 1

        return

    def initialize_from_restart(self, filename):
        # FIXME: this function might have broken edge cases related to a change
        # in dealiasing from 2/3 to 3/2 or vice versa.
        # Also, I'm not sure the h5_hat.refine() function perfectly truncates.

        fh = h5FileIO(filename, 'r', self.comm)
        nvf = len(fh.keys())
        nvar = len(self.Ufft)
        nv = min(nvf, nvar)

        N = self.nk[0]
        Np = fh["U_hat0"].shape[0]

        if nvf < nvar:
            self.print("Warning: restart file contains fewer solution fields "
                       "than requested for simulation!")

        elif nvf > nvar:
            self.print("Warning: restart file contains more solution "
                       "fields than requested for simulation!")

        if N < Np:  # We need to truncate the file data
            h5fft = fft3d([Np]*3, comm=self.comm)
            h5_hat = Function(h5fft)

            for i in range(nv):
                fh.read(f'U_hat{i}', h5_hat)
                h5_hat.refine(N, self.U_hat[i])  # refine also "de-refines"

            self.U_hat *= self.Kdealias
            self.Ufft.backward(self.U_hat, self.U)

        elif Np == N:  # just a regular restart
            fh.read("U_hat", self.U_hat)

            self.U_hat *= self.Kdealias
            self.Ufft.backward(self.U_hat, self.U)

        else:  # N > Np, we need to zero-pad the file data
            N = self.nx[0]  # need physical size, in case of 3/2 dealiasing
            self.all_assert(N % Np == 0)
            padding = N // Np
            h5fft = fft3d([Np]*3, padding=padding, comm=self.comm)
            nfft = CompositeSpace([h5fft, ]*nv)
            U_hat = Function(nfft)

            fh.read("U_hat", U_hat)

            nfft.backward(U_hat, self.U[:nv])
            self.Ufft.forward(self.U, self.U_hat)

        fh.close()

        if nvf < nvar:
            self.U_hat[nv:] = 0.0
            self.U[nv:] = 0.0

        return

    def initialize_taylor_green_vortex(self):
        """
        Generate the Taylor-Green vortex velocity initial condition.

        """
        X = self.fft.local_mesh(True)
        self.u[0] = np.sin(X[0]) * np.cos(X[1]) * np.cos(X[2])
        self.u[1] = -np.cos(X[0]) * np.sin(X[1]) * np.cos(X[2])
        self.u[2] = 0.0
        self.vfft.forward(self.u, self.u_hat)

        return

    def initialize_random_spectrum(self, energy, exponent, rseed=None):
        """
        Generate a random velocity field with a prescribed isotropic spectrum.

        """
        u_hat = self.u_hat
        u = self.u

        self.all_assert(isinstance(rseed, int) or rseed is None)
        rng = np.random.default_rng(rseed)

        # --------------------------------------------------------------
        # Give all wavenumbers a random uniform magnitude and phase
        u_hat[:] = rng.random(u_hat.shape, 'f8')
        theta = rng.random(u_hat.shape, 'f8')
        u_hat *= np.cos(2 * pi * theta) + 1j*np.sin(2 * pi * theta)

        # --------------------------------------------------------------
        # Parseval's Theorem Fix:
        #  Correct those wavenumbers that should be real-valued/Hermitian using
        #  a round-trip FFT. Not efficient, but this is one-time-only.
        #  Also possibly alters the uniform distribution?
        #     - Meh, seems to work.
        self.vfft.backward(u_hat, u)
        self.vfft.forward(u, u_hat)

        # --------------------------------------------------------------
        # Solenoidally-project before rescaling
        self.project_divergence_free(u_hat)

        # --------------------------------------------------------------
        # Re-scale to a simple energy spectrum
        nk = self.num_wavemodes
        kd = self.k_dealias
        delta = pi / self.kmax[0]
        k = np.arange(nk)
        k[0] = 1

        Ek_target = k**exponent * np.exp(-(1/6) * (4 * delta * k)**2)
        Ek_target[0] = 0.0
        Ek_target[kd:] = 0.0
        Ek_target *= energy / np.sum(Ek_target)

        Ek_actual = self.compute_energy_spectrum(u_hat)
        self.soft_assert(np.all(Ek_actual > 1e-20))

        with np.errstate(divide='ignore', invalid='ignore'):
            scaling = np.sqrt(Ek_target / Ek_actual)
        scaling[0] = 0.0     # remove the spatial mean
        scaling[kd:] = 0.0   # remove dealiased modes

        u_hat *= scaling[self.wavemodes]
        self.vfft.backward(u_hat, u)

        # --------------------------------------------------------------
        # Prove the Discrete Parseval's Theorem
        KE1 = 0.5 * allsum(u**2) / (energy * self.num_points)
        KE2 = 0.5 * energy_fourier(u_hat, self.fft) / energy
        KE3 = np.sum(self.compute_energy_spectrum(u_hat)) / energy
        self.print(f"Parseval's theorem check: "
                   f"KE1 = {KE1}, KE2 = {KE2}, KE3 = {KE3}\n", True)

        return

    parser = FileArgParser(
        fromfile_prefix_chars='@',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        allow_abbrev=False,
        argument_default=argparse.SUPPRESS)

    parser.add_argument(
        '--N', type=int, required=True,
        help='Physical-space mesh dimensions.')

    parser.add_argument(
        '--nu', type=float, required=True,
        help="kinematic viscosity")

    parser.add_argument(
        '--dealiasing', type=str.lower, default='3/2',
        choices=['nyquist', 'partial', 'iso2/3', '2/3', 'iso3/2', '3/2'],
        help='Fourier-space dealiasing method.')

    parser.add_argument(
        '--odir', type=str,
        help="Directory for simulation output files. Default is current "
             "working directory.")

    parser.add_argument(
        '--pid', type=str,
        help="Prefix for simulation output files. Default is basename "
             "of ODIR.")

    parser.add_argument(
        '-r', '--restart', nargs='?', default=False, const=True,
        help='Restart previous simulation. from one of two '
             'alternate options: 1) from given RESTART file path, '
             'or, 2) file path constructed from IDIR, '
             'INIT_PID, and FRAME.')

    parser.add_argument(
        '-f', '--frame', type=int,
        help='If RESTART argument omitted, restart from numbered output '
             'given by FRAME. Default is -1 if restart, else set to 0.')

    parser.add_argument(
        '--idir', type=str,
        help="Directory for input files. Default is ODIR")

    parser.add_argument(
        '--init_pid', type=str,
        help="Prefix for input files. Default is PID")

    parser.add_argument(
        '--t_init', type=float, default=0.0,
        help='starting simulation time')

    parser.add_argument(
        '--dt_init', type=float, default=0.0,
        help='starting simulation time. Default of 0.0 assumes a CFL based '
             'dt will be computed before the first timestep.')

    parser.add_argument(
        '--tlimit', type=float, default=np.inf,
        help='maximum simulation time')

    parser.add_argument(
        '--dt_rst', type=float, default=np.inf,
        help="time between HDF5 outputs")

    parser.add_argument(
        '--forcing', action='store_true', default=False,
        help="Enable isotropic forcing mechanism")

    parser.add_argument(
        '--no-forcing', dest='forcing', action='store_false',
        help="Disable isotropic forcing mechanism")

    parser.add_argument(
        '--epsilon', type=float, default=1.0,
        help="Energy dissipation rate")

    parser.add_argument(
        '--kfLow', type=int, default=3,
        help="spectral forcing passband start")

    parser.add_argument(
        '--kfHigh', type=int, default=4,
        help="spectral forcing passband end")

    parser.add_argument(
        '--init_cond', type=str.lower, default='random',
        choices=['random', 'taylor_green'],
        help="Use specified initial condition.")

    parser.add_argument(
        '--KE_init', type=float, default=3.4,
        help="Specify KE for random initial condition.")

    parser.add_argument(
        '--kExp_init', type=float, default=1/3,
        help='Specify infrared power-law for random IC.')

    def get_config(self):
        """Short summary.

        """

        if comm.rank == 0:
            # Clone the parser, but suppress all defaults to ensure that only
            # explicitly-provided options from the user override any options
            # loaded in from a restart file
            rst_parser = deepcopy(self.parser)
            for action in rst_parser._actions:
                action.required = False
                action.default = argparse.SUPPRESS

            # Now read in user-supplied arguments from the command line
            rconf = rst_parser.parse_known_args()[0]

            if hasattr(rconf, 'restart'):
                # Process user-inputs for initialization and restarting
                _resolve_io_arguments(rconf)

                if isinstance(rconf.restart, str):
                    # overwrite defaults to avoid unwanted behavior
                    rconf.init_file = rconf.restart
                    rconf.restart = True

                with h5FileIO(rconf.init_file, 'r', MPI.COMM_SELF) as fh:
                    args, _ = fh.parsable_attrs()

                # if this is a second restart from a legacy run, then
                # init_cond might be stored as 'restart', which is not an
                # allowed option for init_cond, which was stupid of me.
                args = [s for s in args if 'init_cond' not in s]
                config = self.parser.parse_known_args(args)[0]

                # Now override config with rconf options, except frame, that
                # way a value of -1 isn't carried forward.
                delattr(rconf, "frame")
                for k, v in vars(rconf).items():
                    setattr(config, k, v)

            else:
                # Re-parse command-line options with defaults un-suppressed.
                config = self.parser.parse_known_args()[0]
                _resolve_io_arguments(config)

        else:  # MPI.COMM_WORLD.rank > 0
            config = None

        comm.Barrier()
        config = comm.bcast(config, root=0)

        return config


def _resolve_io_arguments(config):
    """Check for existence of options `odir`, `pid`, `idir`, `init_pid`,
    `frame`, and set proper default values accordingly.

    """

    config.odir = getattr(config, 'odir', os.getcwd())
    config.pid = getattr(config, 'pid', os.path.basename(config.odir))
    config.idir = getattr(config, 'idir', config.odir)
    config.init_pid = getattr(config, 'init_pid',
                              os.path.basename(config.idir))
    config.frame = getattr(config, 'frame', -1)

    rdir = f"{config.idir}/restarts"
    prefix = f"{rdir}/{config.init_pid}"
    files = sorted(os.listdir(rdir))

    if config.frame == -2:
        config.init_file = f"{rdir}/{files[-1]}"

    elif config.frame == -1:
        if 'checkpoint' in files[-1]:
            config.init_file = f"{rdir}/{files[-2]}"
        else:
            config.init_file = f"{rdir}/{files[-1]}"

    else:
        config.init_file = (f"{prefix}.{config.frame:03d}.h5")

    return
