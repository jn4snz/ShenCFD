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

"""[summary]

[description]

"""
from mpi4py import MPI  # must always be imported first in top-level script

import gc
import time

import numpy as np

from shenfun import Function, Array
import shencfd as cfd

comm = MPI.COMM_WORLD

###############################################################################
if comm.rank == 0:
    print("\n----------------------------------------------------------\n"
          "MPI-parallel Python shenCFD postprocessing for problem \n"
          "         'Homogeneous Isotropic Turbulence' \n"
          f"started with {comm.size} tasks at {time.strftime('%H:%M:%S')}."
          "\n----------------------------------------------------------\n",
          flush=True)

wt0 = time.perf_counter()

# -----------------------------------------------------------------
K = 1024
Np = 1024
kfLow = 3
kfHigh = 4

WORK3 = "/lustre/scratch3/turquoise/towery"
WORK4 = "/lustre/scratch4/turquoise/towery"
pid = f"DNS_K{K}_N{Np}_F{kfLow}k{kfHigh}"
ddir = f"{WORK4}/DNS/hybrid_ICs"
frame = 55

kf = 2**np.arange(11)
kf = np.sort(np.r_[3/4 * kf[1:], kf])
filter_sizes = kf[kf < Np/2]
ftype = 'gaussian'
ratios = (1.0, 4/3, 1.5, 5/3, 2.0, 7/3, 8/3, 3.0)

# --------------------------------------------------------------
# Loop over filter width
for kf in filter_sizes:
    Ns = np.unique(np.ceil(kf*ratios*1.5))

    filename = f'{ddir}/hybrid_IC_N{Np}_k{int(kf)}_{ftype}.{frame:03d}.h5'
    fh = cfd.h5FileIO(filename, 'r', comm)
    Np = fh["U_hat0"].shape[0]

    dns = cfd.fft3d([Np]*3, comm=comm)
    Z_hat = Function(dns)
    Z = Array(dns)

    for N in Ns:
        ratio = N / (kf * 1.5)
        dns.all_assert(Np >= N)
        dns.all_assert(Np % N == 0)
        skip = Np // N
        sample = (slice(None, None, skip), )*3

        sim = cfd.FourierAnalysis([N]*3, padding=1.5, comm=comm)
        sim.u_hat = Function(sim.vfft)
        sim.k_hat = Function(sim.fft)
        sim.eps_hat = Function(sim.fft)
        sim.Kdealias = sim.fft.get_mask_nyquist()

        sim.u = Array(sim.vfft)
        sim.k = Array(sim.fft)
        sim.eps = Array(sim.fft)
        sim.work = np.empty_like(sim.k)

        # get an unpadded transform at the simulation mesh size
        rfft = cfd.fft3d([N]*3, comm=comm)
        shape = [2, *rfft.forward.input_array.shape]
        Y = np.empty(shape, dtype='f8')

        # --------------------------------------------------------------
        # Read in and truncate the pre-processed DNS data
        K_m = fh.attrs['K_m']
        K_r = fh.attrs['K_r']
        E_m = fh.attrs['E_m']
        E_r = fh.attrs['E_r']
        Fk = K_m / (K_m + K_r)
        Fe = E_m / (E_m + E_r)

        #  -- spectral truncation of u preserves divergence-free condition
        fh.read("U_hat", sim.u_hat)
        sim.u_hat *= sim.Kdealias
        sim.vfft.backward(sim.u_hat, sim.u)

        # -- physical-space sub-sampling of k-eps preserves positivity
        fh.read("U_hat3", Z_hat)
        dns.backward(Z_hat, Z)
        Y[0] = Z[sample]

        fh.read("U_hat4", Z_hat)
        dns.backward(Z_hat, Z)
        Y[1] = Z[sample]

        fh.close()

        # --------------------------------------------------------------
        # test the quality of the truncated initial condition
        sim.work[:] = cfd.dot(sim.u, sim.u)
        k_r = 0.5 * cfd.allsum(sim.work) / sim.num_points

        sim.work[:] = sim.strain_squared(sim.u_hat, out=sim.work)
        e_r = sim.nu * cfd.allsum(sim.work) / sim.num_points

        k_m = cfd.allsum(Y[0]) / np.prod(sim.N)
        e_m = cfd.allsum(Y[1]) / np.prod(sim.N)
        k_min = cfd.allmin(Y[0])
        e_min = cfd.allmin(Y[1])

        fk = k_m / (k_m + k_r)
        fe = e_m / (e_m + e_r)
        err0 = abs(Fk - fk) / Fk
        err1 = abs(Fe - fe) / Fe

        # --------------------------------------------------------------
        # get the dealiased k-eps values and check quality again
        Y = np.log(Y)

        rfft.forward(Y[0], sim.k_hat)
        sim.k_hat *= sim.Kdealias
        sim.fft.backward(sim.k_hat, sim.k)
        sim.k = np.exp(sim.k)

        rfft.forward(Y[1], sim.eps_hat)
        sim.eps_hat *= sim.Kdealias
        sim.fft.backward(sim.eps_hat, sim.eps)
        sim.eps = np.exp(sim.eps)

        km2 = cfd.allsum(sim.k) / sim.num_points
        em2 = cfd.allsum(sim.eps) / sim.num_points

        fk = km2 / (km2 + k_r)
        fe = em2 / (em2 + e_r)
        err2 = abs(Fk - fk) / Fk
        err3 = abs(Fe - fe) / Fe

        err_str = (f'{N}, {ftype}, {ratio:0.2f}, {Fk:0.3e}, {Fe:0.3e}, '
                   f'{err0:0.3e}, {err1:0.3e}, {err2:0.3e}, {err3:0.3e}, '
                   f'{k_min:0.3e}, {e_min:0.3e},')
        sim.print(err_str, True)

        del Y, sim.work, sim.eps, sim.k, sim.u, sim.Kdealias
        del sim.eps_hat, sim.k_hat, sim.u_hat
        del rfft, sim
        gc.collect()

# -----------------------------------------------------------------
comm.Barrier()
seconds = time.perf_counter() - wt0
minutes, rem_seconds = divmod(int(seconds), 60)
hours, minutes = divmod(minutes, 60)
dns.print(
    "\n----------------------------------------------------------\n"
    "Initialization Walltime (H:M:S) = "
    f"{hours}:{minutes:02d}:{rem_seconds:02d}\n"
    "`Homogeneous Isotropic Turbulence' simulation [init-only]\n"
    f"finished at {time.strftime('%H:%M:%S')}.\n"
    "----------------------------------------------------------", True)
