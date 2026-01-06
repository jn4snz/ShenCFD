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

import os
import gc
from math import pi, log2
import time
import subprocess

import numpy as np

from shenfun import Function, Array, CompositeSpace
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
Np = 2048
kfLow = 4
kfHigh = 6
frame = 49

WORK = os.getenv('WORK')
pid = f"DNS_K{K}_N{Np}_F{kfLow}k{kfHigh}"
ddir = f"{WORK}/DNS/{pid}/restarts"
odir = f"{WORK}/DNS/hybrid_ICs"
if comm.rank == 0:
    os.makedirs(odir, exist_ok=True)

eps = 1.0
eta = 2 / K
nu = (eps * eta**4)**(1/3)

filter_sizes = 2**np.arange(9)

# --------------------------------------------------------------
# get the proper Fourier analysis context for the DNS data
iofft = cfd.fft3d([Np]*3, padding=1.5)
io_hat = Function(iofft)

dns = cfd.FourierAnalysis([int(Np * 1.5)]*3)
dns.Ufft = CompositeSpace([dns.fft, ]*5)

dns.v_hat = Function(dns.vfft)
dns.v = Array(dns.vfft)

dns.u_hat = Function(dns.vfft)
dns.U = Array(dns.Ufft)
dns.u = Array(dns.vfft, buffer=dns.U[:3])
dns.k = dns.U[3]
dns.eps = dns.U[4]

filename = f'{ddir}/{pid}.{frame:03d}.h5'

fh = cfd.h5FileIO(filename, 'r', comm)
for i in range(3):
    fh.read(f"U_hat{i}", io_hat)
    iofft.backward(io_hat, dns.v[i])
t_init = fh.attrs['t_init']
fh.close()
dns.vfft.forward(dns.v, dns.v_hat)

del iofft, io_hat
gc.collect()

N = 128
fft = cfd.fft3d([N]*3, padding=dns.nx[0] / N)
Ufft = CompositeSpace([fft, ]*5)
U_hat = Function(Ufft)
Kdealias = fft.get_mask_nyquist()

# --------------------------------------------------------------
# Loop over filter width
for kf in filter_sizes:
    N_old = N
    N = min(dns.nx[0], max(128, int(kf * 12)))
    if N > N_old:
        del U_hat, Ufft, fft
        gc.collect()

        fft = cfd.fft3d([N]*3, padding=dns.nx[0] / N)
        Ufft = CompositeSpace([fft, ]*5)
        U_hat = Function(Ufft)
        Kdealias = fft.get_mask_nyquist()

    nstripes = round(log2(N**3/64**3 + 1))
    if comm.rank == 0:
        subprocess.run(['lfs', 'setstripe', '-c', str(nstripes), odir])

    dns.print(f'processing kf = {kf}', True)
    delta = pi / kf
    G_hat = dns.analytic_gaussian_filter(delta)

    # --------------------------------------------------------------
    # filter the DNS velocity

    dns.u_hat[:] = dns.v_hat * G_hat
    dns.vfft.backward(dns.u_hat, dns.u)

    dns.print(f'dns.u min = {cfd.allmin(dns.u)}, '
              f'dns.u max = {cfd.allmax(dns.u)}', True)

    dns.k[:] = cfd.dot(dns.u, dns.u)
    K_r = 0.5 * cfd.allsum(dns.k) / dns.num_points

    dns.eps[:] = dns.strain_squared(dns.u_hat, out=dns.eps)
    E_r = nu * cfd.allsum(dns.eps) / dns.num_points

    # --------------------------------------------------------------
    # compute the SFS kinetic energy
    dns.k[:] = 0.5 * cfd.dot(dns.v, dns.v)
    dns.k[:] = dns.filter(dns.k, G_hat)
    dns.k -= 0.5 * cfd.dot(dns.u, dns.u)

    K_m = cfd.allsum(dns.k) / dns.num_points
    k_min = cfd.allmin(dns.k)

    # --------------------------------------------------------------
    # compute the SFS viscous dissipation
    dns.eps[:] = dns.strain_squared(dns.v_hat, out=dns.eps)
    dns.eps[:] = dns.filter(dns.eps, G_hat)
    dns.eps -= dns.strain_squared(dns.u_hat)
    dns.eps *= nu

    E_m = cfd.allsum(dns.eps) / dns.num_points
    e_min = cfd.allmin(dns.eps)

    # --------------------------------------------------------------
    # correct for negative values at large k_f
    if k_min < 1e-99 or e_min < 1e-99:
        dns.print('WARNING: k or epsilon non-positive! re-filtering.'
                  f'{k_min}, {e_min}')
        dns.k[:] = dns.filter(dns.k, G_hat)
        dns.eps[:] = dns.filter(dns.eps, G_hat)

        k_min = cfd.allmin(dns.k)
        e_min = cfd.allmin(dns.eps)
        dns.print(f'Re-filtered k_min, eps_min = {k_min}, {e_min}')

    # --------------------------------------------------------------
    # save the filtered/SGS fields to file in spectral space
    Ufft.forward(dns.U, U_hat)
    U_hat *= Kdealias

    kwargs = dict(t_init=t_init, K_r=K_r, E_r=E_r, K_m=K_m, E_m=E_m)
    filename = f'{odir}/hybrid_IC_K{K}_k{int(kf)}.{frame:03d}.h5'
    with cfd.h5FileIO(filename, 'w', comm) as fh:
        fh.write('U_hat', U_hat, kwargs=kwargs)

# -----------------------------------------------------------------
comm.Barrier()
seconds = time.perf_counter() - wt0
minutes, rem_seconds = divmod(int(seconds), 60)
hours, minutes = divmod(minutes, 60)
dns.print(
    "\n----------------------------------------------------------\n"
    "Postprocessing Walltime (H:M:S) = "
    f"{hours}:{minutes:02d}:{rem_seconds:02d}\n"
    "`Homogeneous Isotropic Turbulence' simulation [init-only]\n"
    f"finished at {time.strftime('%H:%M:%S')}.\n"
    "----------------------------------------------------------", True)
