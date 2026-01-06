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
from math import pi
import time

import numpy as np
import h5py
from shenfun import Function
import shencfd as cfd

comm = MPI.COMM_WORLD


def analyze(fh, data, key, log=False):
    # get moments before taking log
    m = cfd.moments(data)

    # take log10 if desired
    if log:
        xmin, xmax = m[-2:]
        if xmin <= 0.0:
            if comm.rank == 0:
                print(f"Warning: {key} has negative xmin = {xmin}")
            xmin = 1e-99
            data[data <= 0.0] = 1e-99

        data = np.log10(data)
        xrange = tuple(np.log10((xmin, xmax)))

    else:
        xrange = m[-2:]

    hist, edges = cfd.histogram1(data, bounds=xrange)

    if comm.rank == 0:
        grp = fh.create_group(key)
        grp['hist'] = hist
        grp['edges'] = edges
        grp.attrs['moments'] = m[:-2]
        grp.attrs['range'] = m[-2:]    # note I store non-log range here
        grp.attrs['log'] = log

    return


###############################################################################
wt0 = time.perf_counter()

if comm.rank == 0:
    print("\n----------------------------------------------------------\n"
          "MPI-parallel Python shenCFD postprocessing for problem \n"
          "         'Homogeneous Isotropic Turbulence' \n"
          f"started with {comm.size} tasks at {time.strftime('%H:%M:%S')}."
          "\n----------------------------------------------------------\n",
          flush=True)

# -----------------------------------------------------------------
# Preliminary setup
K = 1024
Nl = 2048
Np = 2048
kfLow = 4
kfHigh = 6
frame = 49

WORK = os.getenv('WORK')
pid = f"DNS_K{K}_N{Nl}_F{kfLow}k{kfHigh}"
ddir = f"{WORK}/DNS/{pid}/restarts"

files = sorted(os.listdir(ddir))
if 'checkpoint' in files[-1]:
    files = files[:-1]

eps = 1.0
eta = 2 / K
nu = (eps * eta**4)**(1/3)

filter_width = 2**np.arange(9)

stats_file = f'{WORK}/DNS/{pid}/{pid}-filtered_statistics.h5'
fh = None
tgrp = None
kgrp = None
if comm.rank == 0:
    fh = h5py.File(stats_file, 'w', driver='core', block_size=1048576)

# -----------------------------------------------------------------
# Initialize analysis methods and fields
# iofft = cfd.fft3d([Np]*3, padding=1.5)
# io_hat = Function(iofft)

# dns = cfd.FourierAnalysis([int(Np * 1.5)]*3)
dns = cfd.FourierAnalysis([Np]*3)
iK = dns.iK

v_hat = Function(dns.vfft)
u_hat = np.empty_like(v_hat)
w_hat = np.empty_like(v_hat)

V = np.empty((5, *dns.r.shape), dtype=dns.r.dtype)
v = V[:3]
k_t = V[3]
eps_t = V[4]

u = np.empty((3, *dns.r.shape), dtype=dns.r.dtype)
w = np.empty((3, *dns.r.shape), dtype=dns.r.dtype)
work = np.empty_like(dns.r)
P = np.empty_like(dns.r)

# =============================================================================
# DNS Restart File Loop
for filename in files:
    dns.print(f'processing file {filename}', True)
    frame = int(filename.split('.')[1])
    # if frame < ifs or frame >= ife:
    #     continue

    # -----------------------------------------------------------------
    # read in DNS data
    data = cfd.h5FileIO(f'{ddir}/{filename}', 'r', dns.comm)
    for i in range(3):
        # data.read(f"U_hat{i}", io_hat)
        # iofft.backward(io_hat, v[i])
        # dns.fft.forward(v[i], v_hat[i])
        data.read(f"U_hat{i}", v_hat[i], dns.fft)
        dns.fft.backward(v_hat[i], v[i])

    if comm.rank == 0:
        tgrp = fh.create_group(f'{frame:03d}')
        tgrp.attrs['time'] = data.attrs['t_init']

    data.close()

    k_t[:] = 0.5 * cfd.dot(v, v)
    eps_t[:] = nu * dns.strain_squared(v_hat, out=eps_t)

    # =========================================================================
    # Filter Wavenumber Loop
    for kf in filter_width:
        dns.print(f'    k_f = {kf}', True)
        if comm.rank == 0:
            kgrp = tgrp.create_group(str(int(kf)))

        delta = pi / kf
        G_hat = dns.analytic_gaussian_filter(delta)

        # ----------------------------------------------------------------
        # filter the DNS velocity
        u_hat = v_hat * G_hat
        dns.vfft.backward(u_hat, u)

        # ----------------------------------------------------------------
        # compute the SFS kinetic energy
        work[:] = 0.5 * cfd.dot(u, u)
        analyze(kgrp, work, 'k_r', log=True)

        dns.r[:] = k_t
        work[:] = dns.filter(dns.r, G_hat) - work
        analyze(kgrp, work, 'k_m', log=True)

        dns.print('        k_r/k_m done', True)

        # ----------------------------------------------------------------
        # compute the SFS viscous dissipation
        work[:] = nu * dns.strain_squared(u_hat, out=work)
        analyze(kgrp, work, 'eps_r', log=True)

        dns.r[:] = eps_t
        work[:] = dns.filter(dns.r, G_hat) - work
        analyze(kgrp, work, 'eps_m', log=True)

        dns.print('        eps_r/eps_m done', True)

        # ----------------------------------------------------------------
        # compute Production and diagonal terms of S_ij
        P[:] = 0.0
        for i in range(3):
            for j in range(i, 3):
                # exact tau = <ui uj> - <ui> <uj>
                work[:] = v[i] * v[j]
                work[:] = dns.filter(work, G_hat)
                work -= u[i] * u[j]

                # Sij -> dns.r
                Sij = dns.fft.backward(iK[j] * u_hat[i] + iK[i] * u_hat[j])

                # cross-scale transfer aka "Production"
                P -= 0.5 * (1 + (i != j)) * work * Sij

                # store S11, S22, S33 for Skewness calculation
                if i == j:
                    w[i] = Sij

        analyze(kgrp, P, 'prod', log=False)
        analyze(kgrp, w, 'S_diag', log=False)

        dns.print('        Prod/S_diag done', True)
        if comm.rank == 0:
            fh.flush()

        # ----------------------------------------------------------------
        # compute resolved enstrophy
        dns.curl(u_hat, w)
        work[:] = cfd.dot(w, w)
        analyze(kgrp, work, 'enst_r', log=True)

        # ----------------------------------------------------------------
        # compute triangular-upper terms of tau_ij and div(tau)
        w_hat[:] = 0.0
        k = 0
        for i in range(2):
            for j in range(i+1, 3):
                # exact tau = <ui uj> - <ui> <uj>
                w[k] = v[i] * v[j]
                w[k] = dns.filter(w[k], G_hat)
                w[k] -= u[i] * u[j]

                # div(tau)
                tau_ij_hat = dns.fft.forward(w[k])
                w_hat[i] += iK[j] * tau_ij_hat
                w_hat[j] += iK[i] * tau_ij_hat

                k += 1

        analyze(kgrp, w, 'tau_triu', log=False)

        # ----------------------------------------------------------------
        # compute diagonal terms of tau_ij and div(tau)
        for i in range(3):
            # exact tau = <ui ui> - <ui> <ui>
            w[i] = v[i] * v[i]
            w[i] = dns.filter(w[i], G_hat)
            w[i] -= u[i] * u[i]

            # div(tau)
            tau_ii_hat = dns.fft.forward(w[i])
            w_hat[i] += iK[i] * tau_ii_hat

        analyze(kgrp, w, 'tau_diag', log=False)

        dns.vfft.backward(w_hat, w)
        analyze(kgrp, w, 'force', log=False)

        # if comm.rank == 0:
        #     fh.flush()

# -----------------------------------------------------------------
if comm.rank == 0:
    fh.flush()
    fh.close()

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
