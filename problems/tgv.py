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
from mpi4py import MPI  # must always be imported first

import time
from math import pi

import numpy as np
from shenfun import Array, Function

import shencfd as cfd

comm = MPI.COMM_WORLD


class TaylorGreenVortex(cfd.FourierAnalysis):
    def __init__(self):
        super().__init__((64, 64, 64), padding=1.5, comm=comm)

        self.nu = 0.000625

        self.Kdealias = ???  # go back and check the assumed dealiasing!

        self.U_hat = Function(self.vfft)
        self.W_hat = Function(self.vfft)
        self.U = Array(self.vfft)
        self.W = Array(self.vfft)

        self.dx = pi / self.kmax[0]
        self.dt_diff = self.dx**2 / (6 * self.nu)
        self.istep = 0

    def compute_rhs(self, U_hat, dU):
        iK = self.iK
        U = self.U
        W = self.W

        # velocity advection
        W[0] = self.fft.backward(iK[1] * U_hat[2] - iK[2] * U_hat[1])
        W[1] = self.fft.backward(iK[2] * U_hat[0] - iK[0] * U_hat[2])
        W[2] = self.fft.backward(iK[0] * U_hat[1] - iK[1] * U_hat[0])

        dU[0] = self.fft.forward(U[1] * W[2] - U[2] * W[1])
        dU[1] = self.fft.forward(U[2] * W[0] - U[0] * W[2])
        dU[2] = self.fft.forward(U[0] * W[1] - U[1] * W[0])

        dU -= self.nu * self.Ksq * U_hat

        self.project_divergence_free(dU)
        dU *= self.Kdealias

    def stage_update(self, U_hat):
        self.vfft.backward(U_hat, self.U)

    def step_update(self, U_hat, dU, t_sim, dt):
        fmt = cfd.enf
        KE = 0.5 * cfd.allsum(self.U**2) / self.num_points
        self.print(f"{self.istep:3d}, {fmt(t_sim)}, {fmt(dt)}, {fmt(KE, 9)}")

        u_max = cfd.allmax(np.sum(np.abs(self.U), axis=0))
        new_dt = min(2.0 * dt, 0.9 * self.dx / u_max, self.dt_diff)

        self.istep += 1

        return new_dt

    def initialize_taylor_green_vortex(self):
        X = self.fft.local_mesh(True)
        self.U[0] = np.sin(X[0]) * np.cos(X[1]) * np.cos(X[2])
        self.U[1] = -np.cos(X[0]) * np.sin(X[1]) * np.cos(X[2])
        self.U[2] = 0.0
        self.vfft.forward(self.U, self.U_hat)

        return


###############################################################################
def run():
    """Homogeneous Isotropic Turbulence (HIT) CFD simulation.

    """
    t0 = time.perf_counter()

    sim = TaylorGreenVortex()
    sim.initialize_taylor_green_vortex()

    sol = cfd.RK4_integrator(sim, sim.U_hat)
    sol.integrate([0, 0.1])

    KE = 0.5 * cfd.allsum(sim.U**2) / sim.num_points
    sim.soft_assert(np.round(KE - 0.124953117517, 9) == 0, cfd.enf(KE))

    sim.print(f"\n Runtime = {time.perf_counter() - t0}")


if __name__ == "__main__":
    run()
    # import cProfile
    # cProfile.run(isotropic_turbulence())
