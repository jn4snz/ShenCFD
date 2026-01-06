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

"""

"""
from mpi4py import MPI

from math import fsum, sqrt

import numpy as np

from ..maths import dot


def update_smag_viscosity(self, U_hat, *ignore):
    """Computes nuT for all Smagorinsky models. Used at the end of each RK4
    step for statistical outputs and updating the dynamic dt.

    """
    S = self.strain_squared(U_hat[:3], out=self.work[0])
    S = np.sqrt(S)
    self.nuT[:] = self.Cs * self.Delta2 * S

    return


def update_dyn_smag_coeff(self, U_hat, *ignore):
    """Compute Cs using Germano-Lilly proceduce for dynamic Smagorinsky.

    51 total scalar FFTs in function.
    """
    config = self.config
    iK = self.iK
    n2 = (config.test_ratio/config.ic_ratio)**2

    # pointing working memory to readable names
    Sij_hat = self.W_hat[0]
    S = self.work[0]
    St = self.work[1]
    Mij = self.work[2]
    Lij = self.work[3]
    LijMij = self.work[4]
    MijMij = self.work[5]

    self.w_hat[:] = self.Ktest * U_hat[:3]
    self.vfft.backward(self.w_hat, self.w)

    S[:] = 0.0
    St[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij_hat[:] = iK[j] * U_hat[i] + iK[i] * U_hat[j]
            Sij = self.fft.backward(Sij_hat)  # == 2Sij
            S += (1 + (i != j)) * Sij**2  # == 4SijSij

            Sij_hat *= self.Ktest
            Sij = self.fft.backward(Sij_hat)  # == 2Sij
            St += (1 + (i != j)) * Sij**2  # == 4SijSij

    S[:] = np.sqrt(0.5 * S)    # == sqrt(2 Sij Sij)
    St[:] = np.sqrt(0.5 * St)  # == sqrt(2 Sij Sij)

    LijMij[:] = 0.0
    MijMij[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij_hat[:] = iK[j] * U_hat[i] + iK[i] * U_hat[j]
            Mij[:] = S * self.fft.backward(Sij_hat)
            Mij[:] = self.filter(Mij, self.Ktest)
            Mij -= n2 * St * self.fft.backward(self.Ktest * Sij_hat)
            Mij *= self.Delta2

            Lij[:] = self.filter(self.u[i] * self.u[j], self.Ktest)
            Lij[:] -= self.w[i] * self.w[j]

            LijMij += (1 + (i != j)) * Lij * Mij  # note sign
            MijMij += (1 + (i != j)) * Mij**2

    buffer = np.empty(2)
    buffer[0] = fsum(LijMij.flat)  # 0.5 needed for Sij
    buffer[1] = fsum(MijMij.flat)
    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

    self.Cs = buffer[0] / buffer[1]

    # quality check
    if self.Cs < 0.0:
        self.print(f'LijMij = {buffer[0]:0.3e}; MijMij = {buffer[1]:0.3e}; '
                   f'Cs = {self.Cs:0.3e}')
        # typically Cs gets very small before going negative, so this just
        # flips a small negative into a small positive
        self.Cs = np.abs(self.Cs)

    self.nuT[:] = self.Cs * self.Delta2 * S

    return


def rhs_smagorinsky(self, U_hat, dU):
    """Computes the Large-Eddy Equation RHS for Smagorinsky models.

    18 total scalar FFTs in function.
    """
    iK = self.iK
    Cs = self.Cs

    # S is an alias to work[0]
    S = self.strain_squared(U_hat[:3], out=self.work[0])
    S = np.sqrt(S)

    for i in range(3):
        for j in range(i, 3):
            Sij = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])
            Rij_hat = self.fft.forward(Cs * self.Delta2 * S * Sij)

            dU[i] += iK[j] * Rij_hat
            dU[j] += (i != j) * iK[i] * Rij_hat

    return dU


def update_ksgs_viscosity(self, U_hat, *ignore):
    """Computes nuT for all k-SGS models. Used at the end of each RK4
    step for statistical outputs and updating the dynamic dt.

    """
    if self.config.log_ke:
        self.nuT[:] = self.Cs * self.Delta * np.exp(0.5*self.U[3])

    else:
        self.nuT[:] = self.Cs * self.Delta * self.U[3]**0.5

    return


def update_dynk_coeff(self, U_hat, *ignore):
    """Compute Cs using Germano-Lilly proceduce for dynamic k-SGS.

    41 total scalar FFTs in function.
    """
    config = self.config
    iK = self.iK
    n = config.test_ratio / config.ic_ratio

    # pointing working memory to readable names
    Sij_hat = self.W_hat[3]

    Mij = self.work2[0]
    Lij = self.work2[1]
    LijMij = self.work2[2]
    MijMij = self.work
    k_f = self.W[3]

    if config.log_ke:
        k = self.nuT[:] = np.exp(self.U[3])
    else:
        k = self.U[3]

    self.w_hat[:] = self.Ktest * U_hat[:3]
    self.vfft.backward(self.w_hat, self.w)

    k_f[:] = k + 0.5 * dot(self.u, self.u)
    k_f[:] = self.filter(k_f, self.Ktest)
    k_f -= 0.5 * dot(self.w, self.w)

    LijMij[:] = 0.0
    MijMij[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij_hat[:] = iK[j] * U_hat[i] + iK[i] * U_hat[j]
            Mij[:] = k**0.5 * self.fft.backward(Sij_hat)
            Mij[:] = self.filter(Mij, self.Ktest)
            Mij[:] -= n * k_f**0.5 * self.fft.backward(self.Ktest * Sij_hat)
            Mij *= self.Delta

            Lij[:] = self.filter(self.u[i] * self.u[j], self.Ktest)
            Lij -= self.w[i] * self.w[j]

            LijMij += (1 + (i != j)) * Lij * Mij  # note sign
            MijMij += (1 + (i != j)) * Mij**2

    buffer = np.empty(2)
    buffer[0] = fsum(LijMij.flat)  # 0.5 needed for Sij
    buffer[1] = fsum(MijMij.flat)
    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

    self.Cs = buffer[0] / buffer[1]

    return


def update_ldmk_coeff(self, U_hat, *ignore):
    """Compute Cs using Heinz-Mokhtarpoor proceduce for dynamic k-SGS.

    27 total scalar FFTs in function.
    """
    config = self.config
    iK = self.iK
    n = config.test_ratio / config.ic_ratio
    Cs = self.Cs

    # pointing working memory to readable names
    Sij_hat = self.W_hat[3]
    S = self.W[3]

    Mij = self.work2[0]
    Lij = self.work2[1]
    LijMij = self.work2[2]
    MijMij = self.work

    if config.log_ke:
        k = self.nuT[:] = np.exp(self.U[3])
    else:
        k = self.U[3]

    self.w_hat[:] = self.Ktest * U_hat[:3]
    self.vfft.backward(self.w_hat, self.w)

    # "pdf-realizability"
    S[:] = 0.0
    LijMij[:] = 0.0
    MijMij[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij_hat[:] = iK[j] * U_hat[i] + iK[i] * U_hat[j]
            Sij = self.fft.backward(Sij_hat)  # == 2Sij
            S += (1 + (i != j)) * Sij**2  # == 4SijSij

            Mij[:] = n * self.Delta * self.fft.backward(self.Ktest * Sij_hat)
            Lij[:] = self.filter(self.u[i] * self.u[j], self.Ktest)
            Lij -= self.w[i] * self.w[j]

            LijMij += (1 + (i != j)) * Lij * Mij
            MijMij += (1 + (i != j)) * Mij**2

    S[:] = np.sqrt(0.5 * S)  # == sqrt(2 Sij Sij)
    Cs[:] = - LijMij / MijMij  # note sign

    # "stress-realizability"
    limit = self.work
    limit[:] = 23 * k**0.5 / (24 * sqrt(3) * self.Delta * S)
    Cs[:] = np.where(Cs < -limit, -limit, Cs)
    Cs[:] = np.where(Cs > limit, limit, Cs)

    return


def rhs_k_sgs(self, U_hat, dU):
    """Computes the Large-Eddy Equation RHS for k-SGS models.

    19 total scalar FFTs in function.
    """
    iK = self.iK
    k_m = self.U[3]
    km_hat = U_hat[3]
    nuT = self.nuT
    Ssq = self.W[3]
    work = self.work

    nuT[:] = self.Cs * self.Delta * k_m**0.5

    # --------------------------------------------------------------
    # Add Reynolds stress to velocity RHS
    # --------------------------------------------------------------
    Ssq[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])
            Ssq += (1 + (i != j)) * Sij**2
            Rij_hat = self.fft.forward(nuT * Sij)

            dU[i] += iK[j] * Rij_hat
            dU[j] += (i != j) * iK[i] * Rij_hat

    Ssq *= 0.5  # == 2SijSij

    # --------------------------------------------------------------
    # Compute RHS of k-transport
    # --------------------------------------------------------------
    # compute turbulent transport flux in physical space
    self.w_hat[:] = iK * km_hat
    self.vfft.backward(self.w_hat, self.w)
    self.w *= self.sigk_inv*nuT + self.nu

    # add advective flux and transform back to Fourier space
    self.w -= self.u * k_m
    self.vfft.forward(self.w, self.w_hat)

    # take the divergence of the advective-diffusive flux
    dU[3] = dot(iK, self.w_hat)

    # add production and dissipation
    # Production has roughly cubic aliasing (maybe its power 2.5 aliasing?)
    # Dissipation has roughly standard (quadratic) aliasing (power 1.5 v 2)
    work[:] = nuT * Ssq - (self.Ce / self.Delta) * k_m**1.5
    dU[3] += self.fft.forward(work) * self.Ksmooth

    return


def rhs_logk_sgs(self, U_hat, dU):
    """Compute the LES RHS for k-SGS using log(k) as solution variable.

    19 total scalar FFTs in function.
    """
    # point allocated memory to descriptive names
    iK = self.iK
    Cs = self.Cs
    Df = self.Delta

    lnk = self.U[3]
    lnk_hat = U_hat[3]
    nuT = self.nuT
    work = self.work
    Sij = self.work
    Ssq = self.W[3]

    # compute physical-space variables
    nuT[:] = Cs * Df * np.exp(0.5*lnk)

    # --------------------------------------------------------------
    # Add Reynolds stress to velocity RHS
    # --------------------------------------------------------------
    Ssq[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij[:] = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])
            Ssq += (1 + (i != j)) * Sij**2
            Sij *= nuT
            Rij_hat = self.fft.forward(Sij)
            dU[i] += iK[j] * Rij_hat
            dU[j] += (i != j) * iK[i] * Rij_hat

    Ssq *= 0.5  # == 2SijSij

    # --------------------------------------------------------------
    # Compute RHS of k-transport
    # --------------------------------------------------------------
    # compute grad(lnk)
    self.w_hat[:] = iK * lnk_hat
    self.vfft.backward(self.w_hat, self.w)

    # compute lnk source terms
    work[:] = (self.sigk_inv*nuT + self.nu) * dot(self.w, self.w)
    work += Cs * Df * np.exp(-0.5*lnk) * Ssq  # production
    work -= (self.Ce / Df) * np.exp(0.5*lnk)  # dissipation
    dU[3] = self.fft.forward(work) * self.Ksmooth

    # compute conservative turbulent diffusion and advective fluxes
    self.w *= self.sigk_inv*nuT + self.nu
    self.w -= self.u * lnk
    self.vfft.forward(self.w, self.w_hat)

    # take the divergence of the advective-diffusive flux
    dU[3] += dot(iK, self.w_hat)


def update_lns_coeffs(self, U_hat, *ignore):
    """Compute Cmu for Limited Numerical Scales model using static Smagorinsky
    and k-epsilon RANS.

    This function was written assuming unified hybridization, not two-scale
    hybridization.
    """
    Cmu = self.config.Cmu
    nu_LES = self.work
    nu_RANS = self.nuT

    # Compute Smagorinsky nuT
    nu_LES[:] = self.strain_squared(U_hat[:3], out=self.work)
    nu_LES[:] = self.CDelta2 * np.sqrt(nu_LES)
    nu_LES[:] = self.filter(nu_LES, self.Ksmooth)

    if self.config.log_ke:
        nu_RANS[:] = Cmu * np.exp(2*self.U[3] - self.U[4])
    else:
        nu_RANS[:] = Cmu * self.U[3]**2 / self.U[4]
    nu_RANS[:] = self.filter(nu_RANS, self.Ksmooth) + 1e-20

    self.Cmu[:] = np.fmin(1, nu_LES/nu_RANS)

    return
