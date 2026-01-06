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

from math import fsum, pi

import numpy as np

from ..maths import dot
from ..statistics import allsum


def update_baseline_ce2(self, U_hat, *ignore):
    config = self.config

    if config.log_ke:
        self.W[3:] = np.exp(self.U[3:])
        k_m = self.W[3]
        eps = self.W[4]

    else:
        k_m = self.U[3]
        eps = self.U[4]

    self.curl(U_hat[:3], out=self.w)

    ######
    buffer = np.empty(4)

    buffer[0] = fsum((self.u**2).flat) * 0.5
    buffer[1] = fsum(k_m.flat)
    buffer[2] = fsum((self.w**2).flat) * self.nu
    buffer[3] = fsum(eps.flat)

    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

    buffer /= self.num_points
    ######

    K_r, K_m, eps_r, eps_m = buffer
    K_t = K_r + K_m
    eps_t = eps_r + eps_m

    Re = K_t**2 / (self.nu * eps_t)
    f = (Re/30) * (np.sqrt(60/Re + 1) - 1)
    config.Ce2 = f * (1 + (1/(config.p+1) + 1/2)*np.sqrt(60/Re + 1))

    return


def update_keps_viscosity(self, U_hat, *ignore):
    """Computes nuT for all k-epsilon models. Used at the end of each RK4 step
    for statistical outputs and updating the dynamic dt.

    """
    if self.config.log_ke:
        self.nuT[:] = self.Cmu * np.exp(2*self.U[3] - self.U[4])

    else:
        self.nuT[:] = self.Cmu * self.U[3]**2 / self.U[4]

    self.nuT[:] = self.filter(self.nuT, self.Ksmooth)

    return


def update_fsm_coeffs(self, U_hat, *ignore):
    """Compute Cmu for the Speziale version of FSM

    """
    config = self.config
    if config.log_ke:
        eps = self.W[4] = np.exp(self.U[4])
    else:
        eps = self.U[4]

    self.curl(U_hat[:3], out=self.w)
    self.W[0] = self.nu * dot(self.w, self.w) + eps
    eps_t = allsum(self.W[0]) / self.num_points

    f = 1 - np.exp(-config.beta * self.Delta * eps_t**0.25 / self.nu**0.75)
    self.Cmu = f * config.Cmu

    return


def update_vkfsm_coeffs(self, U_hat, *ignore):
    """Compute Cmu for the von Karman spectrum version of FSM

    """
    config = self.config

    C0 = 2 / (3 * 1.6)  # Ck = 1.6
    Z = pi * self.integral_length(U_hat) / self.Delta
    R = (1 + C0**4.5 * Z**3)**(-2/9)
    self.Cmu = R * config.Cmu

    return


def update_fkfsm_coeffs(self, U_hat, *ignore):
    """Compute Cmu for the observed F_k version of FSM

    """
    config = self.config
    if config.log_ke:
        self.W[3] = np.exp(self.U[3])
        k_m = self.W[3]
    else:
        k_m = self.U[3]

    ######
    buffer = np.empty(2)
    buffer[0] = fsum((self.u**2).flat) * 0.5
    buffer[1] = fsum(k_m.flat)
    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
    buffer /= self.num_points
    ######

    K_r, K_m = buffer
    Fk = K_m / (K_r + K_m)
    self.Cmu = Fk * config.Cmu

    return


def update_xfsm_coeffs(self, U_hat, *ignore):
    """Compute Cmu for XLES as FSM.

    """
    config = self.config
    Cmu = self.Cmu

    # F_mu = f_Delta = C*Delta/L_m = C * Delta * eps_m / k_m**(3/2)
    if config.log_ke:
        Cmu[:] = self.Delta * np.exp(self.U[4] - 1.5*self.U[3])
    else:
        Cmu[:] = self.Delta * self.U[4] * self.U[3]**(-1.5)

    Cmu[:] = self.filter(Cmu, self.Ksmooth)
    Cmu[:] = np.fmin(1, Cmu) * config.Cmu

    return


def update_rgfsm_coeffs(self, U_hat, *ignore):
    """Compute Cmu using 4/3rds scaling law as FSM.

    """
    config = self.config
    Cmu = self.Cmu

    # Cmu = ftau ^ 2 = f_Delta ^ 4/3 = (Delta * eps_m)**(4/3) / k_m**2
    if config.log_ke:
        Cmu[:] = self.Delta**(4/3) * np.exp((4/3)*self.U[4] - 2*self.U[3])

    else:
        Cmu[:] = (self.Delta * self.U[4])**(4/3) * self.U[3]**(-2)

    # now smooth and limit Cmu and Ck2 separately
    Cmu[:] = self.filter(Cmu, self.Ksmooth)
    Cmu[:] = np.fmin(1, Cmu) * config.Cmu

    return


def update_pitm_coeffs(self, U_hat, *ignore):
    """Compute Ce2 for the von Karman spectrum version of PITM

    """
    config = self.config

    C0 = 2 / (3 * 1.6)  # Ck = 1.6
    Z = pi * self.integral_length(U_hat) / self.Delta
    R = (1 + C0**4.5 * Z**3)**(-2/9)
    self.Ce2 = config.Ce1 + R * (config.Ce2 - config.Ce1)

    return


def update_opitm_coeffs(self, U_hat, *ignore):
    """Compute Ce2 for the observed F_k version of PITM

    """
    config = self.config
    if config.log_ke:
        self.W[3] = np.exp(self.U[3])
        k_m = self.W[3]
    else:
        k_m = self.U[3]

    ######
    buffer = np.empty(2)
    buffer[0] = fsum((self.u**2).flat) * 0.5
    buffer[1] = fsum(k_m.flat)
    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
    buffer /= self.num_points
    ######

    K_r, K_m = buffer
    Fk = K_m / (K_r + K_m)
    self.Ce2 = config.Ce1 + Fk * (config.Ce2 - config.Ce1)

    return


def update_pans_coeffs(self, U_hat, *ignore):
    """Compute Ce2 for the standard version of PANS (sans Prandtl numbers).

    Standard PANS uses a fixed value of "R", the prescribed filtered DNS value
    of F_k.
    """
    config = self.config

    # Logic guard so that coeffs only computed once, during initialization
    if self._computed:
        pass

    else:
        if config.log_ke:
            self.W[3] = np.exp(self.U[3])
            k_m = self.W[3]
        else:
            k_m = self.U[3]

        ######
        buffer = np.empty(2)
        buffer[0] = fsum((self.u**2).flat) * 0.5
        buffer[1] = fsum(k_m.flat)
        self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)
        buffer /= self.num_points
        ######

        K_r, K_m = buffer
        Fk = K_m / (K_r + K_m)
        self.Ce2 = config.Ce1 + Fk * (config.Ce2 - config.Ce1)

        self._computed = True

    return


def update_opans_coeffs(self, U_hat, *ignore):
    """Compute Ce2 for the observed F_k/F_epsilon verson of PANS.

    This version does not use a fixed "R" value.
    """
    config = self.config

    if config.log_ke:
        self.W[3:] = np.exp(self.U[3:])
        k_m = self.W[3]
        eps = self.W[4]

    else:
        k_m = self.U[3]
        eps = self.U[4]

    self.curl(U_hat[:3], out=self.w)

    ######
    buffer = np.empty(4)

    buffer[0] = fsum((self.u**2).flat) * 0.5
    buffer[1] = fsum(k_m.flat)
    buffer[2] = fsum((self.w**2).flat) * self.nu
    buffer[3] = fsum(eps.flat)

    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

    buffer /= self.num_points
    ######

    K_r, K_m, Eps_r, Eps_m = buffer
    Fk = K_m / (K_r + K_m)
    Fe = Eps_m / (Eps_r + Eps_m)

    self.Ce2 = config.Ce1 + (Fk / Fe) * (config.Ce2 - config.Ce1)

    return


def update_des_coeffs(self, U_hat, *ignore):
    """Compute Ck2 for the standard version of DES.

    """
    config = self.config
    Ck2 = self.Ck2

    # Ck2 = 1/f_Delta = L_m/(C*Delta) = k_m**(3/2) / (eps_m * C * Delta)
    if config.log_ke:
        Ck2[:] = (1/self.Delta) * np.exp(1.5*self.U[3] - self.U[4])
    else:
        Ck2[:] = (1/self.Delta) * self.U[3]**1.5 / self.U[4]

    Ck2[:] = self.filter(Ck2, self.Ksmooth)
    Ck2[:] = np.fmax(1, Ck2)

    return


def update_xles_coeffs(self, U_hat, *ignore):
    """Compute Cmu and Ck2 for XLES.

    """
    config = self.config
    Cmu = self.Cmu
    Ck2 = self.Ck2

    # F_mu = f_Delta = C*Delta/L_m = C * Delta * eps_m / k_m**(3/2)
    if config.log_ke:
        Cmu[:] = self.Delta * np.exp(self.U[4] - 1.5*self.U[3])
    else:
        Cmu[:] = self.Delta * self.U[4] * self.U[3]**(-1.5)

    # Ck2 = 1 / f_Delta
    Ck2[:] = 1 / Cmu

    Cmu[:] = self.filter(Cmu, self.Ksmooth)
    Cmu[:] = np.fmin(1, Cmu) * config.Cmu

    Ck2[:] = self.filter(Ck2, self.Ksmooth)
    Ck2[:] = np.fmax(1, Ck2)

    return


def update_rgtau_coeffs(self, U_hat, *ignore):
    """Compute Cmu, Ck2, Ce1, and Ce2 for RG-tau (using standard RANS baseline)

    """
    config = self.config
    Cmu = self.Cmu
    Ck2 = self.Ck2

    # Cmu = ftau ^ 2 = f_Delta ^ 4/3 = (Delta * eps_m)**(4/3) / k_m**2
    # Ck2 = ftau ^ -1 = f_Delta ^ -2/3 = k_m / (eps_m * Delta)**(2/3)
    if config.log_ke:
        Cmu[:] = self.Delta**(4/3) * np.exp((4/3)*self.U[4] - 2*self.U[3])
        Ck2[:] = np.exp(self.U[3] - (2/3)*self.U[4]) * self.Delta**(-2/3)

    else:
        Cmu[:] = (self.Delta * self.U[4])**(4/3) * self.U[3]**(-2)
        Ck2[:] = self.U[3] * self.U[4]**(-2/3) * self.Delta**(-2/3)

    # now smooth and limit Cmu and Ck2 separately
    Cmu[:] = self.filter(Cmu, self.Ksmooth)
    Cmu[:] = np.fmin(1, Cmu) * config.Cmu

    Ck2[:] = self.filter(Ck2, self.Ksmooth)
    Ck2[:] = np.fmax(1, Ck2)

    self.Ce1[:] = Ck2 * config.Ce1
    self.Ce2[:] = Ck2 * config.Ce2

    return


def update_ocess_coeffs(self, U_hat, *ignore):
    """Compute Ce2 using observed F_L for CES-S

    """
    config = self.config

    if config.log_ke:
        self.W[3:] = np.exp(self.U[3:])
        k_m = self.W[3]
        eps = self.W[4]

    else:
        k_m = self.U[3]
        eps = self.U[4]

    self.curl(U_hat[:3], out=self.w)

    ######
    buffer = np.empty(4)

    buffer[0] = fsum((self.u**2).flat) * 0.5
    buffer[1] = fsum(k_m.flat)
    buffer[2] = fsum((self.w**2).flat) * self.nu
    buffer[3] = fsum(eps.flat)

    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

    buffer /= self.num_points
    ######

    K_r, K_m, Eps_r, Eps_m = buffer
    K_t = K_r + K_m
    Eps_t = Eps_r + Eps_m

    # R = F_L^2 = (Lm / Lt)**2 = F_k**3 / F_eps**2
    R = K_m**3 * K_t**-3 * Eps_m**-2 * Eps_t**2
    self.Ce2 = config.Ce1 + R * (config.Ce2 - config.Ce1)

    return


def update_cess_coeffs(self, U_hat, *ignore):
    """Compute Ce2 using von Karman spectrum version of CES-S

    """
    config = self.config

    C0 = (1.5 * 1.6)**1.5 / pi  # Ck = 1.6
    Delta_s = C0 * self.Delta / self.integral_length(U_hat)

    R = Delta_s**2 * (1 + Delta_s**3)**(-2/3)

    self.Ce2 = config.Ce1 + R * (config.Ce2 - config.Ce1)

    return


def update_cesk_coeffs(self, U_hat, *ignore):
    """Compute Ck2 using von Karman spectrum version of CES-K

    """
    config = self.config

    C0 = (1.5 * 1.6)**1.5 / pi  # Ck = 1.6
    Delta_s = C0 * self.Delta / self.integral_length(U_hat)

    R = Delta_s**2 * (1 + Delta_s**3)**(-2/3)
    a = config.Ce2 / config.Ce1

    self.Ck2 = a - R * (a - 1)

    return


def update_cesu_coeffs(self, U_hat, *ignore):
    """Compute Cmu and Ck2 using von Karman spectrum version of CES-U

    """
    config = self.config

    C0 = (1.5 * 1.6)**1.5 / pi  # Ck = 1.6
    Delta_s = C0 * self.Delta / self.integral_length(U_hat)

    R = Delta_s**2 * (1 + Delta_s**3)**(-2/3)
    a = config.Ce2 / config.Ce1

    self.Ck2 = (a - R * (a - 1))**0.5
    self.Cmu = config.Cmu / self.Ck2

    return


def update_cesx_coeffs(self, U_hat, *ignore):
    """Compute Cmu and Ck2 using von Karman spectrum version of CES-X

    """
    config = self.config

    C0 = (1.5 * 1.6)**1.5 / pi  # Ck = 1.6
    Delta_s = C0 * self.Delta / self.integral_length(U_hat)

    R = Delta_s**2 * (1 + Delta_s**3)**(-2/3)
    a = config.Ce2 / config.Ce1

    self.Ck2 = (a - R * (a - 1))  # == a(1-R) + R, Ck2 goes from 1 to a
    self.Cmu = config.Cmu / self.Ck2  # Cmu goes from 0.09 to 0.09/a

    return


def update_pfsm_coeffs(self, U_hat, *ignore):
    """This is FSM but using f_k, which is also is the Perot-Gadebusch Model
    with alpha set to unity.

    """
    config = self.config
    fk = self.W[0]
    k_r = self.work

    if config.log_ke:
        k_m = self.W[3] = np.exp(self.U[3])
    else:
        k_m = self.U[3]

    k_r[:] = 0.5 * dot(self.u, self.u)
    fk[:] = k_m / (k_m + k_r)
    fk[:] = self.filter(fk, self.Ksmooth)
    self.Cmu[:] = config.Cmu * fk

    return


def update_perot_coeffs(self, U_hat, *ignore):
    """
    Compute Cmu and alpha for the 2007 Perot and Gadebusch model.

    """
    config = self.config
    alpha = self.alpha

    # point working memory to readable names
    fk = self.W[0]
    gif = self.W[1]
    k_r = self.work  # cannot be stored in W[:3]

    if config.log_ke:
        k_m = self.W[3] = np.exp(self.U[3])
    else:
        k_m = self.U[3]

    k_r[:] = 0.5 * dot(self.u, self.u)

    self.fft.forward(np.log(np.sqrt(k_r)), self.W_hat[3])
    self.w_hat[:] = self.iK * self.W_hat[3] * self.Ksmooth
    self.vfft.backward(self.w_hat, self.w)

    dx = self.Delta
    gif[:] = dx**2 * dot(self.w, self.w)
    gif[:] = self.filter(gif, self.Ksmooth)

    fk[:] = k_m / (k_m + k_r)
    fk[:] = self.filter(fk, self.Ksmooth)
    self.Cmu[:] = config.Cmu * fk

    alpha[:] = 1.5 * (1.0 - 0.28 * fk**2 / (gif + 0.11))
    alpha[:] = self.filter(alpha, self.Ksmooth)

    return


def update_dynamic_cmu(self, U_hat, *ignore):
    """Compute Cmu using Germano-Lilly proceduce for dynamic FSM.

    55 total scalar FFTs in function.
    """
    config = self.config
    iK = self.iK

    # pointing working memory to readable names
    work = self.work   # can't be used once nu2 computed
    nu1 = self.nuT
    nu2 = self.work    # can't be in W

    k_f = self.W[3]    # can't be in W[:3] == w
    eps_f = self.W[4]  # can't be in W[:3] == w

    Sij_hat = self.W_hat[3]
    Mij = self.work2[0]
    Lij = self.work2[1]
    MijMij = self.W[3]  # overwrites k_f
    LijMij = self.W[4]  # overwrites eps_f

    if config.log_ke:
        k = self.W[3] = np.exp(self.U[3])
        eps = self.W[4] = np.exp(self.U[4])
        nu1[:] = np.exp(2*self.U[3] - self.U[4])

    else:
        k = self.U[3]
        eps = self.U[4]
        nu1[:] = k**2 / eps

    self.w_hat[:] = self.Ktest * U_hat[:3]
    self.vfft.backward(self.w_hat, self.w)

    k_f[:] = k + 0.5 * dot(self.u, self.u)
    k_f[:] = self.filter(k_f, self.Ktest)
    k_f -= 0.5 * dot(self.w, self.w)

    # strain_squared outputs 2SijSij
    eps_f[:] = eps + self.nu * self.strain_squared(U_hat[:3], out=work)
    eps_f[:] = self.filter(eps_f, self.Ktest)
    eps_f -= self.nu * self.strain_squared(self.w_hat, out=work)

    nu2[:] = k_f**2 / eps_f

    MijMij[:] = 0.0
    LijMij[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij_hat[:] = iK[j] * U_hat[i] + iK[i] * U_hat[j]
            Mij[:] = nu1 * self.fft.backward(Sij_hat)
            Mij[:] = self.filter(Mij, self.Ktest)
            Mij[:] -= nu2 * self.fft.backward(self.Ktest * Sij_hat)

            Lij[:] = self.u[i] * self.u[j]
            Lij[:] = self.filter(Lij, self.Ktest)
            Lij -= self.w[i] * self.w[j]

            LijMij += (1 + (i != j)) * Lij * Mij  # note sign
            MijMij += (1 + (i != j)) * Mij**2

    buffer = np.empty(2)
    buffer[0] = fsum(LijMij.flat)
    buffer[1] = fsum(MijMij.flat)
    self.comm.Allreduce(MPI.IN_PLACE, buffer, op=MPI.SUM)

    self.Cmu = buffer[0] / buffer[1]
    # quality check
    if self.Cmu < 0.0 or self.Cmu > config.Cmu:
        self.print(f'LijMij = {buffer[0]:0.3e}; MijMij = {buffer[1]:0.3e}; '
                   f'Cmu = {self.Cmu:0.3e}')
        self.Cmu = config.Cmu

    return


def rhs_k_epsilon(self, U_hat, dU):
    """Hybrid RANS-LES k-epsilon RHS

    Computes all k-epsilon RHS terms and velocity RHS closure terms with
    potentially variable RANS model coefficients, along with modifications to
    include Perot's alpha coefficient and the modified destruction of k.

    This function requires a total of 27 scalar FFTs.
    """
    # point allocated memory to descriptive names
    iK = self.iK
    k_m = self.U[3]
    km_hat = U_hat[3]
    eps = self.U[4]
    eps_hat = U_hat[4]

    nuT = self.nuT
    alpha = self.alpha
    Ssq = self.W[3]
    work = self.work

    nuT[:] = self.Cmu * k_m**2 / eps
    # smooth the "infinite" aliasing due to division by eps
    nuT[:] = self.filter(nuT, self.Ksmooth)

    # --------------------------------------------------------------
    # Add Reynolds stress to velocity RHS
    # --------------------------------------------------------------
    Ssq[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])
            Ssq += (1 + (i != j)) * Sij**2
            Rij_hat = self.fft.forward(alpha * nuT * Sij)
            dU[i] += iK[j] * Rij_hat
            dU[j] += (i != j) * iK[i] * Rij_hat

    Ssq *= 0.5

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
    # (Production has cubic aliasing when alpha = 1 and nuT already smoothed)
    work[:] = alpha * nuT * Ssq
    dU[3] += self.fft.forward(work) * self.Ksmooth
    dU[3] -= self.Ck2 * eps_hat

    # --------------------------------------------------------------
    # Compute RHS of epsilon-transport
    # --------------------------------------------------------------
    # compute turbulent transport flux in physical space
    self.w_hat[:] = iK * eps_hat
    self.vfft.backward(self.w_hat, self.w)
    self.w *= self.sige_inv*nuT + self.nu

    # add advective flux and transform back to Fourier space
    self.w -= self.u * eps
    self.vfft.forward(self.w, self.w_hat)

    # take the divergence of the advective-diffusive flux
    dU[4] = dot(iK, self.w_hat)

    # add production (cubic aliasing) and destruction (infinite aliasing)
    work[:] = self.Ce1 * self.Cmu * k_m * Ssq - self.Ce2 * eps**2 / k_m
    dU[4] += self.fft.forward(work) * self.Ksmooth

    return dU


def rhs_log_k_eps(self, U_hat, dU):
    """Modified Hybrid RANS-LES k-epsilon RHS using log(k) and log(eps).

    This function requires 27 total scalar FFTs.
    """
    # point allocated memory to descriptive names
    iK = self.iK
    lnk = self.U[3]
    lnk_hat = U_hat[3]
    lne = self.U[4]
    lne_hat = U_hat[4]

    nuT = self.nuT
    work = self.work
    Sij = self.work
    eps_k = self.W[3]
    P_k = self.W[4]

    # compute physical-space variables
    eps_k[:] = np.exp(lne - lnk)
    nuT[:] = self.Cmu * np.exp(2*lnk - lne)
    # smooth the "infinite" aliasing due to division by eps
    nuT[:] = self.filter(nuT, self.Ksmooth)

    # --------------------------------------------------------------
    # Add Reynolds stress to velocity RHS
    # --------------------------------------------------------------
    P_k[:] = 0.0
    for i in range(3):
        for j in range(i, 3):
            Sij[:] = self.fft.backward(iK[j] * U_hat[i] + iK[i] * U_hat[j])
            P_k += (1 + (i != j)) * Sij**2
            Sij *= self.alpha * nuT
            Rij_hat = self.fft.forward(Sij)
            dU[i] += iK[j] * Rij_hat
            dU[j] += (i != j) * iK[i] * Rij_hat

    P_k *= 0.5 * self.Cmu * np.exp(lnk - lne)

    # --------------------------------------------------------------
    # Compute RHS of k-transport
    # --------------------------------------------------------------
    # compute grad(lnk)
    self.w_hat[:] = iK * lnk_hat
    self.vfft.backward(self.w_hat, self.w)

    # compute lnk source terms
    work[:] = (self.sigk_inv*nuT + self.nu) * dot(self.w, self.w)
    work += self.alpha * P_k - self.Ck2 * eps_k
    dU[3] = self.fft.forward(work) * self.Ksmooth

    # compute conservative turbulent diffusion and advective fluxes
    self.w *= self.sigk_inv*nuT + self.nu
    self.w -= self.u * lnk
    self.vfft.forward(self.w, self.w_hat)

    # take the divergence of the advective-diffusive flux
    dU[3] += dot(iK, self.w_hat)

    # --------------------------------------------------------------
    # Compute RHS of epsilon-transport
    # --------------------------------------------------------------
    # compute grad(lne)
    self.w_hat[:] = iK * lne_hat
    self.vfft.backward(self.w_hat, self.w)

    # compute lne source terms
    work[:] = (self.sige_inv*nuT + self.nu) * dot(self.w, self.w)
    work += self.Ce1 * P_k - self.Ce2 * eps_k
    dU[4] = self.fft.forward(work) * self.Ksmooth

    # compute conservative turbulent transport and advective fluxes
    self.w *= self.sige_inv*nuT + self.nu
    self.w -= self.u * lne
    self.vfft.forward(self.w, self.w_hat)

    # take the divergence of the advective-diffusive flux
    dU[4] += dot(iK, self.w_hat)

    return dU
