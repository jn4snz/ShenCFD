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

from math import fsum
import numpy as np
from shenfun import FunctionSpace, TensorProductSpace, VectorSpace

from .maths import dot
from .statistics import allsum
from .utils import MPI_Debugging

COMM_WORLD = MPI.COMM_WORLD


def fft3d(N, padding=1, comm=COMM_WORLD):
    fft0 = FunctionSpace(N[0], padding_factor=padding, dtype='c16')
    fft1 = FunctionSpace(N[1], padding_factor=padding, dtype='c16')
    fft2 = FunctionSpace(N[2], padding_factor=padding, dtype='f8')
    fft_params = dict(planner_effort='FFTW_MEASURE',
                      slab=False,
                      collapse_fourier=False,
                      )
    return TensorProductSpace(comm, (fft0, fft1, fft2), **fft_params)


class FourierAnalysis(MPI_Debugging):

    def __init__(self, N, padding=1, comm=COMM_WORLD):
        self.comm = comm
        self.soft_assert(len(N) == 3)

        self._N = np.array(N, dtype='i8')
        self._padded = padding > 1
        self._fft = fft3d(N, padding, comm=comm)
        self._vfft = VectorSpace(self.fft)

        self._kmax = self.N // 2
        self._nmod = int(sum(self._kmax**2)**0.5) + 1

        K = self._K = self.fft.local_wavenumbers()
        Ksq = self._Ksq = K[0]**2 + K[1]**2 + K[2]**2

        K = np.asarray(np.broadcast_arrays(*K))
        self._K_Ksq = K / np.where(Ksq == 0, 1, Ksq)
        self._iK = 1j * K
        self._Kmod = np.sqrt(Ksq).astype('i4')

        return

    ###########################################################################
    @property
    def N(self):
        return self._N

    @property
    def is_padded(self):
        return self._padded

    @property
    def fft(self):
        return self._fft

    @property
    def vfft(self):
        return self._vfft

    @property
    def num_dofs(self):
        return np.prod(self.N)

    @property
    def nx(self):
        return self.fft.global_shape(False)

    @property
    def local_nx(self):
        return self.fft.shape(False)  # same as self.r.shape

    @property
    def num_points(self):
        return np.prod(self.nx)

    @property
    def nk(self):
        return self.fft.global_shape(True)

    @property
    def local_nk(self):
        return self.fft.shape(True)  # same as self.c.shape

    @property
    def kmax(self):
        return self._kmax

    @property
    def num_kpts(self):
        return np.prod(self.nk)

    @property
    def num_wavemodes(self):
        return self._nmod

    @property
    def K(self):
        return self._K

    @property
    def iK(self):
        return self._iK

    @property
    def K_Ksq(self):
        return self._K_Ksq

    @property
    def Ksq(self):
        """Get the dense array of wavenumber magnitude squared.

        .. math:: \\kappa = |\\mathbf{\\kappa}|^2.

        """
        return self._Ksq

    @property
    def wavemodes(self):
        """Get the dense array of integer wavenumber modes (waveshell indices).

        """
        return self._Kmod

    @property
    def r(self):
        """Real-valued (MPI-local) buffer array for `FFT`.

        `r` is the direct input to ``FFT.forward()`` and
        the direct output from ``FFT.backward()``.

        .. warning:: FFT buffers are overwritten by some `shenfun` backends
        and are therefore potentially unsafe for use as data arrays.

        """
        return self.fft.forward.input_array

    @property
    def c(self):
        """Complex-valued (MPI-local) buffer array for `FFT`.

        `c` is the direct output from ``FFT.forward()`` and
        the direct input to ``FFT.backward()``.

        .. warning:: FFT buffers are overwritten by some `shenfun` backends
        and are therefore potentially unsafe for use as data arrays.

        """
        return self.fft.forward.output_array

    def grad(self, a_hat, out=None):
        iK = self.iK

        if out is None:
            out = np.empty([3, *self.r.shape], dtype=self.r.dtype)

        for d in range(3):
            out[d] = self.fft.backward(iK[d] * a_hat)

        return out

    def div(self, u_hat, out=None):
        iK = self.iK

        if out is None:
            out = np.empty_like(self.r)

        out[:] = self.fft.backward(dot(iK, u_hat))

        return out

    def curl(self, u_hat, out=None):
        iK = self.iK

        if out is None:
            out = np.empty([3, *self.r.shape], dtype=self.r.dtype)

        out[0] = self.fft.backward(iK[1] * u_hat[2] - iK[2] * u_hat[1])
        out[1] = self.fft.backward(iK[2] * u_hat[0] - iK[0] * u_hat[2])
        out[2] = self.fft.backward(iK[0] * u_hat[1] - iK[1] * u_hat[0])

        return out

    def strain_squared(self, u_hat, out=None):
        iK = self.iK

        if out is None:
            out = np.empty_like(self.r)

        out[:] = 0.0
        for i in range(3):
            for j in range(i, 3):
                Sij = self.fft.backward(iK[j] * u_hat[i] + iK[i] * u_hat[j])
                out += (1 + (i != j)) * Sij**2

        out *= 0.5  # leaves 2 * Sij * Sij

        return out

    def project_divergence_free(self, u_hat):
        """Remove divergence component from spectral-space vector field.

        Project the input vector field onto the wavenumber vector field,
        which is equivalent to Helmholtz-Hodge decomposition in physical space.

        Parameters
        ----------
        u_hat : array_like
            Input array, must have shape ``[n, *self.c.shape]`` with n >= 3.
            Assumes vector to be projected is stored in first three components.

        Returns
        -------
        u_hat
            Returns the input for use as an in-line function if desired

        """
        u_hat[:3] += 1j * self.iK * dot(u_hat[:3], self._K_Ksq)

        return u_hat

    def compute_energy_spectrum(self, u_hat):
        """Compute the discrete 1D energy spectrum of the input.

        Parameters
        ----------
        u_hat : array_like
            Input array, must be a scalar or vector field defined on the
            Fourier domain `self.fft`.

        Returns
        -------
        E1d : 1-dimensional `ndarray`
            The discrete 1D energy spectrum of `u_hat`

        """
        if u_hat.ndim == 4:
            E3d = np.sum(np.real(u_hat * np.conj(u_hat)), axis=0)
        else:
            E3d = np.real(u_hat * np.conj(u_hat))

        K2 = self.K[2].reshape(-1)
        E3d[:, :, K2 == 0] *= 0.5
        E3d[:, :, K2 == self.kmax[2]] *= 0.5

        E1d = np.zeros(self.num_wavemodes, dtype=E3d.dtype)
        modes = self.wavemodes
        for k in range(self.num_wavemodes):
            E1d[k] = fsum(E3d[modes == k].flat)

        self.comm.Allreduce(MPI.IN_PLACE, E1d, op=MPI.SUM)

        return E1d

    def filter(self, x, kernel):
        if not np.isscalar(kernel):
            x_hat = self.fft.forward(x)  # x --> self.c
            x_hat *= kernel
            x[:] = self.fft.backward(x_hat)  # self.c --> x

        return x

    def analytic_compact_filter(self, kc):
        """Provides an infinitely differentiable smooth filter with strictly
        positive and compact support in both physical and spectral space.

        From Eyink and Aluie (2009), https://doi.org/10.1063/1.3266883

        """
        with np.errstate(divide='ignore', over='ignore'):
            kp = 4 * self.Ksq / kc**2
            Ghat = np.exp(-kp / np.abs(1 - kp))
            Ghat[kp >= 1] = 0.0  # must index in case value is inf

        G = self.fft.backward(Ghat)**2
        G *= self.num_points / allsum(G)  # FFT needs the num_points norm
        Ghat[:] = self.fft.forward(G).real
        Ghat[self.Ksq >= kc**2] = 0.0

        return Ghat

    def hypergaussian_filter(self, kc, C=8):
        """Provides 8th-order Hyper-Gaussian filter kernel computed pointwise
        directly from the spectral domain analytical formula.

        """
        return np.exp(-C * (self.Ksq / kc**2)**4)

    def analytic_gaussian_filter(self, delta, C=6):
        """Provides Gaussian filter kernel computed pointwise directly from the
        spectral domain analytical formula.

        """
        C = 1 / (4 * C)
        return np.exp(-C * delta**2 * self.Ksq)

    def discrete_gaussian_filter(self, width, C=6):
        """Provides Gaussian filter kernel computed pointwise from the physical
        domain formula and then forward transformed into the spectral domain.

        """
        N = (np.array(self.nx) - 1)//2 + 1
        ixg = [np.r_[0:n, -n:0] for n in N]
        slices = self.fft.local_slice(False)
        ixl = [ix[sl] for ix, sl in zip(ixg, slices)]
        X = np.meshgrid(*ixl, indexing='ij')
        Xsq = X[0]**2 + X[1]**2 + X[2]**2

        # X/width is non-dimensional, thus negating need to define L or dx
        G = np.exp(-C * Xsq / width**2)
        G *= self.num_points / allsum(G)  # FFT needs the num_points norm

        return np.array(self.fft.forward(G).real)

    def analytic_tophat_filter(self, delta):
        """Provides tophat filter kernel computed pointwise directly from the
        spectral domain tensor product of three sinc functions.

        """
        kdelta = np.where(self.K[0] == 0.0, 1e-20, delta * self.K[0])
        Ghat0 = 2 * np.sin(0.5 * kdelta) / kdelta

        kdelta = np.where(self.K[1] == 0.0, 1e-20, delta * self.K[1])
        Ghat1 = 2 * np.sin(0.5 * kdelta) / kdelta

        kdelta = np.where(self.K[2] == 0.0, 1e-20, delta * self.K[2])
        Ghat2 = 2 * np.sin(0.5 * kdelta) / kdelta

        return Ghat0 * Ghat1 * Ghat2  # broadcasts to dense array

    def discrete_tophat_filter(self, width):
        """Provides tophat filter kernel by forming a pointwise mask array in
        the physical domain, which is then forward transformed into the
        spectral domain.

        .. warning:: If width is an even integer, then the mask will not
        be centered about the origin. This could have an impact on the
        expected results of the filtering operation.

        """
        # TODO: look into performing a spectral-space grid-shift in order to
        # center the filter kernel in physical space.
        if np.isscalar(width):
            width = [width]*3

        N = (np.array(self.nx) - 1)//2 + 1
        ixg = [np.r_[0:n, -n:0] for n in N]
        slices = self.fft.local_slice(False)
        ixl = [ix[sl] for ix, sl in zip(ixg, slices)]
        X = np.meshgrid(*ixl, indexing='ij')

        G = np.ones_like(self.r)
        for i in range(3):
            Gm = np.abs(X[i] - 0.5) < width[i]/2
            Gp = np.abs(X[i] + 0.5) < width[i]/2
            G *= Gm.astype('i4') + Gp.astype('i4')

        G *= self.num_points / allsum(G)  # normalize for FFT

        return np.array(self.fft.forward(G).real)
