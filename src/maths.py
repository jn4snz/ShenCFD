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

from math import pi, fsum
import numpy as np


def dot(a, b):
    """Inner product along axis 0 for n-dimensional arrays.

    Based on single-core timings (using arrays of shape [3, 8, 8, 8]),
    `np.einsum('i...,i...', a, b)` wins by nearly a factor
    of 2 over `np.sum(a*b, axis=0)`, and by an order of magnitude
    compared to `np.tensordot(a, b, axes=0)`.

    Parameters
    ----------
    a, b : array_like
        `a` and `b` must have the same shape or `a*b` must be broadcastable.

    Returns
    -------
    `numpy.ndarray`
        dot-product of input arrays along axis 0.

    """
    return np.einsum('i...,i...', a, b)


def cross(a, b, out=None):
    """Cross product along axis 0 for n-dimensional arrays.

    Based on single-core timings (using arrays of shape [3, 8, 8, 8]),
    this manually-coded function is roughly twice as fast as
    `np.cross(a, b, axis=0)`. `np.einsum` is known to be much slower for
    n-dimensional cross-products. Mikael Mortensen noted that he found
    np.cross to be "very slow" in a comment in the source code of
    spectralDNS/maths/cross.py.

    Parameters
    ----------
    a, b : array_like
        axis 0 of both `a` and `b` must be dimension 3, and `a*b` must
        be broadcastable.
    out : array_like, optional
        `out` must have the same shape and datatype as the result of
        `a*b`. If `out` is not provided, then `a` and `b` must have the
        same shape.

    Returns
    -------
    `numpy.ndarray` or `out`
        the cross-product `a` and `b`

    """
    if out is None:
        dtype = (a.flat[:1] * b.flat[:1]).dtype
        out = np.empty_like(a, dtype=dtype)

    out[0] = a[1]*b[2] - a[2]*b[1]
    out[1] = a[2]*b[0] - a[0]*b[2]
    out[2] = a[0]*b[1] - a[1]*b[0]

    return out


def model_spectrum(x, nk, kd, KE, L, eta, kExp):
    """
    Returns a generic model spectrum designed to transition quickly from the
    infrared to inertial range, and have a shallow ultraviolet scaling.

    """
    k = np.arange(nk)
    k[0] = 1

    ki = 2*pi / L
    kL = k / ki  # = k * L / (2*pi)

    keta = k * eta

    # Infrared spectrum (k^2 or k^4 usually)
    E1d = x[0] * (2*KE / ki) * kL**kExp

    # Kolmogorov -5/3 spectrum
    E1d /= (1 + kL**6) ** ((kExp + 5/3) / 6)

    # ultraviolet spectrum
    E1d *= np.exp(- x[1] * keta)

    E1d[0] = 0.0    # zero mean
    E1d[kd:] = 0.0  # dealiased

    return E1d


def simple_spectrum(x, nk, kExp, kd):
    k = np.arange(nk)
    k[0] = 1

    E1d = x[0] * k**kExp
    E1d *= np.exp(- x[1] * k**(4/3))

    E1d[0] = 0.0
    E1d[kd:] = 0.0

    return E1d


def smooth_bandpass(nk, kfLow, kfHigh, kd, order=10):
    """Return a one-dimensional Linkwitz-Riley band-pass filter.

    """
    k = np.arange(nk)
    k[0] = 1

    E1d = 1 - 1/(1 + (k/kfLow)**order)  # nth-order L-R high pass
    E1d *= 1/(1 + (k/kfHigh)**order)    # nth-order L-R low pass

    E1d[0] = 0.0                        # zero mean
    E1d[kd:] = 0.0                      # dealiased
    E1d *= 1.0/E1d.max()                # max of 1

    return E1d


def spectral_correlation_length(E1d):
    r"""Compute the correlation length (integral scale) from a 1D spectrum.

    The discrete, correlation-based, integral length scale formula is

    .. math::

        \ell = \frac{3\pi}{4} \sum \kappa^{-1}\widehat{E}_K(\kappa)\,,

    where :math:`\widehat{E}_K` is the given 1D spectrum and :math:`\kappa`
    is the angular wavenumber.

    Parameters
    ----------
    E1d : array_like
        real-valued 1D spectrum (i.e., 1D physical-space correlation)

    Returns
    -------
    float
        The integral correlation length scale of `E1d`

    """
    k = np.arange(1, E1d.size)
    return 0.75 * pi * fsum(E1d[1:]/k) / fsum(E1d[1:])
