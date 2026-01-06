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

"""MPI-distributed statistics methods.

These methods not require information about domain decomposition or
dimensionality.

"""
from math import fsum
import numpy as np
from mpi4py import MPI
# from .utils import mpi_assert

__all__ = []

WCOMM = MPI.COMM_WORLD


def allsum(data, comm=WCOMM):
    return comm.allreduce(fsum(data.flat), op=MPI.SUM)


def allmin(data, comm=WCOMM):
    fill = np.ma.minimum_fill_value(data)
    return comm.allreduce(np.amin(data, initial=fill), op=MPI.MIN)


def allmax(data, comm=WCOMM):
    fill = np.ma.maximum_fill_value(data)
    return comm.allreduce(np.amax(data, initial=fill), op=MPI.MAX)


def moments(data, w=None, comm=WCOMM, root=0):
    """Compute min, max, and first six moments of MPI-distributed array.

    Computes raw weighted moments

    Parameters
    ----------
    data : array_like
        (MPI-distributed) array of numerical values.
    w : array_like, optional
        Weights array for computing a weighted moment. Must have same
        size as `data`, with `w.flat[i]` being the weight for `data.flat[i]`.
    comm : MPI.Comm, default=MPI.COMM_WORLD
        MPI intracommunicator over which data is distributed.

    Returns
    -------
    m1, m2, m3, m4, m5, m6, gmin, gmax : 8-tuple of scalars
        The MPI-reduced 1st-6th moments, minimum, and maximum of `data`.
        The returned values of `m1`-`m6` are always of type float, while
        `gmin` and `gmax` will be the same data-type as `data`.

    """
    data = np.asarray(data)  # returns view of data as plain ndarray
    gmin = allmin(data, comm)
    gmax = allmax(data, comm)

    if w is None:   # unweighted moments
        w = np.array([1.0])
        wbar = 1.0

    else:
        wbar = None

    buffer = np.empty(8)

    # global element count
    buffer[0] = data.size

    # 1st moment of the weights
    buffer[1] = fsum(w.flat)

    # 1st-6th raw moments
    buffer[2] = fsum((w * data).flat)
    buffer[3] = fsum((w * data**2).flat)
    buffer[4] = fsum((w * data**3).flat)
    buffer[5] = fsum((w * data**4).flat)
    buffer[6] = fsum((w * data**5).flat)
    buffer[7] = fsum((w * data**6).flat)

    if comm.rank == root:
        comm.Reduce(MPI.IN_PLACE, buffer, op=MPI.SUM, root=root)
    else:
        comm.Reduce(buffer, None, op=MPI.SUM, root=root)

    buffer[1:] /= buffer[0]          # divide by global data element count
    buffer[2:] /= wbar or buffer[1]  # divide moments by mean weight

    return (*buffer[2:], gmin, gmax)


def histogram1(x, bins=50, bounds=None, w=None, comm=WCOMM, root=0):
    """Compute an MPI-reduced histogram.

    This function is safe for null-sized arrays.

    Parameters
    ----------
    x : array_like
        (MPI-distributed) array of values to be binned.
    bins : type
        Description of parameter `bins`.
    bounds : type
        Description of parameter `bounds`.
    w : type
        Description of parameter `w`.
    comm : MPI.Comm, default=MPI.COMM_WORLD
        MPI intracommunicator over which data is distributed.

    Returns
    -------
    type
        Description of returned object.

    """
    bounds = bounds or (allmin(x, comm), allmax(x, comm))

    temp, edges = np.histogram(x, bins=bins, range=bounds, weights=w)
    hist = np.ascontiguousarray(temp)

    if comm.rank == root:
        comm.Reduce(MPI.IN_PLACE, hist, op=MPI.SUM, root=root)
    else:
        comm.Reduce(hist, None, op=MPI.SUM, root=root)

    return hist, edges


def histogram2(x, y, bins=50, bounds=None, w=None, comm=WCOMM, root=0):
    """Constructs the joint histogram two MPI-distributed data sets.

    Now safe for null-sized arrays.

    Parameters
    ----------
    x, y : array_like
        Description of parameters `x` and `y`.
    bins : sequence or scalar, default=50
        Description of parameter `bins`.
    bounds : type
        Description of `bounds`.
    w : array_like, optional
        Description of parameter `w`.
    comm : MPI.Comm, default=MPI.COMM_WORLD
        MPI intracommunicator over which data is distributed.

    Returns
    -------
    type
        Description of returned object.

    """
    x_range, y_range = bounds or (None, None)
    x_range = x_range or (allmin(x, comm), allmax(x, comm))
    y_range = y_range or (allmin(y, comm), allmax(y, comm))
    bounds = (x_range, y_range)

    temp, x_edges, y_edges = np.histogram2d(x, y, bins=bins, range=bounds,
                                            weights=w)
    hist = np.ascontiguousarray(temp)

    if comm.rank == root:
        comm.Reduce(MPI.IN_PLACE, hist, op=MPI.SUM, root=root)
    else:
        comm.Reduce(hist, None, op=MPI.SUM, root=root)

    return hist, x_edges, y_edges


def binned_sum(var, cond, bins=100, range=None, comm=WCOMM):
    """Compute an MPI-reduced binned sum (i.e., the conditional mean).

    Computes the MPI-distributed sum of `var` conditioned on the
    binned values of `cond`. This is the MPI-distributed equivalent to
    ``scipy.stats.binned_statistic(cond, var, 'mean', bins, range)``.

    Parameters
    ----------
    var : array_like
        (MPI-distributed) array of values to be summed in conditional bins.
    cond : array_like
        conditions to be binned.
    bins : sequence or int, default=100
        A sequence of bin edges or an integer number of bins between
        the lower and upper range values.
    range : sequence, optional
        A sequence of lower and upper bin edges to be used if the edges
        are not given explicitly in `bins`. The default is to use the
        MPI-reduced minimum and maximum values of `cond`.
    comm : MPI.Comm, default=MPI.COMM_WORLD
        MPI intracommunicator over which data is distributed.

    Returns
    -------
    binsum : [N,]-shaped ndarray
        (All ranks) ndarray of MPI-reduced conditional mean of `var`
        with length N equal to the number of bins needed to digitize
        `cond`, including outliers.
    bincounts : [N,]-shaped ndarray
        (All ranks) MPI-distributed histogram of `indices`, including
        outlier bins.
    indices : MPI-distribued ndarray
        (local to rank) digitized `cond` array with shape `(cond.size, )`
    """

    # Create edge arrays
    if np.isscalar(bins):
        nbins = bins + 2

        # Get the range
        cmin, cmax = range or (allmin(cond, comm), allmax(cond, comm))

        # get the edges
        edges = np.linspace(cmin, cmax, nbins - 1)

    else:
        edges = np.asarray(bins, np.float)
        nbins = edges.size + 1

    indices = np.digitize(cond.ravel(), edges)

    bincounts = np.bincount(indices, minlength=nbins)
    comm.Allreduce(MPI.IN_PLACE, bincounts, op=MPI.SUM)

    binsum = np.bincount(indices, var.ravel(), minlength=nbins)
    comm.Allreduce(MPI.IN_PLACE, binsum, op=MPI.SUM)

    return binsum, bincounts, indices
