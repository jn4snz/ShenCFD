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

from abc import ABC, abstractmethod
import time
import numpy as np
import h5py

from .utils import MPI_Debugging

__all__ = []

comm = MPI.COMM_WORLD


class h5FileIO(h5py.File, MPI_Debugging):
    """Class for reading/writing a single snapshot of 3D data to the parallel
    HDF5 format, much simplified from the original :class:`HDF5File`.

    """
    def __init__(self, filename, mode='r', comm=comm, **h5_kw):
        super().__init__(filename, mode, driver="mpio", comm=comm, **h5_kw)
        self.comm = comm

        return

    def parsable_attrs(self):
        """Convert file attrs into a list of strings for argparse

        """
        args = []
        unparsed = {}
        for key, val in self.attrs.items():
            if isinstance(val, np.ndarray):
                if val.size < 4:
                    args.append('--{key}')
                    args.extend([str(n) for n in val])

                else:  # not a shenCFD command-line input
                    unparsed[key] = val

            elif val is np.bool_(1):    # if val is "True" as saved by h5py
                args.append(f'--{key}')

            elif val is np.bool_(0):    # if val is "False" as saved by h5py
                args.append(f'--no-{key}')

            else:
                args.append(f'--{key}={val}')

        return args, unparsed

    def read(self, name, U, T=None):
        """
        Read ``U``'s `local_slice()` from HDF5 file. `U` must either be a
        :class:`shenfun.Array` or :class:`shenfun.Function` object or a
         :class:`shenfun.TensorProductSpace` must be provided.
        Parameters
        ----------
        name: str
            Base string for tensor field `U`. For example, if `U` is rank 1,
            then scalar fields will be read in from datasets "name0", "name1",
            etc.
        U: :class:`numpy.ndarray`-like or :class:`shenfun.Array`-like
            The data field to be read from the named dataset.
        T: :class:`shenfun.TensorProductSpace`, optional
            The `shenfun` basis space that defines `U`'s global shape and
            local slice of the global array. Must be same Tensor-rank as `U`.

        """
        wt0 = time.perf_counter()

        rank = len(U.shape) - 3
        if hasattr(U, 'local_slice'):
            local_slice = U.local_slice()[-3:]

        else:  # CompositeSpaces are tricksy hobbits, FYI
            Tshape = T.shape(False)[-3:]
            Ushape = U.shape[-3:]
            fwd_out = not Tshape == Ushape
            local_slice = T.local_slice(fwd_out)[-3:]

        if rank == 0:
            dset = self[name]
            with dset.collective:
                U[:] = dset[local_slice]

        else:  # if rank == 1:
            for i in range(U.shape[0]):
                dset = self[f"{name}{i}"]
                with dset.collective:
                    U[i] = dset[local_slice]

        wt1 = time.perf_counter()
        self.print(f" +++ Read {name} from disk in {wt1-wt0} seconds")

        return

    def write(self, name, U, T=None, kwargs={}):
        """Write ``U`` to HDF5 file.

        Parameters
        ----------
        name: str
            Base string for tensor field `U`. For example, if `U` is rank 1,
            then scalar fields will be stored as datasets "name0", "name1",
            etc.
        U: :class:`numpy.ndarray`-like or :class:`shenfun.Array`-like
            The data field to be stored as the named dataset.
        T: :class:`shenfun.TensorProductSpace`, optional
            The `shenfun` basis space that defines `U`'s global shape and
            local slice of the global array. Must be same Tensor-rank as `U`.

        """
        wt0 = time.perf_counter()

        rank = len(U.shape) - 3
        if hasattr(U, 'local_slice'):
            local_slice = U.local_slice()[-3:]
            global_shape = U.global_shape[-3:]  # NOTE the lack of () here.

        else:  # CompositeSpaces are tricksy hobbits, FYI
            Tshape = T.shape(False)[-3:]
            Ushape = U.shape[-3:]
            fwd_out = not Tshape == Ushape
            local_slice = T.local_slice(fwd_out)[-3:]
            global_shape = T.global_shape(fwd_out)[-3:]

        if rank == 0:
            dset = self.require_dataset(
                name, shape=global_shape, dtype=U.dtype)
            with dset.collective:
                dset[local_slice] = U

        else:  # if rank == 1:
            for i in range(U.shape[0]):
                dset = self.require_dataset(
                    f"{name}{i}", shape=global_shape, dtype=U.dtype)
                with dset.collective:
                    dset[local_slice] = U[i]

        for k, v in kwargs.items():
            if v is not None:
                self.attrs[k] = v

            else:
                self.print(f'FIXME: found a None kw, {k}')

        wt1 = time.perf_counter()
        self.print(f" +++ Wrote {name} to disk in {wt1-wt0} seconds")

        return


# !!! WORK IN PROGRESS, PLEASE IGNORE !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
class TimedEvent(ABC):
    """

    """
    def __init__(self, tag, dt=0.0, t_start=0.0, count=0, file=None):
        self.tag = tag
        self.count = count

        if file is None:
            self.t_next = t_start
            self.dt = dt
        else:
            self.from_file(file)

        return

    def __call__(self, U_hat, t_sim=1e99, dt_sim=-1e99):
        """Default values of t_sim and dt_sim should force trigger event and
        then update t_next with dt.

        """
        if (t_sim + 0.01 * self.dt) >= self.t_next:
            self.t_next += max(self.dt, (t_sim + dt_sim) - self.t_next)
            self.count += 1

        return

    @abstractmethod
    def from_file(self, file):
        pass


class TimedOutput(ABC):
    """

    """
    def __init__(self, func, tag, filename, dt, t_start=0.0, count=0):
        self.func = func
        self.tag = tag
        self.filename = filename
        self.t_next = t_start
        self.dt = dt
        self.count = count

    def __call__(self, U_hat, t_sim, dt_sim):
        if (t_sim + 0.01 * self.dt) >= self.t_next:
            self.t_next += max(self.dt, (t_sim + dt_sim) - self.t_next)

        self.func(U_hat, t_sim)

        return

    @abstractmethod
    def restart(self, old_file):
        pass

    @abstractmethod
    def open(self, mode):
        pass

    @abstractmethod
    def close(self):
        pass

    @abstractmethod
    def flush(self):
        pass
