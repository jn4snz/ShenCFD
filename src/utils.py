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

import sys
from traceback import print_stack
from math import floor, log10
import argparse

__all__ = []

comm = MPI.COMM_WORLD


class MPI_Debugging:
    """A "Mixin" class to add simple debug methods to any class with an
    `MPI.Comm` instance stored as the attribute `self.comm`.

    """
    def print(self, string, flush=False):
        """Have rank 0 write `string` to `stdout`.

        """
        if self.comm.rank == 0:
            print(string, flush=flush)

    def soft_abort(self, string, code=1):
        """All tasks exit program with exit code `code` after rank 0 prints
        `string` and a stack trace to `stdout`.

        """
        if self.comm.rank == 0:
            print(string, flush=True)
            print_stack()

        sys.exit(code)

    def all_assert(self, test, string=""):
        """Allreduce `test` and exit program via `soft_abort` if False.

        """
        test = self.comm.allreduce(test, op=MPI.LAND)
        if test is False:
            self.soft_abort(string)

    def soft_assert(self, test, string=""):
        """If ``test is False`` then exit program via `soft_abort`.

        .. warning:: `soft_assert` assumes every MPI task receives identical
        value for `test`. If not, this function seems to hang indefinitely
        for some reason I have yet to fathom, even if `test` evaluates to
        False on all processes! Needs further investigation.

        """
        if test is False:
            self.soft_abort(string)

    def hard_assert(self, test, string=""):
        """If ``test is False`` then call `MPI.Comm.Abort` and hard crash.

        .. note:: Use this in place of Python's builtin `assert` for tests
        that may vary between processors and for which there is no way to
        perform an `all_assert`. The builtin `assert` will only kill its own
        process, leading to the MPI program hanging indefinitely.
        """
        if test is False:
            print(string, flush=True)
            self.comm.Abort()


class FileArgParser(argparse.ArgumentParser):
    def convert_arg_line_to_args(self, raw_line):
        args = []
        line = raw_line.strip().split('#')[0]
        if line:
            key, val = [kv.strip() for kv in line.strip().split('=')]

            if val.lower() == 'true':
                args.append(f'--{key}')

            elif val.lower() == 'false':
                args.append(f'--no-{key}')

            else:
                args.extend((f'--{key}', val))

        return args


def enf(x, p=5):
    """
    TAKEN FROM STACKOVERFLOW (answer from user `poppie`), and then updated to
    use f-strings.
    <https://stackoverflow.com/questions/17973278>

    Returns float/int value `x` formatted in a simplified engineering format -
    using an exponent that is a multiple of 3.
    """
    x = float(x)
    sign = " "

    if x < 0:
        x = -x
        sign = "-"

    if x == 0:
        exp3 = 0
        x3 = 0
    else:
        exp = int(floor(log10(x)))
        exp3 = exp - (exp % 3)
        x3 = x / (10 ** exp3)
        x3 = round(x3, -int(floor(log10(x3)) - (p-1)))

    x3_str = f"{x3:{p+1}.{p}f}"[:p+1]

    if exp3 == 0:
        exp3_str = "   "
    else:
        exp3_str = f"e{exp3:<+02d}"

    return f"{sign}{x3_str}{exp3_str}"
