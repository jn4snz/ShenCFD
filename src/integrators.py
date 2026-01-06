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

import numpy as np


def no_op(*args, **kwargs):
    return


class RK4_integrator:
    """4th-order multi-stage integrator of Runge and Kutta.

    The original 4th-order explicit Runge-Kutta scheme with optional
    inter-stage and inter-step user "update" functions.

    Parameters
    ----------
    ode : `MetaODE` class
        System of ordinary differential equations to be solved.
    y0  : array_like
        Initial value. If `y0` is a subclass of ndarray, then the array buffer
        of `y0` will be used as a base class ndarray solution field.

    """

    def __init__(self, y0, rhs, stage_update=no_op, step_update=no_op):
        self._rhs = rhs                    # not an instance method
        self._stage_update = stage_update  # not an instance method
        self._step_update = step_update    # not an instance method

        self._R = np.asarray(y0)      # current solution register
        self._R0 = np.empty_like(y0)  # previous solution register
        self._R1 = np.empty_like(y0)  # full-step accumulation register
        self._dR = np.zeros_like(y0)  # RHS evaluation register

        return

    @property
    def y(self):
        return self._R

    @property
    def dy(self):
        return self._dR

    def step(self, dt):
        """Short summary.

        Parameters
        ----------
        dt : type
            Description of parameter `dt`.
        args : type
            Description of parameter `args` (the default is []).
        kwargs : type
            Description of parameter `kwargs` (the default is {}).

        Returns
        -------
        type
            Description of returned object.

        """
        _a = [0.5, 0.5, 1.0]
        _b = [1./6., 1./3., 1./3., 1./6.]

        self._R0[:] = self._R1[:] = self._R[:]

        for rk in range(3):
            self._rhs(self._R, self._dR)

            self._R1 += _b[rk] * dt * self._dR
            self._R[:] = self._R0 + _a[rk] * dt * self._dR

            self._stage_update(self._R)

        # --------------------------------------------------------------
        # final RK stage
        self._rhs(self._R, self._dR)

        self._R1 += _b[3] * dt * self._dR
        self._R[:] = self._R1[:]

        self._stage_update(self._R)

        return self._R

    def integrate(self, tspan, y0=None, dy0=None, dt_init=1e-9):
        """Short summary.

        Parameters
        ----------
        tspan : type
            Description of parameter `tspan`.
        y0 : type
            Description of parameter `y0`.
        dt_init : type
            Description of parameter `dt_init` (the default is 1e-9).
        args : type
            Description of parameter `args` (the default is []).
        kwargs : type
            Description of parameter `kwargs` (the default is {}).

        Returns
        -------
        type
            Description of returned object.

        """
        if np.iterable(tspan):
            self.t = tspan[0]
            tlimit = tspan[1]
        else:
            self.t = 0.0
            tlimit = tspan

        if y0 is not None:
            self._R[:] = y0

        if dy0 is not None:
            self._dR[:] = dy0

        self.dt = dt_init

        while self.t < tlimit:
            if tlimit - self.t < self.dt:
                self.dt = tlimit - self.t

            self.step(self.dt)

            # compute dR/dt for update_step
            self._dR[:] = self._R1 - self._R0
            self._dR[:] *= (1.0 / self.dt)

            self.t += self.dt

            self.dt = self._step_update(self._R, self.t, self._dR, self.dt)

        return  # maybe add some kind of success code?
