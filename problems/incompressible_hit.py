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

import sys
import time

import shencfd as cfd

comm = MPI.COMM_WORLD

help_string = f"""\
usage: [MPI] python {sys.argv[0]} [-h] TYPE [type_options] ...

positional argument (case insensitive):
  TYPE
    dns     run a Direct Numerical Simulation
    les     run a Large Eddy Simulation
    ksgs    run a one-equation LES
    hybrid  run a Hybrid RANS/LES Simulation

optional arguments:
  -h,       Show this help message and exit
"""


###############################################################################
def main_script():
    wt0 = time.perf_counter()

    if comm.rank == 0:
        print("\n----------------------------------------------------------\n"
              "MPI-parallel Python shenCFD simulation of problem \n"
              "      'Homogeneous Isotropic Turbulence' \n"
              f"started with {comm.size} tasks at {time.strftime('%H:%M:%S')}."
              "\n----------------------------------------------------------\n",
              flush=True)

    # Initialize the simulation
    # -------------------------------------------------------------------------
    sim_type = str.lower(sys.argv[1])

    if sim_type == 'dns':
        sim = cfd.incompressible_isotropic.DNS()
        dti = min(sim.config.dt_init, sim.compute_dt(1e99))

    elif sim_type == 'les':
        sim = cfd.incompressible_isotropic.LES()
        dti = max(sim.config.dt_init, 1e-6)

    elif sim_type == 'ksgs':
        sim = cfd.incompressible_isotropic.KSGS()
        dti = max(sim.config.dt_init, 1e-6)

    elif sim_type == 'hybrid':
        sim = cfd.incompressible_isotropic.Hybrid()
        dti = max(sim.config.dt_init, 1e-8)

    elif sim_type == '-h':
        if comm.rank == 0:
            print(help_string, flush=True)
        return

    else:
        if comm.rank == 0:
            print("ERROR: did not understand command-line argument!")
            print(help_string, flush=True)
        return

    config = sim.config

    # we don't want init_only being added to restart files!
    init_only = getattr(config, 'init_only', False)
    delattr(config, 'init_only')

    if init_only:
        comm.Barrier()
        seconds = time.perf_counter() - wt0
        minutes, rem_seconds = divmod(int(seconds), 60)
        hours, minutes = divmod(minutes, 60)
        sim.print(
            "\n----------------------------------------------------------\n"
            "Initialization Walltime (H:M:S) = "
            f"{hours}:{minutes:02d}:{rem_seconds:02d}\n"
            "`Homogeneous Isotropic Turbulence' simulation [init-only]\n"
            f"finished at {time.strftime('%H:%M:%S')}.\n"
            "----------------------------------------------------------", True)

        return

    # Integrate forward in time
    # -------------------------------------------------------------------------
    comm.Barrier()
    wt1 = time.perf_counter()

    sol = cfd.RK4_integrator(sim.U_hat,
                             sim.compute_rhs,
                             sim.stage_update,
                             sim.step_update)

    sol.integrate([config.t_init, config.tlimit],
                  dt_init=dti)

    comm.Barrier()
    wt2 = time.perf_counter()

    # Finalize the simulation
    # -------------------------------------------------------------------------
    sim.finalize()

    minutes, rem_seconds = divmod(int(wt2 - wt0), 60)
    hours, minutes = divmod(minutes, 60)

    seconds = wt2 - wt1
    dof_cycles = sim.num_dofs * sim.istep / seconds
    cell_cycles = sim.num_points * sim.istep / seconds

    sim.print(
        "----------------------------------------------------------\n"
        f"Cycles completed = {sim.istep:4d}\n"
        f"Simulation time = {cfd.enf(sol.t, 8)}\n"
        f"Total Walltime (H:M:S) = {hours}:{minutes:02d}:{rem_seconds:02d}\n"
        f"-- startup time (s)     = {wt1 - wt0:0.1f}\n"
        f"-- integration time (s) = {seconds:0.1f}\n"
        f"Total cycles per second = {sim.istep / seconds:0.6g}\n"
        f"-- DOF cycles per second = {dof_cycles:0.6e}\n"
        f"-- cell cycles per second = {cell_cycles:0.6e}\n\n"
        "`Homogeneous Isotropic Turbulence' simulation finished at "
        f"{time.strftime('%H:%M:%S')}.\n"
        "----------------------------------------------------------")

    return


###############################################################################
if __name__ == "__main__":
    main_script()
    # import cProfile
    # cProfile.run(run())
