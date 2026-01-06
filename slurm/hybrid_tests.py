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

from math import ceil, log2, sqrt
import os
import shutil
import subprocess
import textwrap
import time
import datetime
from itertools import product


def enum_nloops(*args):
    yield from zip(product(*(range(len(x)) for x in args)), product(*args))


#############################################################################
# Basically never touch any of this unless switching systems
#############################################################################
ncpus = int(os.getenv('CPUS_PER_NODE'))
mpiprocs = ncpus

HOME = os.getenv('HOME')
WORK = os.getenv('WORK')

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

project = 'hybrid_tests'
os.makedirs(f"{WORK}/{project}/", exist_ok=True)
notebook = f'{HOME}/LAB_NOTEBOOK/shencfd/{project}'

#############################################################################
K = 1024
kfLow = 4
kfHigh = 6
frame = 49
idir = f"{WORK}/DNS/hybrid_ICs"

eps = 1.0
eta = 2 / K
nu = (eps * eta**4)**(1/3)
tau = 1.125
tauK = sqrt(nu / eps)

tau = float(f"{tau:0.3e}")  # rounding to 3 significant digits
tauK = float(f"{tauK:0.3e}")  # rounding to 3 significant digits

tlim = 40.0
trst = 4.0
tstat = 1/8

tlimit = tau * tlim
dt_rst = tau * trst
dt_stat = tau * tstat

##########################################################################
ACCOUNT = 'w22_mass-app'
restart = False
dealias = '2/3'
log_ke = True
smooth_ic = False
smooth_ke = False

# outer loops -> each iteration generates a separate slurm job
Kf = (2, 16, )
iratio = (6, 6, )

# "inner" loops -> many simulation instances per slurm job
models = ('xfsm', 'rgfsm', 'cesk', 'cesx', )
nruns = len(models)

# if you want to test this script by running only a single iteration:
TEST = False

##########################################################################
for ik, kf in enumerate(Kf):
    ratio = iratio[ik]
    N = int(kf * ratio)  # x3 since now using 2/3 dealiasing

    if N < 192:
        ntasks = N
    elif 192 <= N < 768:
        ntasks = N * 2
    else:  # N >= 768
        ntasks = N * 4

    nnodes = ceil(ntasks / mpiprocs)
    nstripes = min(round(log2(nnodes)), 32)
    wt_mins = 90

    if nnodes < 4:
        queue = 'debug'
        debug = ""
        wt_mins = min(30, wt_mins)
    else:
        queue = 'standard'
        debug = "## "

    walltime = str(datetime.timedelta(minutes=wt_mins))
    mintime = walltime

    jobname = f"hybrid_K{K}_k{kf}"
    slurm_script = f"slurm.{jobname}"
    os.chdir(f"{WORK}/{project}/")

    script = f"""\
    #!/bin/bash

    # ---------------------------------------------------------------------
    #SBATCH --account={ACCOUNT}
    #SBATCH --job-name={jobname}
    #SBATCH --output=%j.{jobname}.out      # Name of stdout output file
    #SBATCH --error=%j.{jobname}.out       # Name of stderr error file
    #SBATCH --qos={queue}
    {debug}#SBATCH --reservation=debug

    #SBATCH --nodes={nnodes}
    #SBATCH --ntasks={ntasks}
    #SBATCH --time={walltime}       # Desired walltime (hh:mm:ss)
    #SBATCH --time-min={mintime}    # Minimum acceptable walltime

    #SBATCH --begin=now+1           # do not start job until after n seconds
    # ---------------------------------------------------------------------

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{jobname}
    export PYTHONPATH=$JOBDIR:$PYTHONPATH

    for model in {' '.join(models)}
    do

        pid=K{K}_t{frame}_${{model}}_k{kf}_{ratio//2}dx
        outfile=out.${{pid}}.${{SLURM_JOB_ID}}

        cd {WORK}/{project}/$pid
        cp $JOBDIR/incompressible_hit.py ./
        cp $JOBDIR/sheninputs.$pid ./

        echo running $pid
        srun python incompressible_hit.py hybrid @sheninputs.$pid &> $outfile

        wait
        cp $outfile $JOBDIR/
        echo copied out.$pid to $JOBDIR

    done
    """

    with open(slurm_script, 'wt') as fh:
        fh.write(textwrap.dedent(script))

    print(f"submitting {slurm_script}...")
    output = subprocess.run(['sbatch', slurm_script],
                            capture_output=True, text=True).stdout
    print(output)

    jobid = output.split()[-1]
    jobdir = f"{notebook}/{today}/{jobid}.{jobname}"
    os.makedirs(jobdir, exist_ok=True)

    shutil.copy(slurm_script, jobdir)
    shutil.copy(f"{HOME}/{this_script}", jobdir)
    shutil.copy(f"{HOME}/shencfd/problems/incompressible_hit.py", jobdir)
    shutil.copytree(f"{HOME}/shencfd", f"{jobdir}/shencfd")

    os.chdir(jobdir)

    ##################################################################
    for j, model in enumerate(models):
        if log_ke:
            Cke = min(0.9, 0.01 * kf)
        else:
            Cke = min(0.5, 0.01 * kf)

        pid = f"K{K}_t{frame}_{model}_k{kf}_{ratio//2}dx"
        odir = f"{WORK}/{project}/{pid}"
        inputs_file = f"sheninputs.{pid}"
        os.makedirs(f"{odir}/restarts", exist_ok=True)
        subprocess.run(['lfs', 'setstripe', '-c', '1', odir])
        subprocess.run(['lfs', 'setstripe', '-c', str(nstripes),
                        f'{odir}/restarts'])

        inputs = f"""\
        pid           = {pid}
        odir          = {odir}
        N             = {N}
        nu            = {nu}
        dealiasing    = {dealias}
        log_ke        = {log_ke}
        smooth_ke     = {smooth_ke}
        smooth_ic     = {smooth_ic}

        restart       = {restart}
        init_pid      = {pid}
        idir          = {idir}
        frame         = {frame}

        model         = {model}
        base_ce2      = 1.714
        ic_ratio      = {ratio/3:0.5f}
        test_ratio    = {2*ratio/3:0.5f}

        tlimit        = {tlimit}
        dt_init       = 1.0e-6
        dt_rst        = {dt_rst}
        dt_stat       = {dt_stat}
        Co_keps       = {Cke:0.2e}
        init_only     = False
        """

        with open(inputs_file, 'wt') as fh:
            fh.write(textwrap.dedent(inputs))

    ##################################################################
    if TEST:
        break
