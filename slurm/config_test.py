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
# Basically never touch any of this. Setup is in .bash_profile
ncpus = int(os.getenv('CPUS_PER_NODE'))
mpiprocs = ncpus

HOME = os.getenv('HOME')
WORK = os.getenv('WORK')

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

project = 'config_tests'
os.makedirs(f"{WORK}/{project}/", exist_ok=True)
notebook = f'{HOME}/LAB_NOTEBOOK/shencfd/{project}'

#############################################################################
# Put anything constant between runs here
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

tlim = 20.0
trst = 2.0
tstat = 1/8

tlimit = tau * tlim
dt_rst = tau * trst
dt_stat = tau * tstat

##########################################################################
ACCOUNT = 'w22_mass-app'
restart = False
model = 'xles'

TEST = False  # quit script after first iteration of outer loop
SKIP = False  # skip first iteration of outer loop because already tested

# OUTER LOOPS
Kf = (8, 16, 32, )
iratio = (6, )

# INNER LOOPS
dealias = '2/3'
log_ke = True
smooth_ke = False
smooth_ic = False
C_ke = (0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, )

##########################################################################
for (ik, ir), (kf, ratio) in enum_nloops(Kf, iratio):
    if ik == 0 and ir == 0 and SKIP:
        continue

    N = int(kf * ratio)

    if N < 192:
        ntasks = N
    elif 192 <= N < 512:
        ntasks = N * 2
    else:  # N >= 512
        ntasks = N * 4

    nnodes = ceil(ntasks / mpiprocs)
    nstripes = min(max(ceil(log2(nnodes)), 1), 32)
    wt_mins = 240

    if nnodes < 2:
        nstripes = 1
        queue = 'debug'
        debug = ""
        wt_mins = min(120, wt_mins)
    else:
        queue = 'standard'
        debug = "## "

    walltime = str(datetime.timedelta(minutes=wt_mins))
    mintime = walltime

    jobname = f"cfg_k{kf}_{ratio//2}dx"
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
    ## #SBATCH --mail-type=all
    ## #SBATCH --mail-user=towery@lanl.gov
    # ---------------------------------------------------------------------

    export OMP_NUM_THREADS=1
    export OPENBLAS_NUM_THREADS=1

    JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{jobname}
    export PYTHONPATH=$JOBDIR:$PYTHONPATH

    for Cke in 20 30 40 50 60 70 80 90
    do

    pid=K{K}_t{frame}_{model}_k{kf}_{ratio//2}dx_C${{Cke}}
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
    for Cke in C_ke:
        pid = f"K{K}_t{frame}_{model}_k{kf}_{ratio//2}dx_C{int(Cke*100)}"
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
