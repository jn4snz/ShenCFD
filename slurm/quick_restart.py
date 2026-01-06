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

import os
import shutil
import subprocess
import time
from math import ceil, log2

ncpus = int(os.getenv('CPUS_PER_NODE'))
mpiprocs = ncpus

HOME = os.getenv('HOME')
WORK = os.getenv('WORK')

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

##############################################################################
project = 'hybrid_tests'
sim_type = 'hybrid'
notebook = f'{HOME}/LAB_NOTEBOOK/shencfd/{project}'

pid = "K1024_t49_mpans_const_k64_3dx_PLRS"
frame = -1
tau = 1.125
tauK = 0.015625
dt_rst = 4 * tau
dt_stat = tau / 8
tlimit = 40 * tau

ntasks = 1536
walltime = "1:00:00"
mintime = "1:00:00"
ACCOUNT = 'w22_mass-app'

odir = f"{WORK}/{project}/{pid}"
jobname = pid

##############################################################################
nnodes = ceil(ntasks / mpiprocs)
if nnodes < 3:
    queue = 'debug'
    debug = ""
else:
    queue = 'standard'
    debug = "## "

nstripes = min(round(log2(nnodes)), 32)

subprocess.run(['lfs', 'setstripe', '-c', '1', odir])
subprocess.run(['lfs', 'setstripe', '-c', str(nstripes), f'{odir}/restarts'])
os.chdir(odir)

slurm_script = f"slurm.{pid}"
script = f"""\
#!/bin/bash
# ---------------------------------------------------------------------
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={jobname}
#SBATCH --dependency=singleton
#SBATCH --output=out.{pid}.%j      # Name of stdout output file
#SBATCH --error=out.{pid}.%j       # Name of stderr error file
#SBATCH --qos={queue}
{debug}#SBATCH --reservation=debug
#SBATCH --nodes={nnodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time={walltime}       # Desired walltime (hh:mm:ss)
#SBATCH --time-min={mintime}   # Minimum acceptable walltime
## #SBATCH --mail-type=all
## #SBATCH --mail-user=towery@lanl.gov
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{pid}
export PYTHONPATH=$JOBDIR:$PYTHONPATH

cd {odir}
cp $JOBDIR/incompressible_hit.py ./

extra_args="--dt_init 4.36e-03 --base_ce2 1.7"  # --tlimit {tlimit} --dt_rst {dt_rst} --dt_stat {dt_stat} --no-forcing"
srun python incompressible_hit.py {sim_type} --restart -f {frame} $extra_args

wait

cp out.{pid}.${{SLURM_JOB_ID}} $JOBDIR/
echo "copied stdout log to $JOBDIR"
"""

with open(slurm_script, 'wt') as fh:
    fh.write(script)

print(f"submitting {slurm_script}...")
output = subprocess.run(['sbatch', slurm_script],
                        capture_output=True, text=True).stdout
print(output)

jobid = output.split()[-1]
jobdir = f"{notebook}/{today}/{jobid}.{pid}"
os.makedirs(jobdir, exist_ok=True)

shutil.copy(slurm_script, jobdir)
shutil.copy(f"{HOME}/{this_script}", jobdir)
shutil.copy(f"{HOME}/shencfd/problems/incompressible_hit.py", jobdir)
shutil.copytree(f"{HOME}/shencfd", f"{jobdir}/shencfd")
