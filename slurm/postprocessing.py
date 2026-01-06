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

from math import ceil
import os
import shutil
import subprocess
import time

#############################################################################
# Basically never touch any of this unless switching systems
#############################################################################
ncpus = int(os.getenv('CPUS_PER_NODE'))
mpiprocs = ncpus

HOME = os.getenv('HOME')
WORK = os.getenv('WORK')
ACCOUNT = 'w22_mass-app'

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

project = 'DNS'
os.makedirs(f"{WORK}/{project}/", exist_ok=True)
notebook = f'{HOME}/LAB_NOTEBOOK/shencfd/{project}'

#############################################################################
problem = 'postprocess_hybrid_stats.py'
jobname = 'postprocess_stats'

ntasks = 1024
walltime = "01:00:00"
mintime = "01:00:00"

nnodes = ceil(ntasks / mpiprocs)
queue = 'standard'
slurm_script = f"slurm.{jobname}"

odir = f'{WORK}/DNS/postprocess'
os.makedirs(odir, exist_ok=True)
subprocess.run(['lfs', 'setstripe', '-c', '1', odir])
os.chdir(odir)

script = f"""\
#!/bin/bash
# ---------------------------------------------------------------------
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={jobname}
#SBATCH --dependency=singleton
#SBATCH --output=out.{jobname}.%j      # Name of stdout output file
#SBATCH --error=out.{jobname}.%j       # Name of stderr error file
#SBATCH --qos={queue}

#SBATCH --nodes={nnodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time={walltime}       # Desired walltime (hh:mm:ss)
#SBATCH --time-min={mintime}   # Minimum acceptable walltime
# ---------------------------------------------------------------------
export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{jobname}
export PYTHONPATH=$JOBDIR:$PYTHONPATH

cp $JOBDIR/{problem} ./

srun python {problem}

wait

cp out.{jobname}.${{SLURM_JOB_ID}} $JOBDIR/
echo "copied stdout log to $JOBDIR"
"""

with open(slurm_script, 'wt') as fh:
    fh.write(script)

print(f"submitting {slurm_script}...")
output = subprocess.run(['sbatch', slurm_script],
                        capture_output=True, text=True).stdout
print(output)

jobid = output.split()[-1]
jobdir = f"{notebook}/{today}/{jobid}.{jobname}"
os.makedirs(jobdir, exist_ok=True)

shutil.copy(slurm_script, jobdir)
shutil.copy(f"{HOME}/{this_script}", jobdir)
shutil.copy(f"{HOME}/shencfd/problems/{problem}",
            jobdir)
shutil.copytree(f"{HOME}/shencfd", f"{jobdir}/shencfd")
