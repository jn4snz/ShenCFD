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

from math import pi, ceil, log2, sqrt
import os
import shutil
import subprocess
import textwrap
import time

ncpus = int(os.getenv('CPUS_PER_NODE'))
mpiprocs = ncpus

HOME = os.getenv('HOME')
WORK = os.getenv('WORK')

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

#############################################################################
project = 'DNS'
sim_type = 'dns'
notebook = f'{HOME}/LAB_NOTEBOOK/shencfd/{project}'

K = 1024
N = 2048
kfLow = 4
kfHigh = 6

pid = f"DNS_K{K}_N{N}_F{kfLow}k{kfHigh}"
odir = f"{WORK}/{project}/{pid}"
jobname = pid

ntasks = 8192
walltime = "08:00:00"
mintime = "08:00:00"

DRY_RUN = False
INIT_ONLY = False
dealiasing = '2/3'

Np = 1024
ipid = f"DNS_K{K}_N{Np}_F{kfLow}k{kfHigh}"
idir = f"{WORK}/DNS/{ipid}"
frame = 11

# -----------------------------------------------------------------
tlim = 10.0
trst = 0.5
tstat = 1.0

eps = 1.0
ell_L = 0.19

eta = 2 / K
nu = (eps * eta**4)**(1/3)
ell = 2 * pi * ell_L
KE_eq = (ell * eps)**(2/3)
tau_ell = KE_eq / eps
tau_eta = sqrt(nu / eps)

t_ell = float(f"{tau_ell:0.3e}")  # rounding to 4 significant digits
t_eta = float(f"{tau_eta:0.3e}")  # rounding to 4 significant digits

dt_rst = t_ell * trst
dt_stat = t_eta * tstat
tlimit = tlim * t_ell

# -----------------------------------------------------------------
nnodes = ceil(ntasks / mpiprocs)
if nnodes < 3:
    queue = 'debug'
else:
    queue = 'standard'
nstripes = min(round(log2(nnodes)), 32)

inputs_file = f"sheninputs.{pid}"
slurm_script = f"slurm.{pid}"

os.makedirs(f"{odir}/restarts", exist_ok=True)
subprocess.run(['lfs', 'setstripe', '-c', '1', odir])
subprocess.run(['lfs', 'setstripe', '-c', str(nstripes), f'{odir}/restarts'])
os.chdir(odir)

inputs = f"""\
pid           = {pid}
odir          = {odir}
N             = {N}
nu            = {nu}
dealiasing    = {dealiasing}

# init_cond     = random
# KE_init       = {2*KE_eq:0.5e}

restart       = True
init_pid      = {ipid}
idir          = {idir}
frame         = {frame}

forcing       = True
epsilon       = {eps}
kfLow         = {kfLow}
kfHigh        = {kfHigh}

# t_init        = 0.0
# dt_init       = 1e-5
tlimit        = {tlimit}
dt_rst        = {dt_rst}
dt_stat       = {dt_stat}
# cfl           = 0.9
# Co_diff       = 0.9
# Co_keps       = 0.1

dry_run       = {DRY_RUN}
init_only     = {INIT_ONLY}
"""

with open(inputs_file, 'wt') as fh:
    fh.write(textwrap.dedent(inputs))

if queue == "debug":
    debug = ""
else:
    debug = "## "

script = f"""\
#!/bin/bash
# ---------------------------------------------------------------------
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
cp $JOBDIR/{inputs_file} ./

srun python incompressible_hit.py {sim_type} @{inputs_file}

wait

cp out.{pid}.${{SLURM_JOB_ID}} $JOBDIR/
echo "copied stdout log to $JOBDIR"
"""

with open(slurm_script, 'wt') as fh:
    fh.write(textwrap.dedent(script))

print(f"submitting {slurm_script}...")
output = subprocess.run(['sbatch', slurm_script],
                        capture_output=True, text=True).stdout
print(output)

jobid = output.split()[-1]
jobdir = f"{notebook}/{today}/{jobid}.{pid}"
os.makedirs(jobdir, exist_ok=True)

shutil.copy(inputs_file, jobdir)
shutil.copy(slurm_script, jobdir)
shutil.copy(f"{HOME}/{this_script}", jobdir)
shutil.copy(f"{HOME}/shencfd/problems/incompressible_hit.py", jobdir)
shutil.copytree(f"{HOME}/shencfd", f"{jobdir}/shencfd")
