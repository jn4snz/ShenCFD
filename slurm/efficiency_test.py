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
import datetime

############################################################################
# Basically never touch any of this
############################################################################
ncpus = 36
mpiprocs = 36

HOME = os.getenv('HOME')
WORK = os.getenv('WORK')
TURQ = "/lustre/scratch3/turquoise/towery"

this_script = os.path.basename(__file__)
today = time.strftime('%Y_%m_%d')

project = 'efficiency'
os.makedirs(f"{WORK}/{project}/", exist_ok=True)
notebook = f'{HOME}/LAB_NOTEBOOK/shencfd/{project}'

############################################################################
K = 512
Np = 512
kfLow = 3
kfHigh = 5
fstr = f"F{kfLow}k{kfHigh}_decay"
ipid = f"DNS_K{K}_N{Np}_{fstr}"
idir = f"{TURQ}/DNS/{ipid}"
frame = 39

eps = 1.0
ell_L = 0.23
eta = 2 / K
nu = (eps * eta**4)**(1/3)
ell = 2 * pi * ell_L
KE_eq = (ell * eps)**(2/3)
t_ell = float(f"{ell / sqrt(2/3 * KE_eq):0.5e}")  # round to 6 sigfigs

#############################################################################
N = 256
sim_type = ('dns', 'hybrid')
init_cond = ('random', 'hybrid')
Cmu_model = ('const', 'dynamic')
Ce2_model = ('const', 'const')
tlim = (1.0, 0.1)

nx = (256, 384, 256)
dealias = ('3/2', '2/3', 'partial')
dstr = ('3_2', '2_3', 'partial')

wt_mins = 60

ftype = 'gaussian'
iratio = 4/3
tratio = 8/3

ACCOUNT = 'w21_melt'
TEST = False  # quit script after first iteration of top-line loop

#############################################################################
nruns = len(sim_type) * len(dealias)
ntasks = N * 2
nnodes = ceil(ntasks / mpiprocs)
nstripes = min(round(log2(nnodes)), 32)
walltime = str(datetime.timedelta(minutes=wt_mins))
mintime = walltime

jobname = f"efficiency_test_N{N}"
slurm_script = f"slurm.{jobname}"
os.chdir(f"{WORK}/{project}/")

script = f"""\
#!/bin/bash
# ---------------------------------------------------------------------
#SBATCH --account={ACCOUNT}
#SBATCH --job-name={jobname}
#SBATCH --dependency=singleton
#SBATCH --output=%j.{jobname}.out      # Name of stdout output file
#SBATCH --error=%j.{jobname}.out       # Name of stderr error file
#SBATCH --qos=standard

#SBATCH --nodes={nnodes}
#SBATCH --ntasks={ntasks}
#SBATCH --time={walltime}       # Desired walltime (hh:mm:ss)
#SBATCH --time-min={mintime}    # Minimum acceptable walltime

#SBATCH --begin=now+{nruns}     # do not start job until after n seconds
# ---------------------------------------------------------------------

export OMP_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

JOBDIR={notebook}/{today}/${{SLURM_JOB_ID}}.{jobname}
export PYTHONPATH=$JOBDIR:$PYTHONPATH

for sim in {' '.join(sim_type)}
do
  for dstr in {' '.join(dstr)}
  do

    pid=${{sim}}_K{N}_N{N}_${{dstr}}
    outfile=out.${{pid}}.${{SLURM_JOB_ID}}

    cd {WORK}/{project}/$pid
    cp $JOBDIR/incompressible_hit.py ./
    cp $JOBDIR/sheninputs.$pid ./

    echo running $pid
    srun python incompressible_hit.py ${{sim}} @sheninputs.$pid &> $outfile

    wait
    cp $outfile $JOBDIR/
    echo "copied out.$pid to $JOBDIR"

  done
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
for i in range(len(sim_type)):
    for j in range(len(dealias)):

        tlimit = t_ell * tlim[i]
        pid = f'{sim_type[i]}_K{N}_N{N}_{dstr[j]}'

        odir = f"{WORK}/{project}/{pid}"
        inputs_file = f"sheninputs.{pid}"
        os.makedirs(f"{odir}/restarts", exist_ok=True)
        subprocess.run(['lfs', 'setstripe', '-c', '1', odir])
        subprocess.run(['lfs', 'setstripe', '-c', str(nstripes),
                        f'{odir}/restarts'])

        inputs = f"""\
        pid           = {pid}
        odir          = {odir}
        N             = {nx[j]}
        nu            = {nu}
        dealiasing    = {dealias[j]}

        init_cond     = {init_cond[i]}
        init_pid      = {ipid}
        idir          = {idir}
        frame         = {frame}

        Cmu_model     = {Cmu_model[i]}
        Ce2_model     = {Ce2_model[i]}
        filter_type   = {ftype}
        ic_ratio      = {iratio}
        test_ratio    = {tratio}

        tlimit        = {tlimit}
        dt_rst        = {tlimit}
        dt_stat       = {tlimit}
        """

        with open(inputs_file, 'wt') as fh:
            fh.write(textwrap.dedent(inputs))

    ##################################################################
    if TEST:
        break
