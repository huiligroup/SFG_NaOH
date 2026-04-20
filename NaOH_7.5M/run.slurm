#!/bin/bash

#SBATCH --job-name=NaOH75M_iPI
#SBATCH --partition=chem
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=12
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --account=name_grp

ulimit -d unlimited
ulimit -s unlimited
ulimit -t unlimited
ulimit -v unlimited

source /public/software/hpc4you/anaconda3/etc/profile.d/conda.sh
conda activate lammps-v2023.11.21

cd ${SLURM_SUBMIT_DIR}

IPIPATH=/path/to/i-pi
lmp=$(which lmp)

HOST=$(hostname)
NTIME=36000

echo {"init",$(date -u)} >> LIST
grep '<step>' RESTART >> LIST 2>/dev/null

######### CHECK RESTART FILE ###############
if [ ! -f "RESTART" ]
then
  cp input.xml RESTART
fi
######### CHECK RESTART FILE ###############

# Run the program:
python -u ${IPIPATH}/i-pi RESTART > log.ipi 2>&1 &
sleep 10

# Run the program:
mpirun -n 2 ${lmp} < ipi_md1.lmp > log.lmp1 2>&1 &
mpirun -n 2 ${lmp} < ipi_md2.lmp > log.lmp2 2>&1 &
mpirun -n 2 ${lmp} < ipi_md3.lmp > log.lmp3 2>&1 &
mpirun -n 2 ${lmp} < ipi_md4.lmp > log.lmp4 2>&1 &
mpirun -n 2 ${lmp} < ipi_md5.lmp > log.lmp5 2>&1 &
mpirun -n 2 ${lmp} < ipi_md6.lmp > log.lmp6 2>&1

wait

sleep 10
touch EXIT
sleep 10
rm -f EXIT

grep '<step>' RESTART >> LIST 2>/dev/null
echo {"Final and restart",$(date -u)} >> LIST
