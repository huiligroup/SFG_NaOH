#!/bin/bash

#SBATCH --job-name=NaOH
#SBATCH --partition=chem-lh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=12
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --account=lihui_grp

ulimit -d unlimited
ulimit -s unlimited
ulimit -t unlimited
ulimit -v unlimited

source /public/software/hpc4you/anaconda3/etc/profile.d/conda.sh
conda activate lammps-v2023.11.21
source /public/home/lihui_grp/baiying/i-pi/env.sh

cd "${SLURM_SUBMIT_DIR}"

IPIPATH=/public/home/lihui_grp/baiying/i-pi/bin
lmp=$(which lmp)

echo "{\"init\",$(date -u)}" >> LIST
grep '<step>' RESTART >> LIST 2>/dev/null

if [ ! -f "RESTART"   ]
 then
  cp input.xml RESTART
fi

python -u ${IPIPATH}/i-pi RESTART > log.ipi 2>&1 &
sleep 10

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
echo "{\"Final and restart\",$(date -u)}" >> LIST
