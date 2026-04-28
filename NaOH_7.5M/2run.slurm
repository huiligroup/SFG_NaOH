#!/bin/bash

#SBATCH --job-name=NaOH
#SBATCH --partition=chem-lh
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks-per-node=12
#SBATCH --error=%j.err
#SBATCH --output=%j.out
#SBATCH --account=lihui_grp
#SBATCH --signal=B:TERM@300

ulimit -d unlimited
ulimit -s unlimited
ulimit -t unlimited
ulimit -v unlimited

source /public/software/hpc4you/anaconda3/etc/profile.d/conda.sh
conda activate lammps-v2023.11.21
source /public/home/lihui_grp/baiying/i-pi/env.sh

cd "${SLURM_SUBMIT_DIR}" || exit 1

# 防止上一次残留 EXIT 导致 i-PI 一启动就 soft exit
rm -f EXIT

# 防止每个 MPI rank 又偷偷开多线程，造成 CPU 过度占用
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1
export OPENBLAS_NUM_THREADS=1

IPIPATH=/public/home/lihui_grp/baiying/i-pi/bin
lmp=$(which lmp)

# 第一次运行时，用 input.xml 初始化 RESTART；
# 续跑时，如果 RESTART 已存在，则继续使用 RESTART
if [ ! -f "RESTART" ]; then
  cp input.xml RESTART
fi

echo "{\"init\",$(date -u)}" >> LIST
grep '<step>' RESTART >> LIST 2>/dev/null

# 如果作业收到终止信号，提前通知 i-PI 干净退出并写 RESTART
soft_exit() {
  echo "{\"soft_exit_signal\",$(date -u)}" >> LIST
  touch EXIT
}

trap soft_exit TERM INT

# 启动 i-PI 主控程序
python -u "${IPIPATH}/i-pi" RESTART > log.ipi 2>&1 &
ipi_pid=$!

sleep 10

# 启动 6 个 LAMMPS/NNP 客户端
mpirun -n 2 "${lmp}" < ipi_md1.lmp > log.lmp1 2>&1 &
p1=$!

mpirun -n 2 "${lmp}" < ipi_md2.lmp > log.lmp2 2>&1 &
p2=$!

mpirun -n 2 "${lmp}" < ipi_md3.lmp > log.lmp3 2>&1 &
p3=$!

mpirun -n 2 "${lmp}" < ipi_md4.lmp > log.lmp4 2>&1 &
p4=$!

mpirun -n 2 "${lmp}" < ipi_md5.lmp > log.lmp5 2>&1 &
p5=$!

mpirun -n 2 "${lmp}" < ipi_md6.lmp > log.lmp6 2>&1 &
p6=$!

# 等待 i-PI 正常完成 total_steps 或 total_time。
# 如果 wait 被 Slurm 的 TERM 信号打断，但 i-PI 还活着，就继续等，
# 避免 i-PI 没来得及写完 RESTART。
while true; do
  wait "${ipi_pid}"
  ipi_status=$?

  if kill -0 "${ipi_pid}" 2>/dev/null; then
    echo "{\"ipi_wait_interrupted\",$(date -u)}" >> LIST
    sleep 5
  else
    break
  fi
done

# 等待 6 个 LAMMPS 客户端收尾退出
wait "${p1}"
s1=$?
wait "${p2}"
s2=$?
wait "${p3}"
s3=$?
wait "${p4}"
s4=$?
wait "${p5}"
s5=$?
wait "${p6}"
s6=$?

# 正常结束后清理 EXIT，避免影响下一次续跑
rm -f EXIT

grep '<step>' RESTART >> LIST 2>/dev/null
echo "{\"Final and restart\",$(date -u)}" >> LIST

echo "i-PI exit status: ${ipi_status}" >> LIST
echo "LAMMPS exit status: ${s1} ${s2} ${s3} ${s4} ${s5} ${s6}" >> LIST

# 如果 i-PI 或任意一个 LAMMPS 异常退出，则让 Slurm 看到非零退出状态
if [ "${ipi_status}" -ne 0 ] || [ "${s1}" -ne 0 ] || [ "${s2}" -ne 0 ] || [ "${s3}" -ne 0 ] || [ "${s4}" -ne 0 ] || [ "${s5}" -ne 0 ] || [ "${s6}" -ne 0 ]; then
  exit 1
fi

exit 0
