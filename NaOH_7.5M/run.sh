IPIPATH=<Complete with i-pi path>
lmp=<Complete with lammps path>

HOST=$(hostname)
NTIME=36000

echo  {"init",$( date -u)} >>LIST
grep '<step>'  RESTART >> LIST

######### CHECK RESTART FILE ###############
if [ ! -f "RESTART"   ]
 then
  cp input.xml RESTART
fi
######### CHECK RESTART FILE ###############

# Run the program:

python -u ${IPIPATH}/i-pi RESTART &> log.ipi &
sleep 10

# Run the program:
mpirun -n 2 ${lmp} < ipi_md1.lmp > log.lmp1 &
mpirun -n 2 ${lmp} < ipi_md2.lmp > log.lmp2 &
mpirun -n 2 ${lmp} < ipi_md3.lmp > log.lmp3 &
mpirun -n 2 ${lmp} < ipi_md4.lmp > log.lmp4 &
mpirun -n 2 ${lmp} < ipi_md5.lmp > log.lmp5 &
mpirun -n 2 ${lmp} < ipi_md6.lmp > log.lmp6

sleep 10
touch EXIT
sleep 10
rm EXIT

grep '<step>'  RESTART >> LIST
echo  {"Final and restart",$( date -u)} >>LIST
