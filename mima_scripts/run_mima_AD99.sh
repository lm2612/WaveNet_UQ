#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --tasks-per-node=32
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --constraint=CPU_GEN:RME
#SBATCH --job-name=mima_train_wavenet    # job name
#SBATCH --partition=serc        # partition
#SBATCH --output=/scratch/users/lauraman/WaveNetPyTorch/mima_jobs/%x_%j.out
#SBATCH --error=/scratch/users/lauraman/WaveNetPyTorch/mima_jobs/%x_%j.err

############### SLURM JOB FOR RUNNING MiMA WITH AD99 #####################################
##                                                                                      ##
## This script runs MiMA AD99 for generating the training data. Note it is highly       ##
## specific to computing architecture, directories, etc. and is intended as an example. ## 
##                                                                                      ##
##                                                                                      ##
## sbatch run_mima_AD99.sh                                                                ##
##                                                                                      ##
##                                                                                      ##
##########################################################################################


echo "*** NODE INFO ***"
echo $SLURM_NODELIST
#scontrol show job -d $SLURM_JOBID

echo "**************"

# RUN MiMA SIMULATION
#--------------------------------------------------------------------------------------------------------
# Set up relevant paths and modules
echo "**** SETUP ****"
ulimit -s unlimited

module load ifort
module load icc
module load netcdf-fortran
module load netcdf-c
module list
cwd=`pwd`

#--------------------------------------------------------------------------------------------------------
# Set up run directory
# Setup run directory
run=train_wavenet
executable=${GROUP_HOME}/${USER}/MiMA-machine-learning/build/mima.x
rundir=${SCRATCH}/WaveNetPyTorch/mima_runs/$run

#--------------------------------------------------------------------------------------------------------
# Copy executable to rundir
cp $executable $rundir/
# Copy inputs to rundir
cp -r ${GROUP_HOME}/${USER}/MiMA-machine-learning/inputs/* $rundir/
# No restart files needed as we start from cold and spin-up for 40 years

cd $rundir


#--------------------------------------------------------------------------------------------------------
echo "**** SET UP COMPLETE ****"

echo "******* RUN MIMA *******" 
[ ! -d RESTART ] && mkdir RESTART

N_PROCS=32
CCOMB=${GROUP_HOME}/${USER}/MiMA-machine-learning/build/mppnccombine
NYEAR=80

for ((i=0;i<=$NYEAR;i++))
do 
    echo "Running year $i ..."
    srun --ntasks $N_PROCS --mem 16G mima.x
    echo "Done year $i, now postprocessing... "
    $CCOMB -r atmos_daily_${i}.nc atmos_daily.nc.????
    $CCOMB -r atmos_avg_${i}.nc atmos_avg.nc.????
    $CCOMB -r atmos_all_${i}.nc atmos_all.nc.????
    $CCOMB -r atmos_davg_${i}.nc atmos_davg.nc.????
    $CCOMB -r atmos_dext_${i}.nc atmos_dext.nc.????
    cp RESTART/*res* INPUT/
    [ ! -d restart_history/restart_$i ] && mkdir -p restart_history/restart_$i
    cp -r RESTART/*res* restart_history/restart_$i/
    echo "Done postproc $i"
done

echo "******* RUN COMPLETE FOR ALL YEARS 1 to $NYEAR *****"
echo "done"

