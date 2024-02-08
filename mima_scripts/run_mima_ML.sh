#!/bin/bash
#SBATCH --nodes=1               # number of nodes
#SBATCH --tasks-per-node=16
#SBATCH --time=72:00:00
#SBATCH --mem=16G
#SBATCH --constraint=CPU_GEN:RME
#SBATCH --job-name=mima_ML      # job name
#SBATCH --partition=serc        # partition
#SBATCH --output=/scratch/users/lauraman/WaveNetPyTorch/mima_jobs/%x_%j_%a.out
#SBATCH --error=/scratch/users/lauraman/WaveNetPyTorch/mima_jobs/%x_%j_%a.err


############### ARRAY JOB FOR RUNNING MiMA ENSEMBLE ######################################
##                                                                                      ##
## This script runs MiMA-wavenet ensembles. Note it is highly specific to               ##
## computing architecture, directories, etc. and is intended as an example.             ## 
##                                                                                      ##
## Arrays are used for the random seed                                                  ##
## You must provide the start of the model name e.g., "wavenet"                         ##
## e.g., if your ML models are named wavenet (saved in directories named                ##
## "wavenet_zonal_seed1", "wavenet_meridional_seed1", "wavenet_zonal_seed2" ...)        ##
##                                                                                      ##
## sbatch --array=1-10 run_mima_ML.sh "wavenet"                                         ##
##                                                                                      ##
##                                                                                      ##
##########################################################################################

echo "*** NODE INFO ***"
echo $SLURM_NODELIST

model_name=${1?Error: provide the start of the model name as argument e.g. "wavenet"}

seed=${SLURM_ARRAY_TASK_ID}
echo "Model name: ${model_name}. Seed: ${seed}"

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
run=${model_name}_seed${seed}
executable=${GROUP_HOME}/${USER}/MiMA-machine-learning/build/mima.x
rundir=${SCRATCH}/WaveNetPyTorch/mima_runs/$run
inputdir=${SCRATCH}/WaveNetPyTorch/mima_runs/example_inputs/
#--------------------------------------------------------------------------------------------------------
# Create rundir if it does not already exist
[ ! -d $rundir ] && mkdir $rundir
# Copy input to rundir
cp -r $inputdir/* $rundir/

# Copy executable to rundir
cp $executable $rundir/

# Copy ML models to rundir
[ ! -d $rundir/MODELS ] && mkdir $rundir/MODELS
zonal_model_dir=${SCRATCH}/WaveNetPyTorch/models/${model_name}_zonal_seed${seed}/
cp ${zonal_model_dir}/zonal_wavenet.pth $rundir/MODELS/

meridional_model_dir=${SCRATCH}/WaveNetPyTorch/models/${model_name}_meridional_seed${seed}/
cp ${meridional_model_dir}/meridional_wavenet.pth $rundir/MODELS/

cd $rundir

#--------------------------------------------------------------------------------------------------------
echo "**** SET UP COMPLETE ****"

echo "******* RUN MIMA *******" 
[ ! -d RESTART ] && mkdir RESTART

N_PROCS=16
CCOMB=${GROUP_HOME}/${USER}/MiMA-machine-learning/build/mppnccombine
NYEAR=65
i=45

# Start at year 45, and run for 20 years until year 65. 
while [ $i -le $NYEAR ]
do 
    echo "Running year $i ..."
    srun --ntasks $N_PROCS --mem 16G mima.x
    echo "Success. Done year $i, now postprocessing... "
    $CCOMB -r atmos_daily_${i}.nc atmos_daily.nc.????
    $CCOMB -r atmos_avg_${i}.nc atmos_avg.nc.????
    $CCOMB -r atmos_all_${i}.nc atmos_all.nc.????
    cp RESTART/*res* INPUT/
    [ ! -d restart_history/restart_$i ] && mkdir -p restart_history/restart_$i
    cp -r RESTART/*res* restart_history/restart_$i/

    echo "Done postproc $i"
    ((i++))
done

echo "******* RUN COMPLETE FOR ALL YEARS UP TO $NYEAR *****"
echo "done"

