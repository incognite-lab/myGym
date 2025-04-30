#!/bin/bash
#SBATCH --job-name=myGym3.1
#SBATCH --cpus-per-task=56   
#SBATCH --mem=128G
#SBATCH --time=95:00:00
#SBATCH --partition=compute
# Define and create a unique scratch directory for this job
# /lscratch is local ssd disk on particular node which is faster
# than your network home dir
#SCRATCH_DIRECTORY=/lscratch/${USER}/${SLURM_JOBID}.FM_tiago_rotslide_m2
#mkdir -p ${SCRATCH_DIRECTORY}
#cd ${SCRATCH_DIRECTORY}


# You can copy everything you need to the scratch directory
# ${SLURM_SUBMIT_DIR} points to the path where this script was
# submitted from (usually in your network home dir)
#cp -r ${SLURM_SUBMIT_DIR}/myGym/myGym/ ${SCRATCH_DIRECTORY}

# This is where the actual work is done. In this case, the script only waits.
# The time command is optional, but it may give you a hint on how long the
# command worked
echo AGMDW
cd ..
python train.py --config ./configs/train_AGMDW_RDDL.json --algo multippo -i 50


# After the job is done we copy our output back to $SLURM_SUBMIT_DIR
#cp -r ${SCRATCH_DIRECTORY} ${SLURM_SUBMIT_DIR}/output

# In addition to the copied files, you will also find a file called
# slurm-1234.out in the submit directory. This file will contain all output that
# was produced during runtime, i.e. stdout and stderr.

# After everything is saved to the home directory, delete the work directory to
# save space on /lscratch
# old files in /lscratch will be deleted automatically after some time
#cd ${SLURM_SUBMIT_DIR}
#srm -rf ${SCRATCH_DIRECTORY}




