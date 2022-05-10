#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --output=R-%x.%j.out
#SBATCH --array=0
#SBATCH --account=m1727

source /global/u2/a/aimalz/ve3_elasticc/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
srun -N 1 -n 1 -c 1 /global/u2/a/aimalz/ve3_elasticc/bin/python model_photo-zs.py