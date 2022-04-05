#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug
#SBATCH --output=R-%x.%j.out
#SBATCH --array=0-2

source /global/u2/a/aimalz/ve3_elasticc/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
srun /global/u2/a/aimalz/ve3_elasticc/bin/python model_photo-zs.py