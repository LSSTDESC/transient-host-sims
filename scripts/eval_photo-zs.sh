#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#SBATCH --output=R-%x.%j.out
#SBATCH --array=1-4
#SBATCH --account=m1727

source /global/u2/a/aimalz/ve3_elasticc/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
srun /global/u2/a/aimalz/ve3_elasticc/bin/python eval_photo-zs.py