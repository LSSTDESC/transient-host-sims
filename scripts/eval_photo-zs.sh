#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --qos=regular
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --array=1,4
#SBATCH --time=03:00:00
#SBATCH --output=R-%x.%j.out
#SBATCH --account=m1727

source /global/u2/a/aimalz/ve3_elasticc/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
srun /global/u2/a/aimalz/ve3_elasticc/bin/python eval_photo-zs.py