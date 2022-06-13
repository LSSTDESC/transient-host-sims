#!/bin/bash
#SBATCH --nodes=1
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=1
#SBATCH --mem=4000
#SBATCH --time=00:30:00
#SBATCH --constraint=haswell
#SBATCH --qos=regular
#SBATCH --output=R-%x.%j.out
#SBATCH --array=0-31
#SBATCH --account=m1727


source /global/u2/a/aimalz/ve3_elasticc/bin/activate
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
srun /usr/bin/time /global/u2/a/aimalz/ve3_elasticc/bin/python eval_photo-zs.py