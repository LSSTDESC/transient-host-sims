#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug
module load python
source activate /global/u2/m/mlokken/software/mydescnew
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
python condition_pzflow.py