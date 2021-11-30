#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=00:30:00
#SBATCH --qos=debug
module load python
source activate /global/u2/m/mlokken/mydescdev
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
python apply_pzflow_dc2full.py