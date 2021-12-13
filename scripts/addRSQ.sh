#!/bin/bash
#SBATCH --constraint=haswell
#SBATCH --nodes=1
#SBATCH --time=01:00:00
#SBATCH --qos=regular
#module load python
#source activate /global/u2/m/mlokken/software/mydescnew
eval "$(conda shell.bash hook)"
conda activate pzflow
export HDF5_USE_FILE_LOCKING=FALSE
export DESC_GCR_SITE='nersc'
python addRSQ.py
