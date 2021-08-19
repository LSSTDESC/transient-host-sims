#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=12:00:00
#SBATCH --nodes=1
#SBATCH -e slurm-%j.out
#SBATCH -o slurm-%j.out
#SBATCH --tasks-per-node=32
#SBATCH --constraint=haswell

source ~/.snana

date

cd /global/homes/a/agaglian/transient-host-sims/scripts
conda activate elasticc

python hashing_sbatch.py "SN Ia"
python hashing_sbatch.py "SN II"
python hashing_sbatch.py "SLSN-I"
python hashing_sbatch.py "SN IIP"
python hashing_sbatch.py "SN IIb"
python hashing_sbatch.py "SN IIn"
python hashing_sbatch.py "SN Ib"
python hashing_sbatch.py "SN Ic"
python hashing_sbatch.py "SN Ibc"

date
