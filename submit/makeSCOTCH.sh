#!/bin/bash
#SBATCH --qos=regular
#SBATCH --time=6:00:00
#SBATCH --nodes=1
#SBATCH -e slurm-%j.out
#SBATCH -o slurm-%j.out
#SBATCH --tasks-per-node=1
#SBATCH --constraint=haswell

source ~/.snana

date

cd /global/homes/a/agaglian/transient-host-sims/scripts
activate pzflow

python CreateSCOTCH.py

date
