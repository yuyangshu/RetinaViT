#!/bin/bash

#PBS -N bv_mpi
#PBS -P ey69
#PBS -q gpuvolta
#PBS -l ncpus=192
#PBS -l ngpus=16
#PBS -l mem=1536GB
#PBS -l jobfs=128GB
#PBS -l storage=gdata/ey69
#PBS -l walltime=24:00:00
#PBS -l wd
#PBS -j oe
#PBS -M y.shu@unsw.edu.au
#PBS -m abe



module load python3/3.10.4
module load openmpi/4.1.5

source /home/561/ys4871/venv/bin/activate

# jax installation
# python3 -m pip install -r big_vision/requirements.txt
# python3 -m pip install -U "jax[cuda12]"

# mpi installation
# python3 -m pip install mpi4jax

export TFDS_DATA_DIR=/g/data/ey69/ys4871/tensorflow_datasets
export BV_JAX_INIT=true
export MPI4JAX_USE_CUDA_MPI=1

mpirun --np 16 --map-by numa:SPAN --bind-to numa python3 -m big_vision.train --config big_vision/configs/vit_s16_i1k.py --workdir $PBS_JOBID/`date '+%m-%d_%H%M'`
