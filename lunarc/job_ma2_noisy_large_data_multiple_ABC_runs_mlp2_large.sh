#!/bin/sh


# Set up for run:

# need this since I use a LU project
#SBATCH -A lu2018-2-22

# use gpu nodes
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=3100

# #SBATCH -N 1
# #SBATCH -n 1


# time consumption HH:MM:SS
#SBATCH -t 04:00:00

# name for script
#SBATCH -J ma2_multi_ABC_mlp2

# controll job outputs
#SBATCH -o lunarc_output/MA2_noisy_data/outputs_ma2_multiple_ABC_runs_mlp_%j.out
#SBATCH -e lunarc_output/MA2_noisy_data/errors_ma2_multiple_ABC_runs_mlp_%j.err

# notification
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se
#SBATCH --mail-type=ALL

# load modules

ml load GCC/6.4.0-2.28
ml load CUDA/9.1.85
ml load OpenMPI/2.1.2
ml load cuDNN/7.0.5.15
ml load julia/1.0.0

nvidia-smi

# set correct path
pwd
cd ..
pwd

# run program
julia /home/samwiq/'ABC and deep learning project'/abc-dl/src/'MA2 noisy data'/multiple_ABC_runs_mlp.jl mlp standard 250 2 0 large

# run using
# sbatch test_gpu.sh
