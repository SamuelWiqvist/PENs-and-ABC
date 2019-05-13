#!/bin/sh


# Set up for run:

# need this since I use a LU project
#SBATCH -A lu2018-2-22

# use gpu nodes
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --mem-per-cpu=11000

# #SBATCH -N 1
# #SBATCH -n 1


# time consumption HH:MM:SS
#SBATCH -t 100:00:00

# name for script
#SBATCH -J ar2_pen

# controll job outputs
#SBATCH -o lunarc_output/AR2/outputs_ar2_pen_%j.out
#SBATCH -e lunarc_output/AR2/errors_ar2_pen_%j.err

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
julia /home/samwiq/'ABC and deep learning project'/abc-dl/src/AR2/train_pen.jl standard 50 4 1

# run using
# sbatch test_gpu.sh
