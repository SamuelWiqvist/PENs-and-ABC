#!/bin/sh

# Set up for run:

# need this since I use a LU project
#SBATCH -A lu2018-2-22
#SBATCH -p lu

# use gpu nodes
#SBATCH --mem-per-cpu=12000


# time consumption HH:MM:SS
#SBATCH -t 100:30:00

# name for script
#SBATCH -J univ_alphastable_multi_ABC_dnn_small2

# controll job outputs
#SBATCH -o lunarc_output/univaralphastable/outputs_alphastable_multiple_ABC_dnn_small_%j.out
#SBATCH -e lunarc_output/univaralphastable/errors_alphastable_multiple_ABC_dnn_small_%j.err

# notification
#SBATCH --mail-user=samuel.wiqvist@matstat.lu.se
#SBATCH --mail-type=ALL

# we need to load the cuda stuff here!

# load modules

ml load GCC/6.4.0-2.28
ml load OpenMPI/2.1.2
ml load julia/1.0.0

# set correct path
pwd
cd ..
pwd

# run program
julia /home/samwiq/'ABC and deep learning project'/abc-dl/src/'alpha stable dist'/multiple_ABC_runs_mlp.jl mlp standard 250 25 25 12 3 0 small
