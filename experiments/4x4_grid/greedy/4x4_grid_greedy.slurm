#!/bin/bash

#SBATCH --nodes=1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=06:00:00  
#SBATCH --output=4x4_grid_greedy.out
#SBATCH --error=4x4_grid_greedy.err

source activate new
python 4x4_grid_greedy.py