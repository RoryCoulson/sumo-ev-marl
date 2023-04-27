#!/bin/bash

#SBATCH --nodes=1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=03:00:00  
#SBATCH --output=4x4_grid_dqn.out
#SBATCH --error=4x4_grid_dqn.err

source activate new
python 4x4_grid_dqn.py