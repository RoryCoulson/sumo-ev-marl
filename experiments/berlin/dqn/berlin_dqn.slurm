#!/bin/bash

#SBATCH --nodes=1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=06:00:00  
#SBATCH --output=berlin_dqn.out
#SBATCH --error=berlin_dqn.err

source activate new
python berlin_dqn.py