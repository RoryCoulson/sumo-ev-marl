#!/bin/bash

#SBATCH --nodes=1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=06:00:00  
#SBATCH --output=berlin_greedy.out
#SBATCH --error=berlin_greedy.err

source activate new
python berlin_greedy.py