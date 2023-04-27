#!/bin/bash

#SBATCH --nodes=1     
#SBATCH --tasks-per-node=1
#SBATCH --cpus-per-task=4

#SBATCH --time=05:00:00  
#SBATCH --output=2_station_strip_ppo.out
#SBATCH --error=2_station_strip_ppo.err

source activate new
python 2_station_strip_ppo.py