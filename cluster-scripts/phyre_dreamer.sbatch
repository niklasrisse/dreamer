#!/bin/bash
#SBATCH --gres=gpu:1
#SBATCH --partition=cpu
#SBATCH --cpus-per-task=4
#SBATCH --nodelist=schlaubi
#SBATCH --mail-type=ALL
#SBATCH --job-name=p_dream
#SBATCH --time=100:00:00

module load cuda/10.1
srun phyre_dreamer.sh
