#!/bin/bash
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --gpus=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=50GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/crossval.py \
    --logdir crossval-results/ \
    $1
