#!/bin/bash
#SBATCH --time=100000:00
#SBATCH --partition=ava_s.p
#SBATCH --gpus=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=30GB
#SBATCH --exclude=ava-s4

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/crossval.py \
    --logdir crossval-results/ \
    $1
