#!/bin/bash
#SBATCH --job-name=wsc-cinf
#SBATCH --output=/extra/ucinlp0/rlogan/wsc-cinf.log
#SBATCH --time=10000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --cpus-per-task=32
#SBATCH --gpus=4
#SBATCH --mem=200GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    jobs/superglue/wsc_cinf.yaml
