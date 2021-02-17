#!/bin/bash
#SBATCH --job-name=cola_cinf
#SBATCH --output=/extra/ucinlp0/rlogan/fugue_cola_cinf.txt
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=400GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    --logdir results/cola \
    jobs/fugue/yaml/cola_cinf.yaml
