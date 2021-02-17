#!/bin/bash
#SBATCH --job-name=cola_cinf_corrected
#SBATCH --output=/extra/ucinlp0/rlogan/fugue_cola_cinf_corrected.txt
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --gpus=6
#SBATCH --cpus-per-task=8
#SBATCH --mem=300GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    --logdir results/cola \
    jobs/fugue/yaml/cola_cinf_corrected.yaml
