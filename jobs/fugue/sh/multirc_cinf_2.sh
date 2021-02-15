#!/bin/bash
#SBATCH --job-name=multirc_cinf_2
#SBATCH --output=/extra/ucinlp0/rlogan/fugue_multirc_cinf_2.txt
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --gpus=8
#SBATCH --cpus-per-task=32
#SBATCH --mem=400GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    --logdir results/multirc \
    jobs/fugue/yaml/multirc_cinf_2.yaml
