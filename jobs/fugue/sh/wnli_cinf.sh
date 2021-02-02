#!/bin/bash
#SBATCH --job-name=wnli_cinf
#SBATCH --output=/extra/ucinlp0/rlogan/fugue_wnli_cinf.txt
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s3
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=400GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    --logdir results/wnli \
    jobs/fugue/yaml/wnli_cinf.yaml