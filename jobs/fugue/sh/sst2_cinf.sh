#!/bin/bash
#SBATCH --job-name=sst2_cinf
#SBATCH --output=/extra/ucinlp0/rlogan/fugue_sst2_cinf.txt
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s3
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=400GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    --logdir results/sst2 \
    jobs/fugue/yaml/sst2_cinf.yaml
