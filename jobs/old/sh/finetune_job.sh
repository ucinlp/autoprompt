#!/bin/bash
#SBATCH --job-name=finetune-roberta
#SBATCH --output=/extra/ucinlp0/rlogan/finetune-roberta-output.txt
#SBATCH --time=10000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s1
#SBATCH --cpus-per-task=8
#SBATCH --gpus=8
#SBATCH --mem=400GB

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    jobs/superglue_finetune_roberta.yaml
