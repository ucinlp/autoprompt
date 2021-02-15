#!/bin/bash
#SBATCH --job-name=superglue-cinf-mlm
#SBATCH --output=/extra/ucinlp0/rlogan/cinf-mlm-output.txt
#SBATCH --time=1000:00
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --gpus=8

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    jobs/superglue_continuous_mlm_roberta.yaml \
    --logdir results/roberta-continuous-mlm/
