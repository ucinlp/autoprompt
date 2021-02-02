#!/bin/bash
#SBATCH --job-name=superglue-cinf
#SBATCH --output=/extra/ucinlp0/rlogan/cinf-clf-output.txt
#SBATCH --time=1000:0
#SBATCH --partition=ava_s.p
#SBATCH --nodelist=ava-s0
#SBATCH --gpus=8

srun /home/rlogan/miniconda3/envs/autoprompt/bin/python3.7 \
    scripts/launch.py \
    jobs/superglue_continuous_clf_roberta.yaml \
    --logdir
