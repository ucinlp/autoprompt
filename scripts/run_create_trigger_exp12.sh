#!/bin/bash
# Experiment 12
# Task: fact retrieval (resplit)
# Model: BERT
# Batch sizes: 28
# Iters: 1000
# Filtering: True

datadir=$1
logfile=$2

# Clear files
cat /dev/null > $logfile
cat /dev/null > ${logfile}.log

for path in $datadir/*; do
    filename=$(basename "$path")
    CUDA_VISIBLE_DEVICES=1 python -m lmat.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '[CLS] {sub_label} [T] [T] [T] [T] [T] [T] [T] [P] . [SEP]' \
        --num_cand 10 \
        --accumulation-steps 1 \
        --model-name bert-base-cased \
        --bsz 28 \
        --eval-size 28 \
        --iters 1000 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama >> $logfile 2>> ${logfile}.log
done
