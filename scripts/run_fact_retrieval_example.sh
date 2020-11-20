#!/bin/bash
# Experiment 8
# Task: fact retrieval
# Model: RoBERTa
# Batch sizes: 56
# Iters: 1000
# Filtering: True

datadir=$1
logfile=$2

# Clear files
cat /dev/null > $logfile
cat /dev/null > ${logfile}.log

for path in $datadir/*; do
    filename=$(basename "$path")
    time CUDA_VISIBLE_DEVICES=3 python -m autoprompt.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '<s> {sub_label} [T] [T] [T] [T] [T] [P] . </s>' \
        --num-cand 10 \
        --accumulation-steps 1 \
        --model-name roberta-large \
        --bsz 56 \
        --eval-size 56 \
        --iters 1000 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama >> $logfile 2>> ${logfile}.log
done
