#!/bin/bash
# Experiment 9
# Task: relation extraction
# Model: BERT
# Batch size: 32
# Iters: 500
# Filtering: True

datadir=$1
logfile=$2

# Clear files
cat /dev/null > $logfile
cat /dev/null > ${logfile}.log

for path in $datadir/*; do
    filename=$(basename "$path")
    time CUDA_VISIBLE_DEVICES=4 python -m autoprompt.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '[CLS] {context} [SEP] {sub_label} [T] [T] [T] [T] [T] [P] . [SEP]' \
        --num-cand 10 \
        --accumulation-steps 1 \
        --model-name bert-base-cased \
        --bsz 32 \
        --eval-size 32 \
        --iters 500 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama \
        --use-ctx >> $logfile 2>> ${logfile}.log
done
