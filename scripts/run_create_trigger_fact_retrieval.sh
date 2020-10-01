#!/bin/bash
# Experiment 2: non-token labels + filtering

datadir=$1
logfile=$2
CUDA_VISIBLE_DEVICES=0

# Clear files
cat /dev/null > $logfile
cat /dev/null > ${logfile}.log

for path in $datadir/*; do
    filename=$(basename "$path")
    python -m lmat.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '<s> {sub_label} [T] [T] [T] [P] . </s>' \
        --num_cand 10 \
        --accumulation-steps 1 \
        --model-name roberta-base \
        --bsz 12 \
        --eval-size 12 \
        --iters 10 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama >> $logfile 2>> ${logfile}.log
done
