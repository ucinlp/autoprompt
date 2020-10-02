#!/bin/bash
# Experiment 6
# Task: fact retrieval (original)
# Model: BERT
# Batch sizes: 28, 56, 112
# Iters: 1000
# Filtering: True

datadir=$1
logfile1=$2
logfile2=$3
logfile3=$4
CUDA_VISIBLE_DEVICES=0

# Clear files
cat /dev/null > $logfile1
cat /dev/null > ${logfile1}.log
cat /dev/null > $logfile2
cat /dev/null > ${logfile2}.log
cat /dev/null > $logfile3
cat /dev/null > ${logfile3}.log

for path in $datadir/*; do
    filename=$(basename "$path")
    CUDA_VISIBLE_DEVICES=0 python -m lmat.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '[CLS] {sub_label} [T] [T] [T] [T] [T] [P] . [SEP]' \
        --num_cand 10 \
        --accumulation-steps 1 \
        --model-name bert-base-cased \
        --bsz 28 \
        --eval-size 28 \
        --iters 1000 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama >> $logfile1 2>> ${logfile1}.log
done

for path in $datadir/*; do
    filename=$(basename "$path")
    CUDA_VISIBLE_DEVICES=0 python -m lmat.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '[CLS] {sub_label} [T] [T] [T] [T] [T] [P] . [SEP]' \
        --num_cand 10 \
        --accumulation-steps 1 \
        --model-name bert-base-cased \
        --bsz 56 \
        --eval-size 56 \
        --iters 1000 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama >> $logfile2 2>> ${logfile2}.log
done

for path in $datadir/*; do
    filename=$(basename "$path")
    CUDA_VISIBLE_DEVICES=0 python -m lmat.create_trigger \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --template '[CLS] {sub_label} [T] [T] [T] [T] [T] [P] . [SEP]' \
        --num_cand 10 \
        --accumulation-steps 1 \
        --model-name bert-base-cased \
        --bsz 112 \
        --eval-size 112 \
        --iters 1000 \
        --label-field 'obj_label' \
        --tokenize-labels \
        --filter \
        --print-lama >> $logfile3 2>> ${logfile3}.log
done
