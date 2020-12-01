#!/bin/sh

datadir=$1
logfile=$2
CUDA_VISIBLE_DEVICES=0

# Clear files
cat /dev/null > $logfile

for path in $datadir/*;
do
    filename=$(basename "$path");
    python -m autoprompt.continuous_trigger_fact_retrieval \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --test $path/test.jsonl \
        --template '{sub_label} [T] [T] [T] [T] [T] [P] .' \
        --label-field 'obj_label' \
        --model-name bert-base-cased \
        --lr 3e-2
    break
done
