#!/bin/sh

datadir=$1
logfile=$2
CUDA_VISIBLE_DEVICES=0

# Clear files
cat /dev/null > $logfile

for path in $datadir/*; do
    filename=$(basename "$path")
    export PYTHONPATH="../":"${PYTHONPATH}"    
    python autoprompt/continuous_trigger_fact_retrieval.py \
        --train $path/train.jsonl \
        --dev $path/dev.jsonl \
        --test $path/test.jsonl \
        --model-name bert-base-cased \
        --field-a 'sub_label' \
        --label-field 'obj_label'

done