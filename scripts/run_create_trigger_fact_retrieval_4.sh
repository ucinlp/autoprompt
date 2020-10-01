#!/bin/bash
# Experiment 5: fact retrieval (non-token labels + filtering) using RoBERTa

datadir=$1
logfile=$2

SECONDS=0
i=1
for path in $datadir/*; do
    filename=$(basename "$path")
    echo "Creating trigger for $filename"
    $(CUDA_VISIBLE_DEVICES=6 python -m lmat.create_trigger --train $path/train.jsonl --dev $path/dev.jsonl --template '<s> {sub_label} [T] [T] [T] [T] [T] [P] . </s>' --num_cand 10 --accumulation-steps 1 --model-name roberta-large --bsz 28 --eval-size 56 --iters 500 --label-field 'obj_label' --tokenize-labels --filter >> $logfile)
    echo "Saving results to $logfile"
    ((i++))
done
duration=$SECONDS
echo "Time Elapsed: $(($duration / 60)) min"
