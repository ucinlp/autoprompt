#!/bin/bash

datadir=$1
logfile=$2
# logdir=$2

# mkdir -p $logdir

i=1
for path in $datadir/*; do
    filename=$(basename "$path")
    # logfile="$logdir/$filename.txt"
    echo "Creating trigger for $filename"
    $(CUDA_VISIBLE_DEVICES=1 python -m lmat.create_trigger --train $path/train.jsonl --dev $path/dev.jsonl --template '{sub_label} [T] [T] [T] [T] [T] [P] .' --num_cand 20 --accumulation-steps 20 --model-name bert-base-cased --bsz 128 --eval-size 128 --label-field 'obj_label' --initial-trigger the the the the the --iters 50 >> $logfile)
    echo "Saving results to $logfile"
    ((i++))
done
