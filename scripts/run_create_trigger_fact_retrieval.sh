#!/bin/bash
# Experiment 2: non-token labels + filtering

datadir=$1
logfile=$2

SECONDS=0
i=1
for path in $datadir/*; do
    filename=$(basename "$path")
    # logfile="$logdir/$filename.txt"
    # if [ "$filename" = "P138" ] || [ "$filename" = "P159" ] || [ "$filename" = "P17" ] || [ "$filename" = "P19" ] || [ "$filename" = "P190" ] || [ "$filename" = "P127" ] || [ "$filename" = "P361" ] || [ "$filename" = "P364" ] || [ "$filename" = "P37" ] || [ "$filename" = "P495" ] || [ "$filename" = "P527" ] || [ "$filename" = "P530" ] || [ "$filename" = "P740" ] || [ "$filename" = "P937" ]; then
    echo "Creating trigger for $filename"
    $(CUDA_VISIBLE_DEVICES=0 python -m lmat.create_trigger --train $path/train.jsonl --dev $path/dev.jsonl --template '[CLS] {sub_label} [T] [T] [T] [T] [T] [P] . [SEP]' --num_cand 10 --accumulation-steps 1 --model-name bert-base-cased --bsz 28 --eval-size 56 --iters 500 --label-field 'obj_label' --tokenize-labels --filter >> $logfile)
    echo "Saving results to $logfile"
    # fi
    ((i++))
done
duration=$SECONDS
echo "Time Elapsed: $(($duration / 60)) min"
