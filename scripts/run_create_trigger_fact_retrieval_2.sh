#!/bin/bash
# Experiment 3: non-token labels + no filtering + 1000 iterations

datadir=$1
logfile=$2

SECONDS=0
i=1
for path in $datadir/*; do
    filename=$(basename "$path")
    # logfile="$logdir/$filename.txt"
    # if [ "$filename" != "P1001" ] && [ "$filename" != "P101" ] && [ "$filename" != "P103" ] && [ "$filename" != "P106" ] && [ "$filename" != "P108" ] && [ "$filename" != "P127" ] && [ "$filename" != "P1303" ] && [ "$filename" != "P131" ] && [ "$filename" != "P131" ] && [ "$filename" != "P136" ] && [ "$filename" != "P1376" ] && [ "$filename" != "P138" ] && [ "$filename" != "P140" ] && [ "$filename" != "P1412" ] && [ "$filename" != "P159" ] && [ "$filename" != "P17" ] && [ "$filename" != "P176" ] && [ "$filename" != "P178" ] && [ "$filename" != "P19" ] && [ "$filename" != "P190" ]; then
    echo "Creating trigger for $filename"
    $(CUDA_VISIBLE_DEVICES=4 python -m lmat.create_trigger --train $path/train.jsonl --dev $path/dev.jsonl --template '[CLS] {sub_label} [T] [T] [T] [T] [T] [P] . [SEP]' --num_cand 10 --accumulation-steps 1 --model-name bert-base-cased --bsz 28 --eval-size 56 --iters 1000 --label-field 'obj_label' --tokenize-labels >> $logfile)
    echo "Saving results to $logfile"
    # fi
    ((i++))
done
duration=$SECONDS
echo "Time Elapsed: $(($duration / 60)) min"
