#!/bin/bash

datadir=$1

for path in $datadir/*; do
    relname=$(basename "$path")
    filepath="$path/$relname.jsonl"
    python split_jsonl.py $filepath $path --train_ratio 0.6 --val_ratio 0.2 --include_test
    echo "Split $relname"
done
