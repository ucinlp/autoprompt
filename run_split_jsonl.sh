#!/bin/bash

datadir=$1

for path in $datadir/*; do
    relname=$(basename "$path")
    filepath="$path/$relname.jsonl"
    python split_jsonl.py $filepath $path --train-ratio 0.8 --val-ratio 0.2
    echo "Split $relname"
done
