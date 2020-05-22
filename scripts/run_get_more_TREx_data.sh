#!/bin/bash

declare -a relations=("P264" "P276" "P279" "P36" "P361" "P449" "P463" "P47" "P527" "P937")

for i in "${relations[@]}"
do
    echo $i
    python get_more_TREx_data.py "data/TREx/train_extra/$i/$i.tsv" "data/TREx/train_extra/$i/$i.jsonl" --trex_file "../LAMA/data/LMAT/TREx_test/$i.jsonl" --common_vocab_file "misc/common_vocab_cased.txt" --sleep_time 0.0001 --count 1000
done
