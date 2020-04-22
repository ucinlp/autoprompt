#!/bin/bash

declare -a relations=("P276" "P279" "P30" "P31" "P36" "P361" "P364" "P37" "P39" "P407" "P413" "P449" "P463" "P47" "P495" "P527" "P530" "P740" "P937")

for i in "${relations[@]}"
do
    echo $i
    logfile="out/scrape_log/$i.txt"
    python get_more_TREx_data.py $i "data/TREx_train_more/$i.jsonl" --trex_file "../LAMA/data/LMAT/TREx_test/$i.jsonl" --common_vocab_file "misc/common_vocab_cased.txt" --query_limit 10000 --sleep_time 0.001 > $logfile
done
