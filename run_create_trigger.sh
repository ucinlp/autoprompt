#!/bin/bash

prompt_format_filename="misc/prompt_formats.txt"
manual_prompts_filename="misc/manual_prompts.txt"

datadir1=$1
datadir2=$2
logdir1=$3
logdir2=$4

mkdir -p $logdir1
mkdir -p $logdir2

# CONDITIONAL PROBING (bsz=8)
i=1
for path in $datadir1/*; do
    filename=$(basename "$path")
    logfile="$logdir1/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    echo "Creating trigger for $filename"
    python create_trigger.py $path out --lm bert --iters 50 --bsz 8 --patience 10 --num_cand 10 --beam_size 1 --format "X-5-Y" --use_ctx > $logfile
    echo "Saving results to $logfile"
    ((i++))
done
echo "--------------------------------------------------------------"

i=1
for path in $datadir2/*; do
    filename=$(basename "$path")
    logfile="$logdir2/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    echo "Creating trigger for $filename"
    python create_trigger.py $path out --lm bert --iters 50 --bsz 8 --patience 10 --num_cand 50 --beam_size 1 --format "X-5-Y" --use_ctx > $logfile
    echo "Saving results to $logfile"
    ((i++))
done
echo "--------------------------------------------------------------"

# UNCONDITIONAL PROBING (bsz=64)
# i=1
# for path in $datadir2/*; do
#     # fullfilename=$(basename "$path")
#     # filename=${fullfilename%.*}
#     filename=$(basename "$path")
#     logfile="$logdir2/$filename.txt"
#     prompt_format=$(sed -n ${i}p $prompt_format_filename)
#     manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
#     echo "Creating trigger for $filename"
#     python create_trigger.py $path out --lm bert --iters 50 --bsz 64 --patience 10 --num_cand 10 --beam_size 1 --manual "$manual_prompt" --format "$prompt_format" > $logfile
#     echo "Saving results to $logfile"
#     ((i++))
# done
# echo "--------------------------------------------------------------"

# i=1
# for path in $datadir2/*; do
#     # fullfilename=$(basename "$path")
#     # filename=${fullfilename%.*}
#     filename=$(basename "$path")
#     logfile="$logdir2/$filename.txt"
#     prompt_format=$(sed -n ${i}p $prompt_format_filename)
#     manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
#     echo "Creating trigger for $filename"
#     python create_trigger.py $path out --lm bert --iters 50 --bsz 64 --patience 10 --num_cand 10 --beam_size 1 --format "2-X-2-Y-2" > $logfile
#     echo "Saving results to $logfile"
#     ((i++))
# done
# echo "--------------------------------------------------------------"
