#!/bin/bash

prompt_format_filename="misc/prompt_formats.txt"
manual_prompts_filename="misc/manual_prompts.txt"

# i=1
# for path in ~/workspace/data/LMAT/TREx_train/*; do
#     filename=$(basename "$path")
#     python split_jsonl.py "$path/$filename.jsonl" $path --train-ratio 0.8 --val-ratio 0.2
# done

i=1
for path in ~/workspace/data/LMAT/TREx/*; do
    # fullfilename=$(basename "$path")
    # filename=${fullfilename%.*}
    filename=$(basename "$path")
    logfile="out/uncond/man_cand50_bm3/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    python create_trigger.py $path out --lm bert --iters 50 --bsz 64 --patience 10 --num_cand 50 --beam_size 3 --manual "$manual_prompt" --format "$prompt_format" > $logfile
    echo "Saving results to $logfile"
    ((i++))
done
echo "--------------------------------------------------------------"

i=1
for path in ~/workspace/data/LMAT/TREx/*; do
    # fullfilename=$(basename "$path")
    # filename=${fullfilename%.*}
    filename=$(basename "$path")
    logfile="out/uncond/man_cand100_bm3/$filename.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    manual_prompt=$(sed -n ${i}p $manual_prompts_filename)
    python create_trigger.py $path out --lm bert --iters 50 --bsz 64 --patience 10 --num_cand 100 --beam_size 3 --manual "$manual_prompt" --format "$prompt_format" > $logfile
    echo "Saving results to $logfile"
    ((i++))
done
echo "--------------------------------------------------------------"
