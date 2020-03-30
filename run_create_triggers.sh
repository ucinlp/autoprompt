#!/usr/bin/env bash

prompt_format_filename="prompt_formats.txt"
man_prompts_filename="manual_prompts.txt"

i=1
for d in ~/workspace/data/LMAT/TREx/*; do
    b=$(basename "${d}")
    outfile="out/closed_book/man_cand10/$b.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    man_prompt=$(sed -n ${i}p $man_prompts_filename)
    python create_trigger.py $d out --lm bert --iters 50 --bsz 32 --patience 10 --num_cand 10 --beam_size 1 --format "$prompt_format" --manual "$man_prompt" > $outfile
    echo "Saving results to $outfile"
    ((i++))
done
echo "--------------------------------------------------------------"

i=1
for d in ~/workspace/data/LMAT/TREx/*; do
    b=$(basename "${d}")
    outfile="out/closed_book/man_cand50/$b.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    man_prompt=$(sed -n ${i}p $man_prompts_filename)
    python create_trigger.py $d out --lm bert --iters 50 --bsz 32 --patience 10 --num_cand 50 --beam_size 1 --format "$prompt_format" --manual "$man_prompt" > $outfile
    echo "Saving results to $outfile"
    ((i++))
done
echo "--------------------------------------------------------------"

i=1
for d in ~/workspace/data/LMAT/TREx/*; do
    b=$(basename "${d}")
    outfile="out/closed_book/man_cand100/$b.txt"
    prompt_format=$(sed -n ${i}p $prompt_format_filename)
    man_prompt=$(sed -n ${i}p $man_prompts_filename)
    python create_trigger.py $d out --lm bert --iters 50 --bsz 32 --patience 10 --num_cand 100 --beam_size 1 --format "$prompt_format" --manual "$man_prompt" > $outfile
    echo "Saving results to $outfile"
    ((i++))
done
echo "--------------------------------------------------------------"
