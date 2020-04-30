#!/bin/bash
# counter=0
# sentence_size=100
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "P-S-2-H-Y" --ent_word "Yes" --cont_word "No" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=1
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "Y-P-S-2-Y-H" --ent_word "Yes" --cont_word "No" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=2
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "P-S-2-H-Y" --ent_word "Yes" --cont_word "No" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=3
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/ ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the" --format "Y-P-S-3-Y-H" --ent_word "Yes" --cont_word "No" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=4
# sentence_size=100
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence the" --format "P-S-3-Y-H-" --ent_word "and" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=5
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence the" --format "P-S-Y-3-H" --ent_word "and" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=6
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence the the" --format "P-S-4-H-Y" --ent_word "and" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=7
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/ ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the the" --format "P-S-4-Y-H" --ent_word "and" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
counter=8
mkdir ./out/$counter
python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence the the the" --format "P-S-5-H-Y" --ent_word "and" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
counter=9
mkdir ./out/$counter
python create_trigger.py  /home/yrazeghi/data/ ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the the the" --format "P-S-5-Y-H" --ent_word "and" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
