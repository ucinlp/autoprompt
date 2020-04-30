#!/bin/bash
# counter=200
sentence_size=100
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "P-S-2-H-Y" --ent_word "thus" --cont_word "even" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=201
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "Y-P-S-2-Y-H" --ent_word "thus" --cont_word "even" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=202
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "P-S-2-H-Y" --ent_word "thus" --cont_word "even" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=203
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/ ./out/$counter  --iters 50 --bsz 32 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the" --format "Y-P-S-3-Y-H" --ent_word "thus" --cont_word "even" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=204
# sentence_size=1000
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 10 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "P-S-2-Y-H" --ent_word "so" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=205
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 10 --patience 20 --num_cand 50 --beam_size 2 --manual "the sentence" --format "P-S-Y-2-H" --ent_word "so" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=206
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 10 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the" --format "P-S-Y-3-H" --ent_word "so" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
# counter=207
# mkdir ./out/$counter
# python create_trigger.py  /home/yrazeghi/data/ ./out/$counter  --iters 50 --bsz 10 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the" --format "P-S-3-Y-H" --ent_word "so" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
counter=208
mkdir ./out/$counter
python create_trigger.py  /home/yrazeghi/data/  ./out/$counter  --iters 50 --bsz 10 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the the" --format "P-S-4-Y-H" --ent_word "so" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
counter=209
mkdir ./out/$counter
python create_trigger.py  /home/yrazeghi/data/ ./out/$counter  --iters 50 --bsz 10 --patience 20 --num_cand 50 --beam_size 2 --manual "the the the the" --format "P-S-Y-4-H" --ent_word "so" --cont_word "but" --sentence_size $sentence_size --dataset "MNLI" > ./out/$counter.txt
