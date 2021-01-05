#!/bin/sh
#SBATCH --job-name=fine_cyc
#SBATCH --output=./output.txt
#SBATCH --partition=ava_m.p
#SBATCH --nodelist=ava-m1
#SBATCH --gpus=1
#SBATCH --array=1-20
path=/home/yrazeghi/data/cycic3
RESULTS_DIR=/home/yrazeghi/PHD/autoprompt/results/cycfinetune/BERT/pre
modelpath=/home/yrazeghi/PHD/autoprompt/pretrained_models/can_conceptnet/synthetic-bert-base-uncased
BATCHSIZE=15
srun python -m autoprompt.finetune \
      --train $path/train-f.jsonl \
      --dev $path/dev_a-f.jsonl \
      --test $path/dev_b-f.jsonl \
      --model-name  $modelpath \
      --field-a 'question' \
      --seed $SLURM_ARRAY_TASK_ID \
      --label-map '{"False": 0, "True": 1}' \
      --bias-correction \
      --log-file $RESULTS_DIR/bert_p_${SLURM_ARRAY_TASK_ID}_finetune.log \
      -f \
      --bsz $BATCHSIZE \
      --ckpt-dir  $RESULTS_DIR/${SLURM_ARRAY_TASK_ID} \

#srun python -m autoprompt.finetune \
#      --train $path/train-f.jsonl \
#      --dev $path/dev_a-f.jsonl \
#      --test $path/dev_b-f.jsonl \
#      --model-name  'bert-base-uncased' \
#      --field-a 'question' \
#      --seed 1 \
#      --bias-correction \
#      --log-file /home/yrazeghi/PHD/autoprompt/results/cycfinetune/BERT/bert_b_1_finetune.log \
#      -f \
#      --bsz $BATCHSIZE \
#      --ckpt-dir /home/yrazeghi/PHD/autoprompt/savedmodels/bert_base_finetuned/ \


# the bert bsz was 10
#srun python -m autoprompt.finetune \
#      --train $path/train-f.jsonl \
#      --dev $path/dev_a-f.jsonl \
#      --test $path/dev_b-f.jsonl \
#      --model-name 'bert-large-cased' \
#      --field-a 'question' \
#      --seed $SLURM_ARRAY_TASK_ID \
#      --bias-correction \
#      --log-file $RESULTS_DIR/bert_l_${SLURM_ARRAY_TASK_ID}_finetune.log\
#      -f \
#      --bsz $BATCHSIZE

