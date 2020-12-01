RESULTS_DIR=${1:-results}
mkdir -p $RESULTS_DIR


# NOTE: `label-map` comes from PET.


for SEED in {1..10}
do

  # BoolQ
  python -m autoprompt.create_trigger \
    --train data/FewGLUE/BoolQ/train.jsonl \
    --dev data/FewGLUE/BoolQ/train.jsonl \
    --template '<s> {passage} [T] [T] [T] {question} [T] [T] [T] [P] . </s>' \
    --label-map '{"True": ["Ġyes"], "False": ["Ġno"]}' \
    --model-name 'roberta-large' \
    --seed $SEED \
    --bsz 4 \
    --eval-size 4 \
    --accumulation-steps 2 \
    2> $RESULTS_DIR/BoolQ_${SEED}_autoprompt.log

  # CB
  python -m autoprompt.create_trigger \
    --train data/FewGLUE/CB/train.jsonl \
    --dev data/FewGLUE/CB/train.jsonl \
    --template '<s> {hypothesis} [T] [T] [T] [P] . {premise} . </s>' \
    --label-map '{"entailment": ["Ġyes"], "contradiction": ["Ġno"], "neutral": ["Ġmaybe"]}' \
    --model-name 'roberta-large' \
    --seed $SEED \
    --bsz 4 \
    --eval-size 4 \
    --accumulation-steps 2 \
    2> $RESULTS_DIR/CB_${SEED}_autoprompt.log

  # COPA
  python -m autoprompt.create_trigger \
    --train data/FewGLUE/COPA/train.jsonl \
    --dev data/FewGLUE/COPA/train.jsonl \
    --template '<s> {choice1} [T] [T] [T] {choice2} [T] [T] [P] [T] [T] {premise} . </s>' \
    --label-map '{"0": ["Ġfirst"], "1": ["Ġsecond"]}' \
    --model-name 'roberta-large' \
    --seed $SEED \
    --bsz 4 \
    --eval-size 4 \
    --accumulation-steps 2 \
    2> $RESULTS_DIR/COPA_${SEED}_autoprompt.log

  # RTE
  python -m autoprompt.create_trigger \
    --train data/FewGLUE/RTE/train.jsonl \
    --dev data/FewGLUE/RTE/train.jsonl \
    --template '<s> {hypothesis} [T] [T] [T] [P] . {premise} . </s>' \
    --label-map '{"entailment": ["Ġyes"], "not_entailment": ["Ġno"]}' \
    --model-name 'roberta-large' \
    --seed $SEED \
    --bsz 4 \
    --eval-size 4 \
    --accumulation-steps 2 \
    2> $RESULTS_DIR/RTE_${SEED}_autoprompt.log

  # WiC
  python -m autoprompt.create_trigger \
    --train data/FewGLUE/WiC/train.jsonl \
    --dev data/FewGLUE/WiC/train.jsonl \
    --template '<s> [T] {sentence1} [T] [T] {sentence2} [T] [T] {word} [T] [T] [P] . </s>' \
    --label-map '{"True": ["Ġyes"], "False": ["Ġno"]}' \
    --model-name 'roberta-large' \
    --seed $SEED \
    --bsz 4 \
    --eval-size 4 \
    --accumulation-steps 2 \
    2> $RESULTS_DIR/WiC_${SEED}_autoprompt.log
done
