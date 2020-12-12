RESULTS_DIR=${1:-results}
mkdir -p $RESULTS_DIR

for SEED in {1..10}
do
  # BoolQ
  python -m autoprompt.label_search \
    --train data/FewGLUE/BoolQ/train.jsonl \
    --template '[CLS] {passage} [T] [T] [T] {question} [T] [T] [T] [P] . [SEP]' \
    --label-field 'label' \
    --label-map '{"True": 1, "False": 0}' \
    --model-name 'bert-base-cased' \
    --seed $SEED \
    > $RESULTS_DIR/BoolQ_${SEED}.label_map \
    2> $RESULTS_DIR/BoolQ_${SEED}.log

  # CB
  python -m autoprompt.label_search \
    --train data/FewGLUE/CB/train.jsonl \
    --template '[CLS] {hypothesis} [T] [T] [T] [P] . {premise} . [SEP]' \
    --label-field 'label' \
    --label-map '{"entailment": 0, "contradiction": 1, "neutral": 2}' \
    --model-name 'bert-base-cased' \
    --seed $SEED \
    > $RESULTS_DIR/CB_${SEED}.label_map \
    2> $RESULTS_DIR/CB_${SEED}.log

  # COPA
  python -m autoprompt.label_search \
    --train data/FewGLUE/COPA/train.jsonl \
    --template '[CLS] {choice1} [T] [T] [T] {choice2} [T] [T] [P] [T] [T] {premise} . [SEP]' \
    --label-field 'label' \
    --label-map '{"0": 0, "1": 1}' \
    --model-name 'bert-base-cased' \
    --seed $SEED \
    > $RESULTS_DIR/COPA_${SEED}.label_map \
    2> $RESULTS_DIR/COPA_${SEED}.log

  # RTE
  python -m autoprompt.label_search \
    --train data/FewGLUE/RTE/train.jsonl \
    --template '[CLS] {hypothesis} [T] [T] [T] [P] . {premise} . [SEP]' \
    --label-field 'label' \
    --label-map '{"entailment": 0, "not_entailment": 1}' \
    --model-name 'bert-base-cased' \
    --seed $SEED \
    > $RESULTS_DIR/RTE_${SEED}.label_map \
    2> $RESULTS_DIR/RTE_${SEED}.log

  # WiC
  python -m autoprompt.label_search \
    --train data/FewGLUE/WiC/train.jsonl \
    --template '[CLS] [T] {sentence1} [T] [T] {sentence2} [T] [T] {word} [T] [T] [P] . [SEP]' \
    --label-field 'label' \
    --label-map '{"False": 0, "True": 1}' \
    --model-name 'bert-base-cased' \
    --seed $SEED \
    > $RESULTS_DIR/WiC_${SEED}.label_map \
    2> $RESULTS_DIR/WiC_${SEED}.log

done
