python -m autoprompt.create_trigger \
  --train data/FewGLUE/CB/train.jsonl \
  --dev data/SuperGLUE/CB/val.jsonl \
  --template '<s> {hypothesis} [T] [T] [T] [P] . {premise} . </s>' \
  --label-map '{"entailment": ["Ġyes"], "contradiction": ["Ġno"], "neutral": ["Ġmaybe"]}' \
  --model-name 'roberta-large' \
  --bsz 4 \
  --eval-size 4 \
  --accumulation-steps 2 \
