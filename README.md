# AutoPrompt
An automated method based on gradient-guided search to create prompts for a diverse set of NLP tasks. AutoPrompt demonstrates that masked language models (MLMs) have an innate ability to perform sentiment analysis, natural language inference, fact retrieval, and relation extraction. Check out our [website](https://ucinlp.github.io/autoprompt/) for the paper and more information.

## Table of Contents
* [Setup](#setup)
* [Generating Prompts](#generating-prompts)
* [Label Token Selection](#label-token-selection)
* [Evaluation for Fact Retrieval and Relation Extraction](#evaluation-for-fact-retrieval-and-relation-extraction)
* [Citation](#citation)

## Setup

### 1. Create conda environment
```
conda create -n autoprompt -y python=3.7 && conda activate autoprompt
```

### 2. Install dependecies
Install the required packages
```
pip install -r requirements.txt
```
Also download the spacy model
```
python -m spacy download en
```

### 3. Download the data
The datasets for sentiment analysis, NLI, fact retrieval, and relation extraction are available to download [here](https://drive.google.com/drive/folders/1vVhgnSXmbuJb6GLPn_FErY1xDTh1xyv-?usp=sharing)

There are a couple different datasets for fact retrieval and relation extraction so here are brief overviews of each:
- Fact Retrieval
  - `original`: We used the T-REx subset provided by LAMA as our test set and gathered more facts from the [original T-REx dataset](https://hadyelsahar.github.io/t-rex/) that we partitioned into train and dev sets
  - `original_rob`: We filtered facts in `original` so that each object is a single token for both BERT and RoBERTa
  - `trex`: We split the extra T-REx data collected (for train/val sets of `original`) into train, dev, test sets
- Relation Extraction
  - Trimmed the `original` dataset to compensate for both the [RE baseline](https://github.com/UKPLab/emnlp2017-relation-extraction) and RoBERTa. We also excluded relations `P527` and `P1376` because the RE baseline doesn’t consider them.

## Generating Prompts

### Quick Overview of Templates
A prompt is constructed by mapping things like the original input and trigger tokens to a template that looks something like

`[CLS] {sub_label} [T] [T] [T] [P]. [SEP]`

The example above is a template for generating fact retrieval prompts with 3 trigger tokens where `{sub_label}` is a placeholder for the subject in any (subject, relation, object) triplet in fact retrieval. `[P]` denotes the placement of a special `[MASK]` token that will be used to "fill-in-the-blank" by the language model. Each trigger token in the set of trigger tokens that are shared across all prompts is denoted by `[T]`.

Depending on the language model (i.e. BERT or RoBERTa) you choose to generate prompts, the special tokens will be different. For BERT, stick `[CLS]` and `[SEP]` to each end of the template. For RoBERTa, use `<s>` and `</s>` instead.

### Sentiment Analysis
```
python -m autoprompt.create_trigger \
    --train glue_data/SST-2/train.tsv \
    --dev glue_data/SST-2/dev.tsv \
    --template '<s> {sentence} [T] [T] [T] [P] . </s>' \
    --label-map '{"0": ["Ġworse", "Ġincompetence", "ĠWorse", "Ġblamed", "Ġsucked"], "1": ["ĠCris", "Ġmarvelous", "Ġphilanthrop", "Ġvisionary", "Ġwonderful"]}' \
    --num-cand 100 \
    --accumulation-steps 30 \
    --bsz 24 \
    --eval-size 48 \
    --iters 180 \
    --model-name roberta-large
```

### Natural Language Inference
```
python  -m autoprompt.create_trigger  --train SICK_TRAIN_ALL_S.tsv --dev SICK_DEV_ALL_S.tsv --template '<s> {sentence_A} [P] [T] [T] [T] [T] {sentence_B} </s>'  --label-map '{"ENTAILMENT": ["\u0120Taiwan", "\u0120Ara", "abet"], "CONTRADICTION": ["\u0120Only", "\u0120Didn", "\u0120BUT"], "NEUTRAL": ["icy", "oder", "agna"]}' --bsz 120  --model-name roberta-large
```

### Fact Retrieval
```
python -m autoprompt.create_trigger \
    --train $path/train.jsonl \
    --dev $path/dev.jsonl \
    --template '<s> {sub_label} [T] [T] [T] [P] . </s>' \
    --num-cand 10 \
    --accumulation-steps 1 \
    --model-name roberta-large \
    --bsz 56 \
    --eval-size 56 \
    --iters 1000 \
    --label-field 'obj_label' \
    --tokenize-labels \
    --filter \
    --print-lama
```

### Relation Extraction
```
python -m autoprompt.create_trigger \
    --train $path/train.jsonl \
    --dev $path/dev.jsonl \
    --template '[CLS] {context} [SEP] {sub_label} [T] [T] [T] [P] . [SEP]' \
    --num-cand 10 \
    --accumulation-steps 1 \
    --model-name bert-base-cased \
    --bsz 32 \
    --eval-size 32 \
    --iters 500 \
    --label-field 'obj_label' \
    --tokenize-labels \
    --filter \
    --print-lama \
    --use-ctx
```

## Label Token Selection

For sentiment analysis
```
python -m autoprompt.label_search --train ../data/SST-2/train.tsv --template '[CLS] {sentence} [T] [T] [T] [P]. [SEP]' --label-map '{"0": 0, "1": 1}' --iters 50 --model-name 'bert-base-cased'
```

For NLI
```
python -m autoprompt.label_search --train ../data/SICK-E-balanced/3-balance/SICK_TRAIN_ALL_S.tsv --template '[CLS] {sentence} [T] [T] [T] [P]. [SEP]' --label-map '{"entailment": 0, "contradiction": 1, "neutral": 2}' --iters 50 --model-name 'bert-base-cased'
```

## Evaluation for Fact Retrieval and Relation Extraction

### 1. Setup LAMA
Clone [our fork](https://github.com/taylorshin/LAMA) of the LAMA repo and follow the directions to set it up outside of the AutoPrompt repo.
We recommended creating a separate conda environment for LAMA due to different dependencies and requirements.

Copy the AutoPrompt data folder into the `data` directory of LAMA or set `data_path_pre` in `scripts/run_experiments.py` to a custom data location.

In order to get LAMA to work with RoBERTa, run the following commands:
```
mkdir pre-trained_language_models/roberta
cd pre-trained_language_models/roberta
curl -O https://dl.fbaipublicfiles.com/fairseq/models/roberta.large.tar.gz
tar -xvzf roberta.large.tar.gz
```

### 2. Update prompts
Update the `data/relations.jsonl` file with your own automatically generated prompts

### 3. Configure settings
To change evaluation settings, go to `scripts/run_experiments.py` and update the configurable values accordingly.
Note: each of the configurable settings are marked with a `[CONFIGURABLE]` comment.

- Uncomment the settings of the LM you want to evaluate with (and comment out the other LM settings) in the `LMs` list at the top of the file
- Update the `common_vocab_filename` field to the appropriate file. Anything evaluating both BERT and RoBERTa requires this field to be `common_vocab_cased_rob.txt` instead of the usual `common_vocab_cased.txt`.
- Set `use_ctx` to `True` if running evaluation for Relation Extraction
- Set `synthetic` to `True` for perturbed sentence evaluation for Relation Extraction
- In `get_TREx_parameters` function, set `data_path_pre` to the corresponding data path (e.g. `"../data/relation_extraction"` for Relation Extraction)

### 4. Evaluate prompts
Run the evaluation code
```
python scripts/run_experiments.py
```

### 4. Miscellaneous
Set `PYTHONPATH` if the following error occurs: `ModuleNotFoundError: No module named 'lama'`
```
export PYTHONPATH="${PYTHONPATH}:/path/to/the/AutoPrompt/repo"
```

## Citation
```
@inproceedings{autoprompt:emnlp20,
  author = {Taylor Shin and Yasaman Razeghi and Robert L. Logan IV and Eric Wallace and Sameer Singh},
  title = { {AutoPrompt}: Eliciting Knowledge from Language Models with Automatically Generated Prompts },
  booktitle = {Empirical Methods in Natural Language Processing (EMNLP)},
  year = {2020}
}
```
