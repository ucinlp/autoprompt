## Setup

### 1. Create conda environment and install requirements
```
conda create -n lmat37 -y python=3.7 && conda activate lmat37
pip install -r requirements.txt
```

### 2. Download stuff
Install spacy model
```
python -m spacy download en
```

Get full TREx data (train and test) from https://drive.google.com/drive/folders/1Df4D4qH_34B4_tvDDbMX1Q1IvS1hVG-8?usp=sharing

## Generation

### 1. Create a trigger for a particular relation
```
python create_trigger.py $DATA_PATH out --lm bert --iters 50 --bsz 64 --patience 10 --num_cand 50 --beam_size 3 --manual misc/manual_prompts.txt --format misc/prompt_formats.txt
```

## Evaluation

### 1. Setup LAMA
Go to our fork of the LAMA repo https://github.com/taylorshin/LAMA and follow the directions to set up the LAMA repo outside of this LMAT repo.
It is recommended to create a separate conda environment for LAMA due to different dependencies and requirements.

Copy the LMAT data folder into the `data` directory of LAMA or set `data_path_pre` in `scripts/run_experiments.py` to a custom data location.

### 2. Evaluate prompts
First, update the `data/relations.jsonl` file with your own prompts whether they are manual prompts or automatically generated prompts.
Then, run the evaluation code.
```
python scripts/run_experiments.py
```

### 3. Settings
To change evaluation settings, go to `scripts/run_experiments.py` and update the PARAMETERS accordingly.
For example, set `use_context` to True for conditional prompt evaluation.

### 4. Miscellaneous
Set PYTHONPATH if the following error occurs: `ModuleNotFoundError: No module named 'lama', pip`
```
export PYTHONPATH="${PYTHONPATH}:/path/to/your/module/"
```
