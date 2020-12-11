"""
Preprocessors for dealing with different input files.
"""
import csv
import json



def _stringify(d):
    return {k: str(v) for k, v in d.items()}


def preprocess_csv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            yield row


def preprocess_tsv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def preprocess_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield _stringify(json.loads(line))


# REMINDER: You need to add whatever preprocessing functions you've written to
# this dict to make them available to the training scripts.
PREPROCESSORS = {
    '.csv': preprocess_csv,
    '.tsv': preprocess_tsv,
    '.jsonl': preprocess_jsonl,
}
