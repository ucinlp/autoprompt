"""
Preprocessors for dealing with different input files.
"""
import csv
import json


PREPROCESSORS = {
    'csv': process_csv,
    'tsv': process_tsv,
    'jsonl': process_jsonl,
}

def _stringify(d):
    return {k: str(v) for k, v in d.items()}


def process_csv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            yield row


def process_tsv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def process_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield _stringify(json.loads(line))
