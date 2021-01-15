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


def preprocess_multirc(fname):
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            passage_text = data['passage']['text']
            for question in data['passage']['questions']:
                question_text = question['question']
                for answer in question['answers']:
                    answer_text = answer['text']
                    label = answer['label']
                    yield {
                        'passage': passage_text,
                        'question': question_text,
                        'answer': answer_text,
                        'label': str(label),
                    }


def preprocess_wsc(fname):
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Highligh pronoun in sentence
            words = data['text'].split()
            pronoun_idx = data['target']['span2_index']
            words[pronoun_idx] = '*' + words[pronoun_idx] + '*'
            highlighted_sentence = ' '.join(words)
            pronoun = data['target']['span2_text']
            label = data['target']['span1_text']
            yield {
                'sentence': highlighted_sentence,
                'pronoun': pronoun,
                'label': label,
            }


# REMINDER: You need to add whatever preprocessing functions you've written to
# this dict to make them available to the training scripts.
PREPROCESSORS = {
    '.csv': preprocess_csv,
    '.tsv': preprocess_tsv,
    '.jsonl': preprocess_jsonl,
    'multirc': preprocess_multirc,
    'wsc': preprocess_wsc,
}
