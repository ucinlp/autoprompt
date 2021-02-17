"""
Preprocessors for dealing with different input files.
"""
import csv
import json
import io


def _stringify(d):
    return {k: str(v) for k, v in d.items()}


def preprocess_csv(fname, **kwargs):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter=',')
        for row in reader:
            yield row


def preprocess_tsv(fname, **kwargs):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def preprocess_jsonl(fname, **kwargs):
    with open(fname, 'r') as f:
        for line in f:
            yield _stringify(json.loads(line))

def preprocess_jsonl_string(fname, **kwargs):
    with io.StringIO(fname) as f:
        for line in f:
            yield _stringify(json.loads(line))

def preprocess_multirc(fname, **kwargs):
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


def preprocess_wsc(fname, **kwargs):
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            # Highlight pronoun in sentence
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


def preprocess_record(fname, **kwargs):
    """
    Heavily copied from:
        https://github.com/timoschick/pet/blob/master/pet/tasks.py
    """
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            text = data['passage']['text']
            seen_entities = set()
            entities = []

            for entity in data['passage']['entities']:
                start = entity['start']
                end = entity['end']
                mention = text[start:end+1]
                if mention not in seen_entities:
                    entities.append(mention)
                    seen_entities.add(mention)

            text = text.replace('@highlight\n', '- ')
            questions = data['qas']
            for question in questions:
                question_text = question['query']
                answers = set()
                for answer in question.get('answers', []):
                    answer_text = answer['text']
                    answers.add(answer_text)

            labels = []
            for entity in entities:
                labels.append((entity, entity in answers))

            yield {
                'passage': text,
                'question': question_text,
                'labels': labels
            }


def preprocess_copa(fname, train, **kwargs):
    # Unlike other preprocessors this one behaves differently for training and evaluation data.
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)

            # Map question to PET conjunction.
            conjunction = 'so' if data['question'] == 'effect' else 'because'

            # Sort labels so that true is always first
            labels = [
                (data['choice1'].lower()[:-1], bool(data['label']==0)),
                (data['choice2'].lower()[:-1], bool(data['label']==1))
            ]
            labels.sort(key=lambda x: x[1], reverse=True)


            # Output original and flipped arguments.
            original = {
                'premise': data['premise'][:-1].lower(),
                'choice1': data['choice1'][:-1].lower(),
                'choice2': data['choice2'][:-1].lower(),
                'conjunction': conjunction,
                'labels': labels,
            }
            yield original

            if train:
                flipped = original.copy()
                flipped['choice1'] = original['choice2']
                flipped['choice2'] = original['choice1']
                yield flipped


def preprocess_kilt(fname):
    with open(fname, 'r') as f:
        for line in f:
            data = json.loads(line)
            input_text = data['input']
            answer_text = data['output'][0]['answer']
            if len(answer_text.split()) == 1:
                yield {
                    'input': input_text,
                    'answer': answer_text,
                }


# REMINDER: You need to add whatever preprocessing functions you've written to
# this dict to make them available to the training scripts.
PREPROCESSORS = {
    'csv': preprocess_csv,
    'tsv': preprocess_tsv,
    'jsonl': preprocess_jsonl,
    'jsonl-string': preprocess_jsonl_string,
    'multirc': preprocess_multirc,
    'wsc': preprocess_wsc,
    'copa': preprocess_copa,
    'record': preprocess_record,
    'kilt': preprocess_kilt,
}

