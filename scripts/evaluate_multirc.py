import argparse
import csv
import json
import pathlib

import numpy as np


def compute_f1a(labels, preds):
    agree = (labels * preds).sum()
    p = agree / preds.sum()
    r = agree / labels.sum()
    return 2 * p * r / (p + r)


def compute_em(labels, preds, qids):
    numerator = 0
    denominator = 0
    for i in range(np.max(qids)):
        denominator += 1
        numerator += np.all(labels[qids==i] * preds[qids==i])
    return numerator / denominator
        

def main(args):
    labels = []
    qids = []
    qid = 0
    with open(args.ground_truth, 'r') as f:
        for line in f:
            data = json.loads(line)
            for question in data['passage']['questions']:
                for answer in question['answers']:
                    labels.append(answer['label'])
                    qid.append(qid)
                qid += 1
    labels = np.array(labels)
    qids = np.array(qids)
    for ckpt_dir in args.ckpt_dirs:
        fields = ckpt_dir.stem.split('_')
        ckpt_file = ckpt_dir / 'predictions'
        with open(ckpt_file, 'r') as f:
            preds = np.array([int(line.strip()) for line in f])
        f1a = compute_f1a(labels, preds)
        em = compute_em(labels, preds, qids)
        out = ','.join([*fields, f1a, em])
        print(out)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ground_truth', type=str)
    parser.add_argument('ckpt_dirs', nargs='+', type=pathlib.Path)

    args = parser.parse_args()

    main(args)

