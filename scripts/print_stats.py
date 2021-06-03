import argparse
from collections import defaultdict
import csv

import numpy as np
import scipy.stats


parser = argparse.ArgumentParser()
parser.add_argument('inputs', type=str, nargs='+')
parser.add_argument('--p_cutoff', type=float, default=0.05)
args = parser.parse_args()


keys = set()
def get_scores(fname):
    scores = defaultdict(list)
    with open(fname, 'r') as f:
        reader = csv.DictReader(f)
        header = next(reader)
        for line in reader:
            for k, v in line.items():
                scores[k].append(float(v))
                keys.add(k)
    return {k: np.array(v) for k, v in scores.items()}


all_scores = {fname: get_scores(fname) for fname in args.inputs}


def format_name(fname):
    _, _, _, _, trig, _, ftune, *_ = fname.split('_')
    ftune, *_ = ftune.split('/')
    return f'{trig:>5} {ftune}'.ljust(32)


def print_results(all_scores, key):
    n = len(args.inputs)
    results = [['-' for _ in range(n)] for _ in range(n)]
    for i, row in enumerate(args.inputs):
        for j, col in enumerate(args.inputs):
            _, p_value = scipy.stats.ttest_ind(
                all_scores[row][key],
                all_scores[col][key],
                alternative='greater',
                equal_var=False
            )
            if p_value < args.p_cutoff:
                results[i][j] = '>'

    for i in range(n):
        wins = sum(results[i][j] == '>' for j in range(n))
        row = format_name(args.inputs[i]) + ' '.join(results[i][j] for j in range(n)) + ' ' + str(wins)
        print(row)



for key in keys:
    print(key)
    print_results(all_scores, key)

