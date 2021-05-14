import argparse
from collections import defaultdict
import csv

import numpy as np
import scipy


parser = argparse.ArgumentParser()
parser.add_argument('inputs', type=str, nargs='+')
parser.add_argument('p_cutoff', type=float, default=0.05)
args = parser.parse_args()

def get_scores(fname):
    scores = defaultdict(list)
    with open(args.fname, 'r') as f:
        reader = csv.DictReader(f)
        header = next(reader)
        for line in reader:
            for k, v in line.items():
                scores[k].append(float(v))
    return {k: np.array(v) for k, v in scores.items()}


all_scores = {fname: get_scores(fname) for fname in args.inputs}
n = len(args.inputs)
results = [['-' for _ in range(n)] for _ in range(n)]
for i, row in enumerate(args.inputs):
    for j, col in enumerate(args.inputs):
        _, p_value = scipy.stats.ttest_ind(
            all_scores[row],
            all_scores[col],
            alternative='greater',
            equal_var=False
        )
        if p_value < args.p_cutoff:
            results[i][j] = '>'


for i in range(n):
    row = ' '.join(results[i][j] for j in range(n))
    print(row)

