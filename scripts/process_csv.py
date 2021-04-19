import argparse
from collections import defaultdict
import csv
from statistics import mean, stdev

parser = argparse.ArgumentParser()
parser.add_argument('input', type=str)
args = parser.parse_args()

scores = defaultdict(list)
with open(args.input, 'r') as f:
    reader = csv.DictReader(f)
    header = next(reader)
    for line in reader:
        for k, v in line.items():
            scores[k].append(float(v))

for k, v in scores.items():
    if k == 'seed':
        continue
    print(f'{k} Mean: {100*mean(v):0.2f} Std. Dev: ({100*stdev(v):0.2f})')
