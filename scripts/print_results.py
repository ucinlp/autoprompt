#! /usr/bin/env python

import argparse
import csv
from pathlib import Path
import re
import sys


ACC_REGEX = re.compile('(?<=Accuracy:  ).*')


def main(args):
    writer = csv.writer(sys.stdout)
    stem = args.input.stem
    fields = stem.split('_')
    with open(args.input, 'r') as f:
        for line in f:
            pass
        match = ACC_REGEX.search(line)
        if match:
            acc = match.group(0)
            writer.writerow((*fields, acc))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=Path)
    args = parser.parse_args()

    main(args)
