#!/usr/bin/env python

import argparse
import glob
import json
import logging
import os

import sklearn.metrics


logger = logging.getLogger(__name__)


METRICS = {
    'matthews': sklearn.metrics.matthews_corrcoef,
    'f1': sklearn.metrics.f1_score,
}


def main(args):
    logger.info('Loading ground truth...')
    labels =[]
    label_map = {}
    if args.pos_label is not None:
        label_map[args.pos_label] = 1
    metric = METRICS[args.metric]
    with open(args.dataset, 'r') as f:
        for line in f:
            data = json.loads(line)
            label = data['label']
            if label not in label_map:
                if len(label_map) == 1 and args.pos_label is not None:
                    label_map[label] = 0
                else:
                    label_map[label] = len(label_map)
            label_id = label_map[label]
            labels.append(label_id)
    logger.info(f'Label map: {label_map}')

    for subdir in glob.glob(args.preds):
        logger.info(f'Processing: {subdir}')
        pred_fname = os.path.join(subdir, 'predictions')
        preds = []
        try:
            with open(pred_fname, 'r') as f:
                preds = [label_map[line.strip()] for line in f]
        except FileNotFoundError:
            continue
        else:
            score = metric(labels, preds) 
            print(score)




            


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=str)
    parser.add_argument('preds', type=str)
    parser.add_argument('-m', '--metric', type=str, choices=list(METRICS.keys()))
    parser.add_argument('-p', '--pos_label', type=str, default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    main(args)

