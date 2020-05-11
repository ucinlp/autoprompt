import os
import argparse
import numpy as np
from astropy.table import Table
from tabulate import tabulate

# tabulate.PRESERVE_WHITESPACE = True

def compare_and_display(args):
    # NOTE: CONFIGURABLE
    p1 = 'LMAT'
    p2 = 'LPAQA'
    comp_metric = 'Test P@1'

    headers = ['', p1, '', p2]
    table = []
    # Number of times p1 was greater than p2
    num_gt = 0
    num_lt = 0
    num_eq = 0

    for f in os.listdir(args.in_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.tex'):
            rel_name = os.path.basename(filename).replace('.tex', '')
            filepath = os.path.join(args.in_dir, filename)
            tab = Table.read(filepath).to_pandas()

            # NOTE: CONFIGURABLE
            # 0 -> LAMA (data=L), 2 -> LPAQA (Top1), data=L, 4 -> Init=man, cand=10, data=L
            row_p1 = tab.loc[8]
            val_p1 = row_p1[comp_metric]
            row_p2 = tab.loc[2]
            val_p2 = row_p2[comp_metric]

            if val_p1 > val_p2:
                comp = '>'
                num_gt += 1
            elif val_p1 < val_p2:
                comp = '<'
                num_lt += 1
            else:
                comp = '='
                num_eq += 1

            table.append([rel_name, val_p1, comp, val_p2])

    print('Metric to compare:', comp_metric)
    print(tabulate(table, headers=headers, tablefmt='pretty'))
    print('Stats:')
    print('{} > {} for {} relations'.format(p1, p2, num_gt))
    print('{} < {} for {} relations'.format(p1, p2, num_lt))
    print('{} = {} for {} relations'.format(p1, p2, num_eq))


def compute_oracle_score(args):
    # metric = 'Test P@1'
    metric = 'Dev P@1'

    # Relation-level scores
    rel_lvl_scores = []
    for f in os.listdir(args.in_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.tex'):
            rel_name = os.path.basename(filename).replace('.tex', '')
            filepath = os.path.join(args.in_dir, filename)
            tab = Table.read(filepath).to_pandas()

            # 4 -> Init=man, cand=10, data=L ... 9 -> Init=man, cand=10, data=F
            scores = []
            row_L = tab.loc[4]
            scores.append(row_L[metric])
            row_B = tab.loc[5]
            scores.append(row_B[metric])
            row_C = tab.loc[6]
            scores.append(row_C[metric])
            row_D = tab.loc[7]
            scores.append(row_D[metric])
            row_E = tab.loc[8]
            scores.append(row_E[metric])
            row_F = tab.loc[9]
            scores.append(row_F[metric])

            # For each relation, pick the best score
            best_score = max(scores)
            rel_lvl_scores.append(best_score)
    
    print('Oracle Score:', np.mean(rel_lvl_scores))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Compare two sets of results with LATEX tables')
    parser.add_argument('in_dir', type=str, help='Directory containing result table TEX files')
    # parser.add_argument('out_dir', type=str, help='Directory to store JSONL files')
    args = parser.parse_args()

    compare_and_display(args)

    # Compute oracle score for a method (ex: man_cand10)
