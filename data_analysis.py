import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from collections import Counter
import utils

def inspect_relation(args, filename, model_vocab, common_vocab):
    if os.path.isfile(filename):
        rel_name = os.path.basename(filename).replace('.jsonl', '')
    else:
        rel_name = filename.replace('.jsonl', '')

    facts = utils.load_TREx_data(args, filename)
    num_common = 0 # Number of samples in the common vocab subset which is a subset of model vocab
    num_model = 0 # Number of samples in model vocab but not in common vocab
    num_neither = 0 # Number of samples in neither model nor common vocab
    for fact in tqdm(facts):
        sub, obj = fact
        # First check if object is in common vocab
        if obj in common_vocab:
            num_common += 1
        else:
            # If not in common vocab, could be in model vocab
            if obj in model_vocab:
                num_model += 1
            else:
                # Not in common or model vocab
                num_neither += 1
    assert len(facts) == num_common + num_model + num_neither
    print('{} -> num facts: {}, num common: {}, num model: {}, num neither: {}'.format(rel_name, len(facts), num_common, num_model, num_neither))

    # Plot distribution of gold objects
    obj_set = Counter([obj for sub, obj in facts])
    top_obj_set = obj_set.most_common(10)
    print(top_obj_set)
    print()
    gold_objs = pd.DataFrame(top_obj_set, columns=['obj', 'freq'])

    fig, ax = plt.subplots()
    gold_objs.sort_values(by='freq').plot.barh(x='obj', y='freq', ax=ax)
    plt.savefig(os.path.join(args.out_dir, rel_name + '.png'), bbox_inches='tight')
    plt.close()

def inspect_dataset(args):
    """
    Count the number of samples that have tokens that are not in the common vocab subset
    """
    # Make out dir if it doesn't exist
    os.makedirs(args.out_dir, exist_ok=True)
    
    for f in os.listdir(args.data_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.jsonl'):
            inspect_relation(args, os.path.join(args.data_dir, filename), model_vocab, common_vocab)


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, help='Directory containing TREx relation files')
    parser.add_argument('--data_file', type=str, help='Optional: File containing data for a particular TREx relation')
    parser.add_argument('--out_dir', type=str, help='Directory to store output plots')
    parser.add_argument('--model_vocab_file', type=str, help="File containing a specific model's vocab")
    parser.add_argument('--common_vocab_file', type=str, help='File containing common vocab subset')
    args = parser.parse_args()

    model_vocab = utils.load_vocab(args.model_vocab_file)
    print('Model vocab size:', len(model_vocab))
    common_vocab = utils.load_vocab(args.common_vocab_file)
    print('Common vocab size:', len(common_vocab))
    print()

    if args.data_file:
        inspect_relation(args, args.data_file, model_vocab, common_vocab)
    else:
        inspect_dataset(args)
