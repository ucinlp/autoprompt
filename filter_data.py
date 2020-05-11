import os
import json
import random
import argparse
import utils

def filter_data(args):
    # Go through data and remove duplicates
    data = []
    with open(args.in_file, 'r') as f_in:
        unique_facts = set()
        samples = f_in.readlines()
        for sample in samples:
            sample = json.loads(sample)
            sub_uri = sample['sub_uri']
            pred_id = sample['predicate_id']
            obj_uri = sample['obj_uri']
            fact = (sub_uri, pred_id, obj_uri)

            if fact not in unique_facts:
                unique_facts.add(fact)
                data.append(sample)
    return data


def get_data(args):
    data = []
    with open(args.in_file, 'r') as f_in:
        # unique_facts = set()
        samples = f_in.readlines()
        for sample in samples:
            sample = json.loads(sample)
            data.append(sample)
    return data


def main(args):
    # data = filter_data(args)
    # print('Data size after filtering:', len(data))
    data = get_data(args)
    # Subsample
    with open(args.out_file, 'w+') as f_out:
        subsample = random.sample(data, args.count)
        for s in subsample:
            f_out.write(json.dumps(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter and sample facts from TREx-train data for a relation')
    parser.add_argument('--in_file', type=str, help='File containing fact samples for a TREx relation')
    parser.add_argument('--out_file', type=str, help='File to store filtered and sampled facts')
    parser.add_argument('--count', type=int, help="Number of samples to subsample")
    args = parser.parse_args()
    main(args)
