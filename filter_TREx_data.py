import os
import json
import random
import argparse
from collections import defaultdict
from tqdm import tqdm
import utils


def main(args):
    for f in os.listdir(args.in_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.jsonl'):
            rel_id = os.path.basename(filename).replace('.jsonl', '')
            print('Filtering {}'.format(rel_id))
            filepath_in = os.path.join(args.in_dir, filename)
            fact_to_ctx = defaultdict(list)
            with open(filepath_in, 'r') as f_in:
                lines = f_in.readlines()
                for line in tqdm(lines):
                    sample = json.loads(line)
                    fact = (sample['obj_uri'], sample['obj_label'], sample['sub_uri'], sample['sub_label'], sample['predicate_id'])
                    evidences = sample['evidences']
                    # TODO: handle case where evidences is empty
                    fact_to_ctx[fact].append(evidences[0])

            # Go through fact_to_ctx dictionary and write facts and their context sentences to out file
            filepath_out = os.path.join(args.out_dir, rel_id + '.jsonl')
            with open(filepath_out, 'w+') as f_out:
                for key, val in tqdm(fact_to_ctx.items()):
                    obj_uri, obj_label, sub_uri, sub_label, predicate_id = key
                    f_out.write(json.dumps({
                        'obj_uri': obj_uri,
                        'obj_label': obj_label,
                        'sub_uri': sub_uri,
                        'sub_label': sub_label,
                        'predicate_id': predicate_id,
                        'evidences': val
                    }) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Filter facts after parsing the full TREx dataset (not including test). Combine duplicate facts with different context sentences.')
    parser.add_argument('in_dir', type=str, help='File containing fact samples for a TREx relation')
    parser.add_argument('out_dir', type=str, help='File to store filtered and sampled facts')
    args = parser.parse_args()
    main(args)
