import os
import json
import random
import argparse
import utils


def main(args):
    for f in os.listdir(args.in_dir):
        filename = os.fsdecode(f)
        if filename.endswith('.jsonl'):
            rel_id = os.path.basename(filename).replace('.jsonl', '')
            print('Sampling from {}'.format(rel_id))
            filepath_in = os.path.join(args.in_dir, filename)
            filepath_out = os.path.join(args.out_dir, rel_id + '.jsonl')
            # Make directories in path if they don't exist
            os.makedirs(os.path.dirname(filepath_out), exist_ok=True)

            # Get data
            samples = []
            with open(filepath_in, 'r') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    sample = json.loads(line)
                    samples.append(sample)

            # Subsample
            with open(filepath_out, 'w+') as f_out:
                if len(samples) < args.count:
                    count = len(samples)
                else:
                    count = args.count
                subsample = random.sample(samples, count)
                for s in subsample:
                    f_out.write(json.dumps(s) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Subsample from a set of facts in a JSONL file')
    parser.add_argument('in_dir', type=str, help='File containing facts to sample from')
    parser.add_argument('out_dir', type=str, help='File to store subsampled facts')
    parser.add_argument('--count', type=int, default=100, help="Number of samples to subsample")
    args = parser.parse_args()
    main(args)
