import os
import json
import random
import argparse

def write_jsonl(filename, json_obj_list):
    """
    Create JSONL file from list of JSON objects (dictionaries)
    """
    with open(filename, 'w') as f:
        for json_obj in json_obj_list:
            f.write(json.dumps(json_obj) + '\n')

def read_jsonl(filename):
    """
    Reads a jsonl file by filename. Returns an iterator.
    """
    with open(filename) as f:
        for i, l in enumerate(f.readlines()):
            l = l.strip()
            try:
                yield json.loads(l)
            except Exception as e:
                print('Error reading JSONL', e, i, l)

def train_val_test_split(data, train_ratio=0.9, val_ratio=0.05):
    """
    Test ratio is 1 - (train_ratio + val_ratio)
    """
    random.shuffle(data)
    num_data = len(data)
    train_index = int(num_data * train_ratio)
    train_set = data[:train_index]
    val_index = train_index + int(num_data * val_ratio)
    val_set = data[train_index:val_index]
    test_set = data[val_index:]
    return train_set, val_set, test_set

def train_val_split(data, train_ratio=0.9, val_ratio=0.05):
    """
    Test ratio is 1 - (train_ratio + val_ratio)
    """
    random.shuffle(data)
    num_data = len(data)
    train_index = int(num_data * train_ratio)
    train_set = data[:train_index]
    val_index = train_index + int(num_data * val_ratio)
    val_set = data[train_index:val_index]
    return train_set, val_set

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Randomly splits a JSONL file into train, validation and test subsets')
    parser.add_argument('src', type=str, help='Path input file')
    parser.add_argument('out', type=str, help='Path to output directory')
    parser.add_argument('--train-ratio', type=float, default=0.9)
    parser.add_argument('--val-ratio', type=float, default=0.05)
    args = parser.parse_args()

    data = list(read_jsonl(args.src))

    """
    # Train test split
    train_set, val_set, test_set = train_val_test_split(data, args.train_ratio, args.val_ratio)

    # Create JSONL file
    write_jsonl(os.path.join(args.out, 'train.jsonl'), train_set)
    write_jsonl(os.path.join(args.out, 'dev.jsonl'), val_set)
    if len(test_set) > 0:
        write_jsonl(os.path.join(args.out, 'test.jsonl'), test_set)
    """

    train_set, val_set = train_val_split(data, args.train_ratio, args.val_ratio)
    write_jsonl(os.path.join(args.out, 'train.jsonl'), train_set)
    write_jsonl(os.path.join(args.out, 'dev.jsonl'), val_set)
