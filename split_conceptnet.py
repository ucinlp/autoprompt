import os
import json
import argparse
from tqdm import tqdm

PRED_LIST = [
    'HasSubevent',
    'MadeOf',
    'HasPrerequisite',
    'MotivatedByGoal',
    'AtLocation',
    'CausesDesire',
    'IsA',
    'NotDesires',
    'Desires',
    'CapableOf',
    'PartOf',
    'HasA',
    'UsedFor',
    'ReceivesAction',
    'Causes',
    'HasProperty'
]

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

def split(args):
    samples = read_jsonl(args.in_file)
    p1, p2, p3, p4, p5, p6, p7, p8, p9, p10, p11, p12, p13, p14, p15, p16 = [], [], [], [], [], [], [], [], [], [], [], [], [], [], [], []
    for sample in tqdm(samples):
        pred = sample['pred']
        masked_sents = sample['masked_sentences']
        if 'obj_label' in sample:
            obj = sample['obj_label']
        else:
            obj = sample['obj']
        if 'sub_label' in sample:
            sub = sample['sub_label']
        else:
            sub = sample['sub']

        if pred == 'HasSubevent':
            p1.append(sample)
        elif pred == 'MadeOf':
            p2.append(sample)
        elif pred == 'HasPrerequisite':
            p3.append(sample)
        elif pred == 'MotivatedByGoal':
            p4.append(sample)
        elif pred == 'AtLocation':
            p5.append(sample)
        elif pred == 'CausesDesire':
            p6.append(sample)
        elif pred == 'IsA':
            p7.append(sample)
        elif pred == 'NotDesires':
            p8.append(sample)
        elif pred == 'Desires':
            p9.append(sample)
        elif pred == 'CapableOf':
            p10.append(sample)
        elif pred == 'PartOf':
            p11.append(sample)
        elif pred == 'HasA':
            p12.append(sample)
        elif pred == 'UsedFor':
            p13.append(sample)
        elif pred == 'ReceivesAction':
            p14.append(sample)
        elif pred == 'Causes':
            p15.append(sample)
        else:
            p16.append(sample)

    write_jsonl(os.path.join(args.out_dir, 'P101', 'P101.jsonl'), p1)
    write_jsonl(os.path.join(args.out_dir, 'P102', 'P102.jsonl'), p2)
    write_jsonl(os.path.join(args.out_dir, 'P103', 'P103.jsonl'), p3)
    write_jsonl(os.path.join(args.out_dir, 'P104', 'P104.jsonl'), p4)
    write_jsonl(os.path.join(args.out_dir, 'P105', 'P105.jsonl'), p5)
    write_jsonl(os.path.join(args.out_dir, 'P106', 'P106.jsonl'), p6)
    write_jsonl(os.path.join(args.out_dir, 'P107', 'P107.jsonl'), p7)
    write_jsonl(os.path.join(args.out_dir, 'P108', 'P108.jsonl'), p8)
    write_jsonl(os.path.join(args.out_dir, 'P109', 'P109.jsonl'), p9)
    write_jsonl(os.path.join(args.out_dir, 'P110', 'P110.jsonl'), p10)
    write_jsonl(os.path.join(args.out_dir, 'P111', 'P111.jsonl'), p11)
    write_jsonl(os.path.join(args.out_dir, 'P112', 'P112.jsonl'), p12)
    write_jsonl(os.path.join(args.out_dir, 'P113', 'P113.jsonl'), p13)
    write_jsonl(os.path.join(args.out_dir, 'P114', 'P114.jsonl'), p14)
    write_jsonl(os.path.join(args.out_dir, 'P115', 'P115.jsonl'), p15)
    write_jsonl(os.path.join(args.out_dir, 'P116', 'P116.jsonl'), p16)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Splits ConceptNet data into each relation/predicate')
    parser.add_argument('in_file', type=str, help='Path input file')
    parser.add_argument('out_dir', type=str, help='Path to output directory')
    args = parser.parse_args()
    split(args)
    print('Finished.')
