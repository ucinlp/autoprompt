import os
import json
import time
import random
import asyncio
import argparse
import numpy as np
from collections import defaultdict
from tqdm import tqdm
from qwikidata.linked_data_interface import get_entity_dict_from_api
from qwikidata.entity import WikidataItem, WikidataProperty
from pytorch_transformers import BertTokenizer
import utils

COUNT = 0

def get_id_from_url(url):
    """
    Extract Wikidata entity id from URL
    """
    return url.split('/')[-1]


async def map_async(fn, iterator, count, max_tasks=10, sleep_time=0.01):
    tasks = set()

    for x in iterator:
        if len(tasks) >= max_tasks:
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)

        new_task = asyncio.ensure_future(fn(x))
        tasks.add(new_task)
        await asyncio.sleep(random.random() * sleep_time)

        global COUNT
        if COUNT >= count:
            return

    await asyncio.wait(tasks)


async def increment_count():
    global COUNT
    COUNT += 1


async def get_fact(query, args, tokenizer, trex_set, common_vocab, f_out):
    """
    Collect more facts for the TREx-train set from LPAQA
    """
    line = query.strip().split('\t')
    sub_url, sub, obj_url, obj = line
    sub_id = get_id_from_url(sub_url)
    obj_id = get_id_from_url(obj_url)

    # First, make sure fact is not in TREx test set
    if (sub_id, obj_id) in trex_set:
        return

    # Make sure object is a single token
    if len(tokenizer.tokenize(obj)) != 1:
        return

    # Make sure object is in common vocab subset
    if obj not in common_vocab:
        return

    # Make sure subject is prominent (has a Wikipedia page)
    try:
        q_dict = get_entity_dict_from_api(sub_id)
        q = WikidataItem(q_dict)
        if not q.get_sitelinks():
            return
    except ValueError:
        return

    # Some entities don't have labels so the subject label is the URI
    if sub_id == sub:
        return

    # print('Writing fact: {} - {}', sub, obj)
    f_out.write(json.dumps({'sub_uri': sub_id, 'obj_uri': obj_id, 'sub_label': sub, 'obj_label': obj}) + '\n')

    # Increment global count
    await increment_count()


async def main(args):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    # Load common vocab subset
    common_vocab = utils.load_vocab(args.common_vocab_file)

    # Go though TREx test set and save every sub-obj pair/fact in a dictionary
    trex_set = set()
    with open(args.trex_file, 'r') as f_in:
        lines = f_in.readlines()
        for line in tqdm(lines):
            line = json.loads(line)
            trex_set.add((line['sub_uri'], line['obj_uri']))

    # Get relation ID, i.e. P108
    filename = os.path.basename(os.path.normpath(args.in_file))
    rel_id = filename.split('.')[0]

    queries = []
    with open(args.in_file, 'r') as f_in:
        queries = f_in.readlines()

    with open(args.out_file, 'a+') as f_out:
        await map_async(lambda q: get_fact(q, args, tokenizer, trex_set, common_vocab, f_out), queries, args.count, args.max_tasks, args.sleep_time)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather more facts for TREx-train')
    parser.add_argument('in_file', type=str, help='TSV file containing Wikidata subject-relation-object triplets')
    parser.add_argument('out_file', type=str, help='JSONL file with new facts')
    parser.add_argument('--trex_file', type=str, help='Path to TREx test set for a relation')
    parser.add_argument('--common_vocab_file', type=str, help='File containing common vocab subset')
    parser.add_argument('--count', type=int, default=1000, help='Number of new samples to gather from Wikidata for a particular relation')
    parser.add_argument('--sleep_time', type=float, default=0.01)
    parser.add_argument('--max_tasks', type=int, default=50)
    args = parser.parse_args()

    start = time.time()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    # Zero-sleep to allow underlying connections to close
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    # Measure elapsed time
    end = time.time()
    print('Elapsed time: {} sec'.format(end - start))
