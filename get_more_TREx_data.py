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
from qwikidata.sparql import get_subclasses_of_item, return_sparql_query_results
from pytorch_transformers import BertTokenizer
import utils

COUNT = 0

async def map_async(fn, iterator, max_tasks=10, sleep_time=0.01):
    tasks = set()
    
    for x in iterator:
        if len(tasks) >= max_tasks:
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        new_task = asyncio.ensure_future(fn(x))
        tasks.add(new_task)
        await asyncio.sleep(random.random() * sleep_time)

    await asyncio.wait(tasks)


async def increment_count():
    global COUNT
    COUNT += 1


async def process_fact(query, args, tokenizer, trex_set, common_vocab, f_out):
    """
    Takes in a tuple of (sub_id, obj_id, sub_label, obj_label) and writes it to out file if it's a valid fact
    """
    # line = query.strip().split('\t')
    # sub_url, sub, obj_url, obj = line
    # sub_id = utils.get_id_from_url(sub_url)
    # obj_id = utils.get_id_from_url(obj_url)

    sub_id, obj_id, sub_label, obj_label = query

    # First, make sure fact is not in TREx test set
    if (sub_id, obj_id) in trex_set:
        return

    # Make sure object is a single token
    if len(tokenizer.tokenize(obj_label)) != 1:
        return

    # Make sure object is in common vocab subset
    if obj_label not in common_vocab:
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
    if sub_id == sub_label:
        return

    f_out.write(json.dumps({'sub_uri': sub_id, 'obj_uri': obj_id, 'sub_label': sub_label, 'obj_label': obj_label}) + '\n')

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

    # Get facts with SPARQL query
    start = time.perf_counter()
    offset = 0
    keep_looping = True
    while keep_looping:
        # Exit early if scraping is taking more than half an hour
        if time.perf_counter() - start > 1800:
            print('Scraping taking more than half an hour. Exiting early.')
            break

        try:
            sparql_query = """
            SELECT ?item ?itemLabel ?value ?valueLabel
            WHERE { ?item wdt:""" + args.rel + """ ?value SERVICE wikibase:label { bd:serviceParam wikibase:language "[AUTO_LANGUAGE],en". } }
            LIMIT """ + str(args.query_limit) + """ OFFSET """ + str(offset)
            # If there are no more facts to query, this function will return json.decoder.JSONDecodeError which inherits from ValueError
            res = return_sparql_query_results(sparql_query)

            if 'results' in res and 'bindings' in res['results'] and res['results']['bindings']:
                queries = []
                for b in res['results']['bindings']:
                    sub_id = utils.get_id_from_url(b['item']['value'])
                    obj_id = utils.get_id_from_url(b['value']['value'])
                    sub_label = b['itemLabel']['value']
                    obj_label = b['valueLabel']['value']
                    queries.append((sub_id, obj_id, sub_label, obj_label))

                with open(args.out_file, 'a+') as f_out:
                    await map_async(lambda q: process_fact(q, args, tokenizer, trex_set, common_vocab, f_out), queries, args.max_tasks, args.sleep_time)

                    if args.num_samples > 0:
                        global COUNT
                        if COUNT >= args.num_samples:
                            return

                offset += args.query_limit
                print('OFFSET:', offset)
            else:
                # Results (bindings) is empty which means WQS ran out of facts
                print('Query result is empty.')
                keep_looping = False
        except json.decoder.JSONDecodeError:
            # Case where Wikidata Query Service has no more facts to return
            print('Wikidata Query Service ran out of facts.')
            keep_looping = False

    # Measure elapsed time
    end = time.perf_counter() - start
    print('Elapsed time: {} sec'.format(round(end, 2)))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Gather more facts for TREx-train')
    # parser.add_argument('in_file', type=str, help='TSV file containing Wikidata subject-relation-object triplets')
    parser.add_argument('rel', type=str, help='Wikidata relation ID')
    parser.add_argument('out_file', type=str, help='JSONL file with new facts')
    parser.add_argument('--trex_file', type=str, help='Path to TREx TEST set for a relation')
    parser.add_argument('--common_vocab_file', type=str, help='File containing common vocab subset')
    parser.add_argument('--num_samples', type=int, default=-1, help='Number of new samples to gather from Wikidata for a particular relation')
    parser.add_argument('--query_limit', type=int, default=1000, help='SPARQL limit')
    parser.add_argument('--sleep_time', type=float, default=0.01)
    parser.add_argument('--max_tasks', type=int, default=50)
    args = parser.parse_args()

    print('Collecting more data for relation {}...'.format(args.rel))
    # start = time.perf_counter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    # Zero-sleep to allow underlying connections to close
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    # Measure elapsed time
    # end = time.perf_counter() - start
    # print('Elapsed time: {} sec'.format(round(end, 2)))
