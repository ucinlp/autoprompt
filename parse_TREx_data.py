import os
import re
import json
import time
import random
import argparse
import aiohttp
import asyncio
import spacy
from tqdm import tqdm
from nltk import tokenize
from pytorch_transformers import BertTokenizer
from aiohttp.client_exceptions import ContentTypeError, ClientConnectorError
import utils
import constants

nlp = spacy.load('en_core_web_sm')

# Entity ID to label in canonical form
EID_TO_LABEL = {}

def get_POS(obj_label):
    """
    Get Part-Of-Speech tag of an object label (single-token)
    """
    doc = nlp(obj_label)
    return doc[0].pos_


def get_NER(obj_label):
    """
    Get Named Entity Recognition label of an object label (single-token)
    """
    doc = nlp(obj_label)
    if not doc.ents:
        return None
    return doc.ents[0].label_


def is_complete(d, threshold):
    """
    Checks if all values in a dictionary are greater than equal to the threshold number
    """
    for key, val in d.items():
        if val < threshold:
            return False
    return True


def load_TREx_test(trex_test_dir):
    print('Loading facts from TREx-test set')
    trex_set = set()
    for filename in tqdm(trex_test_dir):
        filename = os.fsdecode(filename)
        if filename.endswith('.jsonl'):
            filepath = os.path.join(args.trex_test_dir, filename)
            with open(filepath, 'r') as f_in:
                lines = f_in.readlines()
                for line in lines:
                    line = json.loads(line)
                    trex_set.add((line['sub_uri'], line['predicate_id'], line['obj_uri']))
    return trex_set


async def fetch_label(session, eid):
    """
    Gets entity's label (canonical form)
    """
    url = "https://www.wikidata.org/wiki/Special:EntityData/{}.json".format(eid)
    try:
        async with session.get(url) as response:
            data = await response.json()
            if 'entities' in data:
                if eid in data['entities']:
                    if 'labels' in data['entities'][eid]:
                        if 'en' in data['entities'][eid]['labels']:
                            if 'value' in data['entities'][eid]['labels']['en']:
                                return data['entities'][eid]['labels']['en']['value']
            return None
    except (ContentTypeError, ClientConnectorError) as e:
        return None


async def parse_triple(args, triple_dict, text, sents, sent_len_list, relations, rel_to_count, trex_set, common_vocab, tokenizer):
    async with aiohttp.ClientSession() as session:
        pred_dict = triple_dict['predicate']
        obj_dict = triple_dict['object']
        sub_dict = triple_dict['subject']
        pred_id = utils.get_id_from_url(pred_dict['uri'])
        obj_id = utils.get_id_from_url(obj_dict['uri']) # some object IDs turn out to be "XMLSchema#dateTime" which is invalid so skip
        sub_id = utils.get_id_from_url(sub_dict['uri'])

        # If the object's surface form is multiple tokens then its canonical form is likely to also be multiple tokens
        # Skipping facts with objects with multi-token surface form will speed up the whole process
        # if len(tokenizer.tokenize(obj_dict['surfaceform'])) > 1:
        #     return

        # If a fact's annotator is Simple_Coreference, the sub or obj can be something like "It"
        sub_annot = sub_dict['annotator']
        obj_annot = obj_dict['annotator']
        simp = 'Simple_Coreference'
        triple = (sub_id, pred_id, obj_id)

        if pred_id in relations and sub_id.startswith('Q') and pred_id.startswith('P') and obj_id.startswith('Q') and sub_annot != simp and obj_annot != simp and triple not in trex_set:
            ### Get the canonical form of the object ###
            global EID_TO_LABEL
            if obj_id not in EID_TO_LABEL:
                obj_label = await fetch_label(session, obj_id)
                if not obj_label:
                    return
                # Cache object's canonical form
                EID_TO_LABEL[obj_id] = obj_label
            else:
                obj_label = EID_TO_LABEL[obj_id]

            # Skip objects that have canonical forms with multiple tokens
            if len(tokenizer.tokenize(obj_label)) != 1:
                return

            ### Get the canonical form of the subject ###
            if sub_id not in EID_TO_LABEL:
                sub_label = await fetch_label(session, sub_id)
                if not sub_label:
                    return
                # Cache subject's canonical form
                EID_TO_LABEL[sub_id] = sub_label
            else:
                sub_label = EID_TO_LABEL[sub_id]

            ### Extract context sentence ###
            ctx_sent = ''
            masked_sent = ''
            # Object appears in context sentence
            if 'boundaries' in obj_dict and obj_dict['boundaries']:
                obj_start = obj_dict['boundaries'][0] # inclusive
                obj_end = obj_dict['boundaries'][1] # exclusive
                for l, sent in zip(sent_len_list, sents):
                    if obj_end < l:
                        ctx_sent = sent
                        break
                # Mask out the object in context sentence using the provided boundaries of the object
                obj_len = obj_end - obj_start
                ctx_sent_start = text.index(ctx_sent)
                rel_obj_start = obj_start - ctx_sent_start
                masked_sent = ctx_sent[:rel_obj_start] + constants.MASK + ctx_sent[rel_obj_start+obj_len:]

            # print(sub_label, pred_id, obj_label)
            # print(ctx_sent)
            # print(masked_sent)
            # print()

            # Finally write fact to file if it passes all criteria
            filepath = os.path.join(args.out_dir, pred_id + '.jsonl')
            # Make directories in path if they don't exist
            os.makedirs(os.path.dirname(filepath), exist_ok=True)

            fact_json = {
                'obj_uri': obj_id,
                'obj_label': obj_label,
                'sub_uri': sub_id,
                'sub_label': sub_label,
                'predicate_id': pred_id,
                'evidences': []
            }
            # Only include context sentence if masked sentence isn't empty which means the context was valid
            if masked_sent:
                fact_json['evidences'].append({
                    'sub_surface': sub_dict['surfaceform'],
                    'obj_surface': obj_dict['surfaceform'],
                    'masked_sentence': masked_sent
                })

            with open(filepath, 'a+') as f_out:
                # Update rel_to_count dict
                rel_to_count[pred_id] += 1
                f_out.write(json.dumps(fact_json) + '\n')


async def main(args):
    # Set up relations list containing all of the TREx relations
    relations = constants.TREX_RELATIONS
    # relations = [
    #     'P30', 'P36', 'P37', 'P39',
    #     'P101', 'P103', 'P108', 'P127', 'P136', 'P138', 'P140', 'P159',
    #     'P176', 'P178', 'P190', 'P264', 'P276', 'P361', 'P407', 'P413',
    #     'P449', 'P495', 'P527', 'P740', 'P937', 'P1001', 'P1303', 'P1376'
    # ]

    # Initialize dictionary of relation to sample count
    rel_to_count = dict.fromkeys(relations, 0)

    # Keep track of already removed relations
    removed = []

    # Load every fact from TREx-test set
    trex_set = load_TREx_test(os.listdir(args.trex_test_dir))

    # Load common vocab
    common_vocab = utils.load_vocab(args.common_vocab_file)

    # Load Bert tokenizer
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    # Download the full TREx dataset here https://hadyelsahar.github.io/t-rex/downloads/
    dir_list = os.listdir(args.in_dir)

    # Randomize the order of doc sets which will address the issue of sequential data collection
    random.shuffle(dir_list)

    doc_set_count = 0
    for filename in dir_list:
        filename = os.fsdecode(filename)
        doc_set_count += 1
        print('Parsing {}: {}/{}'.format(filename, doc_set_count, len(dir_list)))
        if filename.endswith('.json'):
            start = time.perf_counter()
            filepath = os.path.join(args.in_dir, filename)

            with open(filepath, 'r') as f_in:
                docs = json.load(f_in)
                # random.shuffle(docs) # NOTE: Shuffle documents for debugging
                queries = []
                for d, doc in enumerate(tqdm(docs)):
                    text = doc['text']
                    sents = tokenize.sent_tokenize(text)
                    sent_len_sum = 0
                    sent_len_list = []
                    for i, sent in enumerate(sents):
                        # Every sentence after the first has a space separating it and the previous sentence
                        # so plus one is needed to account for this space since nltk set_tokenize removes extra spaces
                        if i > 0:
                            sent_len_sum += (1 + len(sent))
                        else:
                            sent_len_sum += len(sent)
                        sent_len_list.append(sent_len_sum)

                    triples = doc['triples']
                    # unique_triples = set()
                    for triple in triples:
                        queries.append((triple, text, sents, sent_len_list))
                        
                await utils.map_async(lambda q: parse_triple(args, *q, relations, rel_to_count, trex_set, common_vocab, tokenizer), queries, args.max_tasks, args.sleep_time)

            if args.threshold > 0:
                # Check if each relation has at least 1000 data points and remove relations that have 1000 samples from TREx relations
                for rel, count in rel_to_count.items():
                    if rel not in removed and count >= args.threshold:
                        print('Finished', rel)
                        relations.remove(rel)
                        removed.append(rel)

                # If all relations have at least 1000 samples, exit
                if is_complete(rel_to_count, args.threshold):
                    break

            elapsed = time.perf_counter() - start
            print('Parsing {} took {} seconds.'.format(filename, elapsed))
            print('Entity Map Size:', len(EID_TO_LABEL))

    print('Relations and their sample counts:', rel_to_count)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse original TREx data into JSONL files with facts and context sentences')
    parser.add_argument('in_dir', type=str, help='Directory containing the original TREx dataset (sets of documents)')
    parser.add_argument('out_dir', type=str, help='Directory to store JSONL files')
    parser.add_argument('--trex_test_dir', type=str, help='Path to TREx TEST set for all relation')
    parser.add_argument('--common_vocab_file', type=str, help='File containing common vocab subset')
    parser.add_argument('--threshold', type=int, default=-1, help='Minimum number of samples each relation should have at the end')
    parser.add_argument('--sleep_time', type=float, default=1e-4)
    parser.add_argument('--max_tasks', type=int, default=50)
    args = parser.parse_args()

    start = time.perf_counter()
    loop = asyncio.get_event_loop()
    loop.run_until_complete(main(args))
    # Zero-sleep to allow underlying connections to close
    loop.run_until_complete(asyncio.sleep(0))
    loop.close()
    elapsed = time.perf_counter() - start
    print('Total elapsed time: {} sec'.format(elapsed))
