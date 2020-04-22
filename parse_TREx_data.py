import os
import json
import argparse
from tqdm import tqdm
from nltk import tokenize
from pytorch_transformers import BertTokenizer
import utils
import constants

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


def parse_doc_set(args, in_file, relations, rel_to_count, trex_set, common_vocab, tokenizer, unique_triples_global):
    with open(in_file, 'r') as f_in:
        docs = json.load(f_in)
        # doc_count = len(docs)
        for d, doc in enumerate(tqdm(docs)):
            """
            text = doc['text']
            sents = tokenize.sent_tokenize(text)

            len_sum = 0
            len_list = []
            # TODO: account for spaces before new sentences like "...President of France. Andorra is the..." nltk set_tokenize removes extra spaces
            for sent in sents:
                # plus one is needed because end boundary of a word is exclusive
                len_sum += (len(sent) + 1)
                len_list.append(len_sum)
            """

            triples = doc['triples']
            sub_id_to_label = {}
            obj_id_to_label = {}
            unique_triples = set()
            # print('Extracting facts for doc {}/{}'.format(d+1, doc_count))
            for triple in triples:
                pred_dict = triple['predicate']
                obj_dict = triple['object']
                sub_dict = triple['subject']

                """
                obj_start = obj['boundaries'][0] # inclusive
                obj_end = obj['boundaries'][1] # exclusive

                obj_sent = ''
                for l, sent in zip(len_list, sents):
                    if obj_end <= l:
                        obj_sent = sent
                        break
                
                text[obj_start:obj_end]
                """

                pred_id = utils.get_id_from_url(pred_dict['uri'])
                obj_id = utils.get_id_from_url(obj_dict['uri']) # some object IDs turn out to be "XMLSchema#dateTime" which is invalid so skip
                obj_label = obj_dict['surfaceform']
                sub_id = utils.get_id_from_url(sub_dict['uri'])
                sub_label = sub_dict['surfaceform']
                # If a fact's annotator is Simple_Coreference, the sub or obj can be something like "It"
                sub_annot = sub_dict['annotator']
                obj_annot = obj_dict['annotator']
                simp = 'Simple_Coreference'
                tri = (sub_id, pred_id, obj_id)

                if tri not in unique_triples_global and pred_id in relations and sub_id.startswith('Q') and pred_id.startswith('P') and obj_id.startswith('Q') and sub_annot != simp and obj_annot != simp:
                    unique_triples.add(tri)
                    sub_id_to_label[sub_id] = sub_label
                    obj_id_to_label[obj_id] = obj_label

            # Integrate current unique triples into global set
            unique_triples_global = unique_triples | unique_triples_global

            # Filter out facts in TREx-test, multi-token objects, and objects not in common vocab
            # print('Filtering facts')
            for triple in list(unique_triples):
                sub_id, pred_id, obj_id = triple
                sub_label = sub_id_to_label[sub_id]
                obj_label = obj_id_to_label[obj_id]

                # First, make sure fact is not in TREx test set
                if triple in trex_set:
                    continue

                # Make sure object is a single token
                if len(tokenizer.tokenize(obj_label)) != 1:
                    continue

                # Make sure object is in common vocab subset
                if obj_label not in common_vocab:
                    continue

                # Finally write fact to file if it passes all criteria
                filepath = os.path.join(args.out_dir, pred_id + '.jsonl')
                with open(filepath, 'a+') as f_out:
                    # Update rel_to_count dict
                    rel_to_count[pred_id] += 1
                    f_out.write(json.dumps({'sub_uri': sub_id, 'obj_uri': obj_id, 'sub_label': sub_label, 'obj_label': obj_label}) + '\n')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Parse original TREx data')
    # parser.add_argument('in_file', type=str, help='JSON file containing parts of original TREx dataset')
    parser.add_argument('in_dir', type=str, help='Directory containing the original TREx dataset (sets of documents)')
    parser.add_argument('out_dir', type=str, help='Directory to store JSONL files')
    parser.add_argument('--trex_test_dir', type=str, help='Path to TREx TEST set for all relation')
    parser.add_argument('--common_vocab_file', type=str, help='File containing common vocab subset')
    parser.add_argument('--threshold', type=int, default=1000, help='Minimum number of samples each relation should have at the end')
    args = parser.parse_args()

    # TREx relations
    relations = [
        'P17', 'P19', 'P20', 'P27', 'P30', 'P31', 'P36', 'P37', 'P39', 'P47',
        'P101', 'P103', 'P106', 'P108', 'P127', 'P131', 'P136', 'P138', 'P140', 'P159',
        'P176', 'P178', 'P190', 'P264', 'P276', 'P279', 'P361', 'P364', 'P407', 'P413',
        'P449', 'P463', 'P495', 'P527', 'P530', 'P740', 'P937', 'P1001', 'P1303', 'P1376', 'P1412'
    ]

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

    # This is so we don't get duplicate facts as we go through the TREx dataset
    unique_triples_global = set()

    # Download the full TREx dataset here https://hadyelsahar.github.io/t-rex/downloads/
    for filename in os.listdir(args.in_dir):
        filename = os.fsdecode(filename)
        print('Parsing', filename)
        if filename.endswith('.json'):
            filepath = os.path.join(args.in_dir, filename)
            parse_doc_set(args, filepath, relations, rel_to_count, trex_set, common_vocab, tokenizer, unique_triples_global)

            # Check if each relation has at least 1000 data points and remove relations that have 1000 samples from TREx relations
            for rel, count in rel_to_count.items():
                if rel not in removed and count >= args.threshold:
                    print('Finished', rel)
                    relations.remove(rel)
                    removed.append(rel)

            # If all relations have at least 1000 samples, exit
            if is_complete(rel_to_count, args.threshold):
                break

    print('Relations and their sample counts:', rel_to_count)
