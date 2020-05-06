import os
import random
import argparse
import numpy as np
import heapq
import torch
import spacy
import matplotlib.pyplot as plt
from copy import deepcopy
from operator import itemgetter
from my_bert_model import MyBertForMaskedLM
from transformers import glue_processors as processors
#from transformers import BertTokenizer, BertForMaskedLM
from pytorch_transformers import BertTokenizer, BertForMaskedLM
import constants
import utils
import lama_utils
import time


nlp = spacy.load("en_core_web_sm")


# Returns the wordpiece embedding weight matrix
def get_embedding_weight(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == constants.BERT_EMB_DIM: # only add a hook to wordpiece embeddings, not position embeddings
                return module.weight.detach()


# Add hooks for embeddings
extracted_grads = []
def extract_grad_hook(module, grad_in, grad_out):
    extracted_grads.append(grad_out[0])

def add_hooks(language_model):
    for module in language_model.modules():
        if isinstance(module, torch.nn.Embedding):
            if module.weight.shape[0] == constants.BERT_EMB_DIM: # only add a hook to wordpiece embeddings, not position
                module.weight.requires_grad = True
                module.register_backward_hook(extract_grad_hook)


def get_cand(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, device, true_labels, class_labels_ids):
    index = (target_tokens[0] != -1)
    batch_size = source_tokens.size()[0]
    trigger_tokens = torch.tensor(trigger_tokens, device=device).repeat(batch_size, 1)
    # Make sure to not modify the original source tokens
    src = source_tokens.clone()
    src = src.masked_scatter_(trigger_mask.to(torch.uint8), trigger_tokens).to(device)
    dst = target_tokens.to(device)
    model.eval()
    outputs = model(src, masked_lm_labels=dst, token_type_ids=segment_ids)
    loss, pred_scores = outputs[:2]
    class_pred_scores = []
    max_score = -1
    pred_label = -1
    for i, label_id in enumerate(class_labels_ids):
        if pred_scores[:,index,label_id].item() > max_score:
            max_score = pred_scores[:,index,label_id].item()
            pred_label = class_labels_ids[i]
    res = 0
    if (pred_label == true_labels[0][0]):
        res = 1
    return res


def run_eval(args):
    np.random.seed(0)
    torch.random.manual_seed(0)
    torch.cuda.manual_seed(0)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(device)
    class_labels = args.class_labels.split('-')
    masked_words = args.masked_words.split('-')
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    model = MyBertForMaskedLM.from_pretrained('bert-base-cased')
    model.eval()
    model.to(device)

    add_hooks(model) # add gradient hooks to embeddings
    total_vocab_size = constants.BERT_EMB_DIM  # total number of subword pieces in the specified model

    eval_data = get_eval_data(args, class_labels, masked_words)

    # Get all unique objects from train data
    unique_objects = utils.get_unique_objects(eval_data)
    # Store token ids for each object in batch to check if candidate == object later on
    obj_token_ids = tokenizer.convert_tokens_to_ids(unique_objects)

    # TODO: make this dynamic to work for other datasets. Make special symbols dictionary
    # Initialize special tokens
    cls_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(constants.BERT_CLS))
    unk_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(constants.BERT_UNK))
    sep_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(constants.BERT_SEP))
    mask_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(constants.MASK))
    pad_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(constants.BERT_PAD))
    period_token = tokenizer.convert_tokens_to_ids(tokenizer.tokenize('.'))

    # constants.SPECIAL_SYMBOLS
    special_token_ids = [cls_token, unk_token, sep_token, mask_token, pad_token]


    prompt_format = args.format.split('-')
    trigger_token_length = sum([int(x) for x in prompt_format if x.isdigit()])
    if args.manual:
        print('Trigger initialization: MANUAL')
        init_trigger = args.manual
        init_tokens = tokenizer.tokenize(init_trigger)
        trigger_token_length = len(init_tokens)
        trigger_tokens = tokenizer.convert_tokens_to_ids(init_tokens)

    all_count = 0
    true_count = 0
    for batch in utils.iterate_batches(eval_data, args.bsz, True):
        all_count = all_count +1
        # YAS source_tokens, target_tokens, trigger_mask, segment_ids = utils.make_batch(tokenizer, batch, trigger_tokens, prompt_format, args.use_ctx, cls_token, sep_token, mask_token, pad_token, period_token, device)
        source_tokens, target_tokens, trigger_mask, segment_ids, labels = utils.make_batch_glue(tokenizer, batch, trigger_tokens, prompt_format, args.use_ctx, cls_token, sep_token, mask_token, pad_token, period_token, device)
        result = get_cand(model, source_tokens, target_tokens, trigger_tokens, trigger_mask, segment_ids, device, labels, obj_token_ids)
        true_count = true_count + result
    print("accuracy", true_count/all_count)


def get_eval_data(args, class_labels, masked_words):

    # dev_file = os.path.join(args.data_dir, 'val.jsonl')
    dev_file = os.path.join(args.data_dir, 'dev.jsonl')
    # dev_data = load_TREx_data(args, dev_file)
    dev_data = load_GLUE_data(args.data_dir, dev_file , False, glue_name = args.dataset , down_sample = False, sentence_size = args.sentence_size, class_labels=class_labels, masked_words=masked_words)
    print('Num samples in dev data:', len(dev_data))

    return dev_data


def load_GLUE_data(args, filename, is_train, glue_name, sentence_size, class_labels, masked_words, down_sample = False):
    facts = []
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)
    processor = processors[glue_name.lower()]()
    print("hahahah", class_labels)
    print("beeee", masked_words)
    #TOOD: make this filepath as input
    #/home/yrazeghi/data
    if is_train:
        data = processor.get_train_examples(args+glue_name)
    else:
        data = processor.get_dev_examples(args+glue_name)
    for d in data:
        label = d.label
        if label in class_labels: #todo change this
            ind = class_labels.index(label)
            premiss = d.text_a
            premiss = premiss[:-1]
            hypothesis = d.text_b
            hypothesis = hypothesis[:-1]
            sub = premiss + " *%* " + hypothesis
            # sub = "pick a context sentence that has obj_surface equal equal equal equal "
            # print("label:::", label)
            # print("word::::", )
            obj = masked_words[ind]
            # print("word::::", obj)
            if len(tokenizer.tokenize(sub)) > sentence_size:
                continue
            if down_sample:
                r_rand = random.uniform(0, 1)
                if r_rand < 0.005:
                    facts.append((sub, obj))
            else:
                facts.append((sub, obj))
        # print('Total facts before:', len(lines))
        # print('Invalid facts:', num_invalid_facts)
    print('Total facts after:', len(facts))
    return facts



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('data_dir', type=str)
    parser.add_argument('--lm', type=str, default='bert')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--format', type=str, default='X--5-Y', help='Prompt format')
    parser.add_argument('--use_ctx', action='store_true', help='Use context sentences for open-book probing')
    parser.add_argument('--manual', type=str, help='Manual prompt')
    parser.add_argument('--class_count', type=int, default=2, help='number of classes')
    parser.add_argument('--masked_words', type=str, default="and-but", help='mask words')
    parser.add_argument('--class_labels', type=str, default="entailment-contradiction", help='class labels')
    parser.add_argument('--dataset', type=str, default="MNLI")
    parser.add_argument('--sentence_size', type=int, default=50)
    args = parser.parse_args()
    run_eval(args)
