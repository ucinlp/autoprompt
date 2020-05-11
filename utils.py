import os
import re
import json
import random
import asyncio
import numpy as np
import torch
from copy import deepcopy
from pytorch_transformers import BertTokenizer
import constants


async def map_async(fn, iterator, max_tasks=10, sleep_time=0.01):
    tasks = set()
    
    for x in iterator:
        if len(tasks) >= max_tasks:
            _, tasks = await asyncio.wait(tasks, return_when=asyncio.FIRST_COMPLETED)
        
        new_task = asyncio.ensure_future(fn(x))
        tasks.add(new_task)
        await asyncio.sleep(random.random() * sleep_time)

    await asyncio.wait(tasks)


def load_TREx_data(args, filename):
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased', do_lower_case=False)

    facts = []
    with open(filename, newline='') as f:
        lines = f.readlines()
        num_invalid_facts = 0
        for line in lines:
            sample = json.loads(line)
            sub = sample['sub_label']
            obj = sample['obj_label']

            # Skip facts with objects that consist of multiple tokens
            # TODO: different tokenizers split words differently...
            if len(tokenizer.tokenize(obj)) > 1:
                num_invalid_facts += 1
                continue

            # For conditional probing, skip facts that don't have context sentence
            if 'evidences' not in sample:
                num_invalid_facts += 1
                continue
            
            ##################################### CONDITIONAL PROBING #####################################
            evidences = sample['evidences']
            # To make the number of samples used between open-book and closed-book probe
            # settings, we need to only consider facts that include context sentences
            valid_contexts = []
            # For each evidence, replace [MASK] in the masked sentence with obj_surface. But the actual answer/object should be obj_label
            for evidence in evidences:
                ctx = evidence['masked_sentence']
                obj_surface = evidence['obj_surface']
                # Only consider context samples where object surface == true object label, and grab the first one
                if obj_surface == obj:
                    valid_contexts.append(ctx)
            # Randomly pick a context sentence that has obj_surface equal to the obj_label
            if not valid_contexts:
                # print('Invalid fact with no context - sub: {}, obj: {}'.format(sub, obj))
                num_invalid_facts += 1
            else:
                context = random.choice(valid_contexts)
                context_words = context.split()
                if len(context_words) > constants.MAX_CONTEXT_LEN:
                    # If context is too long, use the first X tokens (it's ok if obj isn't included)
                    context = ' '.join(context_words[:constants.MAX_CONTEXT_LEN])
                    # print('Sample context too long ({}), truncating.'.format(len(context_words)))
                context = context.replace(constants.MASK, obj_surface)
                facts.append((sub, obj, context))
            ###############################################################################################

            # Facts only consist of sub and obj for unconditional probing
            # facts.append((sub, obj))

        print('Total facts before:', len(lines))
        print('Invalid facts:', num_invalid_facts)
        print('Total facts after:', len(facts))

    return facts


def get_all_datasets(args):
    datasets = []

    train_file = os.path.join(args.data_dir, 'train.jsonl')
    train_data = load_TREx_data(args, train_file)
    print('Num samples in TREx train data:', len(train_data))

    # dev_file = os.path.join(args.data_dir, 'val.jsonl')
    dev_file = os.path.join(args.data_dir, 'dev.jsonl')
    dev_data = load_TREx_data(args, dev_file)
    print('Num samples in TREx dev data:', len(dev_data))

    datasets.append((train_data, dev_data))

    return datasets


def iterate_batches(inputs, batch_size, shuffle=False):
    """
    Split data into batches and return them as a generator
    """
    size = len(inputs)
    inputs = np.array(inputs)
    if shuffle:
        indices = np.arange(size)
        np.random.shuffle(indices)
    for start_idx in range(0, size, batch_size):
        end_idx = min(start_idx + batch_size, size)
        if shuffle:
            excerpt = indices[start_idx:end_idx]
        else:
            excerpt = slice(start_idx, end_idx)
        yield inputs[excerpt]


def make_batch(tokenizer, batch, trigger_tokens, prompt_format, use_ctx, cls_token, sep_token, mask_token, pad_token, period_token, device):
    """
    For BERT, [CLS] token marks the beginning of a sentence and [SEP] marks separation/end of sentences
    """
    source_tokens_batch = []
    target_tokens_batch = []
    trigger_mask_batch = []
    segment_ids_batch = []
    
    for sample in batch:
        # print('PROMPT:', build_prompt(tokenizer, sample, trigger_tokens))
        source_tokens = []
        target_tokens = []
        trigger_mask = []
        segment_ids = [] # used to distinguish different sentences

        if use_ctx:
            sub, obj, ctx = sample
            sub_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sub))
            obj_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))
            ctx_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(ctx))
        else:
            sub, obj = sample
            sub_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(sub))
            obj_tokens = tokenizer.convert_tokens_to_ids(tokenizer.tokenize(obj))

        trigger_idx = 0
        # Add CLS token at the beginning
        source_tokens.extend(cls_token)
        target_tokens.append(-1)
        trigger_mask.append(0)
        # Add context if probe setting is open-book (use context)
        if use_ctx:
            # From CLS token right before
            segment_ids.append(0)
            # Add context tokens
            source_tokens.extend(ctx_tokens)
            target_tokens.extend([-1] * len(ctx_tokens))
            trigger_mask.extend([0] * len(ctx_tokens))
            segment_ids.extend([0] * len(ctx_tokens))
            # Add SEP token to distinguish sentences
            source_tokens.extend(sep_token)
            target_tokens.append(-1)
            trigger_mask.append(0)
            segment_ids.append(0)
        
        # Keep track of length of first sentence only for conditional probing which uses two sentences
        first_sent_len = len(segment_ids)

        for part in prompt_format:
            if part == 'X':
                # Add subject
                source_tokens.extend(sub_tokens)
                target_tokens.extend([-1] * len(sub_tokens))
                trigger_mask.extend([0] * len(sub_tokens))
            elif part == 'Y':
                # Add MASKED object
                source_tokens.extend(mask_token)
                target_tokens.extend(obj_tokens)
                trigger_mask.extend([0] * len(obj_tokens))
            else:
                # Add triggers
                num_trigger_tokens = int(part)
                source_tokens.extend(trigger_tokens[trigger_idx:trigger_idx+num_trigger_tokens])
                target_tokens.extend([-1] * (num_trigger_tokens))
                trigger_mask.extend([1] * (num_trigger_tokens))
                # Update trigger idx
                trigger_idx += num_trigger_tokens

        # Add period at end of prompt
        source_tokens.extend(period_token)
        target_tokens.append(-1)
        trigger_mask.append(0)
        # Add SEP token at the end
        source_tokens.extend(sep_token)
        target_tokens.append(-1)
        trigger_mask.append(0)

        if use_ctx:
            segment_ids.extend([1] * (len(source_tokens) - first_sent_len))
        else:
            segment_ids.extend([0] * len(source_tokens))

        # Add encoded prompt to batch
        source_tokens_batch.append(torch.tensor(source_tokens))
        target_tokens_batch.append(torch.tensor(target_tokens))
        trigger_mask_batch.append(torch.tensor(trigger_mask))
        segment_ids_batch.append(torch.tensor(segment_ids))

    # Get max length sequence for padding
    seq_len = [s.size(0) for s in source_tokens_batch]
    max_len = np.max(seq_len)

    # Pad the batch
    source_tokens_batch = torch.nn.utils.rnn.pad_sequence(source_tokens_batch, batch_first=True, padding_value=pad_token[0])
    target_tokens_batch = torch.nn.utils.rnn.pad_sequence(target_tokens_batch, batch_first=True, padding_value=-1)
    trigger_mask_batch = torch.nn.utils.rnn.pad_sequence(trigger_mask_batch, batch_first=True)
    segment_ids_batch = torch.nn.utils.rnn.pad_sequence(segment_ids_batch, batch_first=True, padding_value=pad_token[0])

    # Move to GPU
    source_tokens_batch = source_tokens_batch.to(device)
    target_tokens_batch = target_tokens_batch.to(device)
    trigger_mask_batch = trigger_mask_batch.to(device)
    segment_ids_batch = segment_ids_batch.to(device)

    return source_tokens_batch, target_tokens_batch, trigger_mask_batch, segment_ids_batch


def get_unique_objects(data):
    objs = set()
    for sample in data:
        # TODO: make this switch between ctx and no ctx dynamic
        # sub, obj = sample
        sub, obj, ctx = sample
        # print('sub: {}, obj: {}, ctx: {}'.format(sub, obj, ctx))
        objs.add(obj)
    return list(objs)


def load_vocab(vocab_filename):
    with open(vocab_filename, "r") as f:
        lines = f.readlines()
    vocab = [x.strip() for x in lines]
    return vocab


def get_id_from_url(url):
    """
    Extract Wikidata entity id from URL
    """
    return url.split('/')[-1]


def replace_nth(s, sub, rep, n):
    """
    Replace the Nth occurence of a substring with specified replacement string
    """
    where = [m.start() for m in re.finditer(sub, s)][n-1]
    before = s[:where]
    after = s[where:]
    after = after.replace(sub, rep, 1)
    return before + after
