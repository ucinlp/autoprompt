import argparse
import json
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm

import utils


logger = logging.getLogger(__name__)


class GradientStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_gradient = None
        module.register_backward_hook(self.hook)

    def hook(self, module, grad_in, grad_out):
        self._stored_gradient = grad_out[0]

    def get(self):
        return self._stored_gradient


def load_pretrained(model_name):
    """
    Loads pretrained HuggingFace config/model/tokenizer, as well as performs required
    initialization steps to facilitate working with triggers.
    """
    config = AutoConfig.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name)
    model.eval()
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    utils.add_task_specific_tokens(tokenizer)
    return config, model, tokenizer


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_embeddings(model, config):
    """Returns the wordpiece embedding module."""
    base_model = getattr(model, config.model_type)
    embeddings = base_model.embeddings.word_embeddings
    return embeddings


def hotflip_attack(averaged_grad,
                   embedding_matrix,
                   increase_loss=False,
                   num_candidates=1):
    """Returns the top candidate replacements."""
    with torch.no_grad():
        gradient_dot_embedding_matrix = torch.matmul(
            embedding_matrix,
            averaged_grad
        )
        if not increase_loss:
            gradient_dot_embedding_matrix *= -1
        _, top_k_ids = gradient_dot_embedding_matrix.topk(num_candidates)

    return top_k_ids


def replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask):
    """Replaces the trigger tokens in input_ids."""
    out = model_inputs.copy()
    input_ids = model_inputs['input_ids']
    trigger_ids = trigger_ids.repeat(trigger_mask.size(0), 1)
    filled = input_ids.masked_scatter(trigger_mask, trigger_ids)
    out['input_ids'] = filled
    return out


def get_loss(model, model_inputs, trigger_ids, label_token_ids):
    # Copy dict so pop operations don't have unwanted side-effects
    model_inputs = model_inputs.copy()
    trigger_mask = model_inputs.pop('trigger_mask')
    predict_mask = model_inputs.pop('predict_mask')
    model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
    logits, *_ = model(**model_inputs)
    predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
    return F.cross_entropy(predict_logits, label_token_ids)


def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name)
    model.to(device)
    templatizer = utils.TriggerTemplatizer(args.template, tokenizer)
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.encode(args.initial_trigger, add_special_tokens=False)
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)
    label_map = json.loads(args.label_map)

    logger.info('Loading datasets')
    collator = utils.TriggerCollator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_dataset(args.train, templatizer)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.load_dataset(args.dev, templatizer)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=False, collate_fn=collator)

    for i in range(args.iters):

        logger.info(f'Iteration: {i}')

        logger.info('Training')
        for model_inputs, labels in tqdm(train_loader):
            # Shuttle inputs to GPU
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}

            # TODO: Clean up. Biggest difficulty will be allowing label mapper to handle multiple
            # candidate labels if that is a desired function.
            label_tokens = [label_map[x] for x in labels]
            label_token_ids = tokenizer.convert_tokens_to_ids(label_tokens)
            label_token_ids = torch.tensor(label_token_ids, device=device)

            for token_to_flip in range(templatizer.num_trigger_tokens):
                model.zero_grad()
                loss = get_loss(model, model_inputs, trigger_ids, label_token_ids)
                loss.backward()

                # Perform hotflip attack
                grad = embedding_gradient.get()
                bsz, _, emb_dim = grad.size()
                selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
                grad = torch.masked_select(grad, selection_mask).view(bsz, -1, emb_dim)
                averaged_grad = grad.mean(dim=0)[token_to_flip]
                candidates = hotflip_attack(averaged_grad,
                                            embeddings.weight,
                                            increase_loss=False,
                                            num_candidates=args.num_cand)
                for candidate in candidates:
                    temp_trigger = trigger_ids.clone()
                    temp_trigger[:, token_to_flip] = candidate
                    with torch.no_grad():
                        new_loss = get_loss(model, model_inputs, temp_trigger, label_token_ids)
                    if new_loss < loss:
                        trigger_ids = temp_trigger
                        loss = new_loss

        logger.info('Evaluating')
        all_label_ids = tokenizer.convert_tokens_to_ids(list(label_map.values()))
        lookup = {label: i for i, label in enumerate(label_map.keys())}
        all_label_ids = torch.tensor(all_label_ids, device=device).unsqueeze(0)
        correct = 0
        total = 0
        for model_inputs, labels in tqdm(dev_loader):
            true = torch.tensor([lookup[l] for l in labels], device=device)
            # Shuttle inputs to GPU
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            model_inputs.pop('trigger_mask')  # Gotta do it
            predict_mask = model_inputs.pop('predict_mask')
            with torch.no_grad():
                logits, *_ = model(**model_inputs)
            predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
            predict_logits = predict_logits.gather(-1, all_label_ids.repeat(predict_logits.size(0), 1))
            predictions = predict_logits.argmax(-1)
            correct += (predictions == true).sum().float()
            total += predictions.size(0)
        dev_acc = correct / total

        logger.info(f'Trigger tokens: {tokenizer.decode(trigger_ids.squeeze(0).tolist())}')
        logger.info(f'Dev accuracy: {dev_acc}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, help='JSON object defining label map')

    parser.add_argument('--initial-trigger', type=str, default=None, help='Manual prompt')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--iters', type=int, default='100', help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--model-name', type=str, default='bert-base-cased', help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--use_ctx', action='store_true', help='Use context sentences for open-book probing')
    parser.add_argument('--patience', type=int, default=5)
    parser.add_argument('--num_cand', type=int, default=10)
    parser.add_argument('--sentence_size', type=int, default=50)

    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    run_model(args)
