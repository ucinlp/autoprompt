import time
import json
import random
import argparse
import logging
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer
from tqdm import tqdm

import lmat.utils as utils


logger = logging.getLogger(__name__)
# TODO: remove when not running script that applies trigger search to all relations
# logger.disabled = True


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


class PredictWrapper:
    """
    PyTorch transformers model wrapper. Handles necc. preprocessing of inputs for triggers
    experiments.
    """
    def __init__(self, model):
        self._model = model

    def __call__(self, model_inputs, trigger_ids):
        # Copy dict so pop operations don't have unwanted side-effects
        model_inputs = model_inputs.copy()
        trigger_mask = model_inputs.pop('trigger_mask')
        predict_mask = model_inputs.pop('predict_mask')
        model_inputs = replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
        logits, *_ = self._model(**model_inputs)
        predict_logits = logits.masked_select(predict_mask.unsqueeze(-1)).view(logits.size(0), -1)
        return predict_logits


class AccuracyFn:
    """
    Computing the accuracy when a label is mapped to multiple tokens is difficult in the current
    framework, since the data generator only gives us the token ids. To get around this we
    compare the target logp to the logp of all labels. If target logp is greater than all (but)
    one of the label logps we know we are accurate.
    """
    def __init__(self, tokenizer, label_map, device):
        self._all_label_ids = []
        self._pred_to_label = []
        for label, label_tokens in label_map.items():
            self._all_label_ids.append(utils.encode_label(tokenizer, label_tokens).to(device))
            self._pred_to_label.append(label)

    def __call__(self, predict_logits, gold_label_ids):
        # Get total log-probability for the true label
        gold_logp = get_loss(predict_logits, gold_label_ids)

        # Get total log-probability for all labels
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]

        # Add up the number of entries where loss is greater than or equal to gold_logp.
        ge_count = all_label_logp.le(gold_logp.unsqueeze(-1)).sum(-1)
        correct = ge_count.le(1)  # less than in case of num. prec. issues

        return correct.float()

    # TODO: @rloganiv - This is hacky. Replace with something sensible.
    def predict(self, predict_logits):
        bsz = predict_logits.size(0)
        all_label_logp = []
        for label_ids in self._all_label_ids:
            label_logp = get_loss(predict_logits, label_ids.repeat(bsz, 1))
            all_label_logp.append(label_logp)
        all_label_logp = torch.stack(all_label_logp, dim=-1)
        _, predictions = all_label_logp.max(dim=-1)
        predictions = [self._pred_to_label[x] for x in predictions.tolist()]
        return predictions


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


def get_loss(predict_logits, label_ids):
    predict_logp = F.log_softmax(predict_logits, dim=-1)
    target_logp = predict_logp.gather(-1, label_ids)
    target_logp = target_logp - 1e32 * label_ids.eq(0)  # Apply mask
    target_logp = torch.logsumexp(target_logp, dim=-1)
    return -target_logp


def run_model(args):
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    config, model, tokenizer = load_pretrained(args.model_name)
    model.to(device)
    embeddings = get_embeddings(model, config)
    embedding_gradient = GradientStorage(embeddings)
    predictor = PredictWrapper(model)

    # Special tokens that will be filtered out when selecting the top candidates for each token flip (only for fact retrieval and relation extraction)
    special_token_ids = [
        tokenizer.cls_token_id,
        tokenizer.unk_token_id,
        tokenizer.sep_token_id,
        tokenizer.mask_token_id,
        tokenizer.pad_token_id
    ]

    # Fact retrieval and relation extraction don't need label map
    label_map = None
    if args.label_map:
        label_map = json.loads(args.label_map)
    logger.info(f"Label map: {label_map}")
    templatizer = utils.TriggerTemplatizer(
        args.template,
        tokenizer,
        label_map=label_map,
        label_field=args.label_field,
        add_special_tokens=False
    )

    # Obtain the initial trigger tokens and label mapping
    if args.initial_trigger:
        trigger_ids = tokenizer.convert_tokens_to_ids(args.initial_trigger)
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)

    # NOTE: Accuracy can only be computed if a fixed pool of labels is given, which currently
    # requires the label map to be specified. Since producing a label map may be cumbersome (e.g.,
    # for link prediction tasks), we just use loss as the evaluation metric in these cases.
    if label_map:
        evaluation_fn = AccuracyFn(tokenizer, label_map, device)
    else:
        evaluation_fn = get_loss

    # TODO: @rloganiv - Something a little cleaner.
    filter_candidates = set()
    if label_map:
        for label_tokens in label_map.values():
            filter_candidates.update(
                utils.encode_label(tokenizer, label_tokens).tolist().pop()
            )
    logger.info(f'Filter candidates: {filter_candidates}')

    logger.info('Loading datasets')
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(args.train, templatizer, limit=args.limit)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.load_trigger_dataset(args.dev, templatizer)
    dev_loader = DataLoader(dev_dataset, batch_size=args.eval_size, shuffle=False, collate_fn=collator)

    logger.info('Getting all unique objects')
    # Get all unique objects from train data and check if hotflip candidates == object later on
    unique_objects = utils.get_unique_objects(args.train)

    logger.info('Evaluating')
    numerator = 0
    denominator = 0
    for model_inputs, labels in tqdm(dev_loader):
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        with torch.no_grad():
            predict_logits = predictor(model_inputs, trigger_ids)
        numerator += evaluation_fn(predict_logits, labels).sum().item()
        denominator += labels.size(0)
    dev_metric = numerator / (denominator + 1e-13)
    logger.info(f'Dev metric: {dev_metric}')

    # Measure elapsed time of trigger search
    start = time.time()

    # NOTE: For sentiment analysis and NLI, we want to maximize accuracy. But for fact retrieval and relation extraction, we want to minimize loss.
    if label_map:
        best_dev_metric = 0
    else:
        best_dev_metric = 999999

    # TODO: consider adding early stopping
    for i in range(args.iters):
        logger.info(f'Iteration: {i}')

        logger.info('Accumulating Gradient')
        model.zero_grad()

        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)
        averaged_grad = None

        # Accumulate
        for step in pbar:
            # Shuttle inputs to GPU
            try:
                model_inputs, labels = next(train_iter)
            except StopIteration:
                logger.info('No more training data, skipping to next step')
                break
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            predict_logits = predictor(model_inputs, trigger_ids)
            loss = get_loss(predict_logits, labels).mean()
            loss.backward()

            grad = embedding_gradient.get()
            bsz, _, emb_dim = grad.size()
            selection_mask = model_inputs['trigger_mask'].unsqueeze(-1)
            grad = torch.masked_select(grad, selection_mask)
            grad = grad.view(bsz, templatizer.num_trigger_tokens, emb_dim)

            if averaged_grad is None:
                averaged_grad = grad.sum(dim=0) / args.accumulation_steps
            else:
                averaged_grad += grad.sum(dim=0) / args.accumulation_steps

        logger.info('Evaluating Candidates')
        pbar = tqdm(range(args.accumulation_steps))
        train_iter = iter(train_loader)

        # Sentiment analysis and NLI use random token flipping while fact retrieval and relation extraction use sequential token flipping
        if label_map:
            token_to_flip_list = [random.randrange(templatizer.num_trigger_tokens)]
        else:
            token_to_flip_list = list(range(templatizer.num_trigger_tokens))

        for token_to_flip in token_to_flip_list:
            candidates = hotflip_attack(averaged_grad[token_to_flip],
                                        embeddings.weight,
                                        increase_loss=False,
                                        num_candidates=args.num_cand)
            # print('CANDIDATES:', tokenizer.convert_ids_to_tokens(candidates))

            # Filter candidates if task is fact retrieval or relation extraction
            # TODO: handle edge case where ALL candidates are filtered out
            # TODO: follow the footsteps of filter_candidates logic on line 356
            if not label_map:
                filtered_candidates = []
                for cand in candidates:
                    # print('CAND:', cand, tokenizer.decode([cand]))
                    # Make sure to exclude special tokens like [CLS] from candidates
                    if cand in special_token_ids:
                        # logger.info("Skipping candidate {} because it's a special symbol: {}".format(cand, tokenizer.decode([cand])))
                        continue

                    # Make sure object/answer token is not included in the trigger -> prevents biased/overfitted triggers for each relation
                    if tokenizer.decode([cand]).strip().lower() in unique_objects:
                        # logger.info("Skipping candidate {} because it's the same as a gold object: {}".format(cand, tokenizer.decode([cand])))
                        continue

                    # Ignore capitalized word pieces and hopefully this heuristic will deal with proper nouns like Antarctica and ABC
                    # if any(c.isupper() for c in tokenizer.decode([cand])):
                    #     logger.info("Skipping candidate {} because it's probably a proper noun: {}".format(cand, tokenizer.decode([cand])))
                    #     continue
                    
                    filtered_candidates.append(cand)

                # Update candidates
                candidates = filtered_candidates[:]

            current_score = 0
            # candidate_scores = torch.zeros(args.num_cand, device=device)
            candidate_scores = torch.zeros(len(candidates), device=device)
            denom = 0
            for step in pbar:
                try:
                    model_inputs, labels = next(train_iter)
                except StopIteration:
                    logger.info('No more training data, skipping to next step')
                    break
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                with torch.no_grad():
                    predict_logits = predictor(model_inputs, trigger_ids)
                    eval_metric = evaluation_fn(predict_logits, labels)

                # Update current score
                current_score += eval_metric.sum()
                denom += labels.size(0)

                # NOTE: Instead of iterating over tokens to flip we randomly change just one each
                # time so the gradients don't get stale.
                for i, candidate in enumerate(candidates):

                    if candidate.item() in filter_candidates:
                        candidate_scores[i] = -1e32
                        continue

                    temp_trigger = trigger_ids.clone()
                    temp_trigger[:, token_to_flip] = candidate
                    with torch.no_grad():
                        predict_logits = predictor(model_inputs, temp_trigger)
                        eval_metric = evaluation_fn(predict_logits, labels)

                    candidate_scores[i] += eval_metric.sum()

            # TODO: make this block of code more elegant and concise
            if label_map:
                if (candidate_scores > current_score).any():
                    logger.info('Better trigger detected.')
                    best_candidate_score = candidate_scores.max()
                    best_candidate_idx = candidate_scores.argmax()
                    trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    logger.info(f'Train Accuracy: {best_candidate_score / (denom + 1e-13): 0.4f}')
                else:
                    logger.info('No improvement detected. Skipping evaluation.')
                    continue
            else:
                if (candidate_scores < current_score).any():
                    logger.info('Better trigger detected.')
                    best_candidate_score = candidate_scores.min()
                    best_candidate_idx = candidate_scores.argmin()
                    trigger_ids[:, token_to_flip] = candidates[best_candidate_idx]
                    logger.info(f'Train Accuracy: {best_candidate_score / (denom + 1e-13): 0.4f}')
                else:
                    logger.info('No improvement detected. Skipping evaluation.')
                    continue

        logger.info('Evaluating')
        numerator = 0
        denominator = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            with torch.no_grad():
                predict_logits = predictor(model_inputs, trigger_ids)
            numerator += evaluation_fn(predict_logits, labels).sum().item()
            denominator += labels.size(0)
        dev_metric = numerator / (denominator + 1e-13)

        logger.info(f'Trigger tokens: {tokenizer.convert_ids_to_tokens(trigger_ids.squeeze(0))}')
        logger.info(f'Dev metric: {dev_metric}')

        if (label_map and dev_metric > best_dev_metric) or (not label_map and dev_metric < best_dev_metric):
            logger.info('Best performance so far')
            best_trigger_ids = trigger_ids.clone()
            best_dev_metric = dev_metric

    logger.info(f'Best tokens: {tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))}')
    logger.info(f'Best dev metric: {best_dev_metric}')
    print(args.train)
    print(f'Best tokens: {tokenizer.convert_ids_to_tokens(best_trigger_ids.squeeze(0))}')
    print(f'Best dev metric: {best_dev_metric}')

    # TODO: print out trigger friendly format (i.e. handling ##subwords, combinging the trigger tokens together)

    # Measure elapsed time
    end = time.time()
    print('Elapsed time: {} min\n'.format(round((end - start) / 60, 4)))



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--dev', type=Path, required=True, help='Dev data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, help='JSON object defining label map')

    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field. For fact retrieval, it should be obj_label.')

    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--eval-size', type=int, default=256, help='Eval size')
    parser.add_argument('--iters', type=int, default=100,
                        help='Number of iterations to run trigger search algorithm')
    parser.add_argument('--accumulation-steps', type=int, default=10)
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--use_ctx', action='store_true',
                        help='Use context sentences for open-book probing')
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
