import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

import autoprompt.utils as utils
from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


class GenerativeEvaluator:
    """Used for generative evaluation."""
    def __init__(
            self,
            model,
            tokenizer,
            decoding_strategy,
            **kwargs
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._decoding_strategy = decoding_strategy
        
    def __call__(self, model_inputs, labels, train=True):
        if train:
            predict_mask = model_inputs['predict_mask']
            loss, logits, *_ = forward_w_triggers(self._model, model_inputs, labels)
            preds = torch.zeros_like(model_inputs['input_ids'])
            preds = preds.masked_fill(model_inputs['attention_mask'].bool(), -100)
            preds[predict_mask] = logits.argmax(dim=-1)[predict_mask]
        else:
            loss = torch.tensor(0.0, device=labels.device)
            preds = decode(self._model, model_inputs, decoding_strategy=self._decoding_strategy)
        correct = (preds == labels).all(dim=-1).sum()
        return loss, correct, preds


class ClassificationEvaluator:
    """Used for evaluating classifiers (e.g., tasks w/ fixed label pools)."""
    def __init__(
        self,
        model,
        tokenizer,
        label_map,
        **kwargs
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._label_map = label_map
        label_tokens = self._tokenizer(
            list(label_map.values()),
            add_special_tokens=False,
            return_tensors='pt',
        )['input_ids']
        if label_tokens.size(1) != 1:
            raise ValueError(
                'Multi-token labels not supported for multiple choice evaluation'
            )
        self._label_tokens = label_tokens.view(1, -1)

    def __call__(self, model_inputs, labels, train=True):

        # Ensure everything is on the same device
        label_tokens = self._label_tokens.to(labels.device)

        # Get predictions
        predict_mask = model_inputs['predict_mask']
        labels = labels[predict_mask].unsqueeze(-1)
        logits, *_ = forward_w_triggers(self._model, model_inputs)
        predict_logits = torch.gather(
            logits[predict_mask],
            dim=-1,
            index=label_tokens.repeat(labels.size(0), 1)
        )
        preds = predict_logits.argmax(dim=-1, keepdims=True)

        # Convert label tokens to their indices in the label map.
        _, label_inds = torch.where(labels.eq(label_tokens))
        label_inds = label_inds.unsqueeze(-1)

        # Get loss
        predict_logp = F.log_softmax(predict_logits, dim=-1)
        loss = -predict_logp.gather(-1, label_inds).mean()

        # Get evaluation score
        correct = preds.eq(label_inds).sum()

        return loss, correct, preds


EVALUATORS = {
    'generative': GenerativeEvaluator,
    'classification': ClassificationEvaluator,
}


def forward_w_triggers(model, model_inputs, labels=None):
    """
    Run model forward w/ preprocessing for continuous triggers.

    Parameters
    ==========
    model : transformers.PretrainedModel
        The model to use for predictions.
    model_inputs : Dict[str, torch.LongTensor]
        The model inputs.
    labels : torch.LongTensor
        (optional) Tensor of labels. Loss will be returned if provided.
    """
    # Ensure destructive pop operations are only limited to this function.
    model_inputs = model_inputs.copy()
    trigger_mask = model_inputs.pop('trigger_mask')
    predict_mask = model_inputs.pop('predict_mask')
    input_ids = model_inputs.pop('input_ids')

    # Get embeddings of input sequence
    batch_size = input_ids.size(0)
    inputs_embeds = model.embeds(input_ids)
    inputs_embeds[trigger_mask] = model.relation_embeds.repeat((batch_size, 1))
    model_inputs['inputs_embeds'] = inputs_embeds
    
    return model(**model_inputs, labels=labels)


def decode(model, model_inputs, decoding_strategy="iterative"):
    """
    Decode from model.

    Parameters
    ==========
    model : transformers.PretrainedModel
        The model to use for predictions.
    model_inputs : Dict[str, torch.LongTensor]
        The model inputs.
    decoding_strategy : str
        The decoding strategy. One of: parallel, monotonic, iterative.
        * parallel: all predictions made at the same time.
        * monotonic: predictions decoded from left to right.
        * iterative: predictions decoded in order of highest probability.
    """
    assert decoding_strategy in ['parallel', 'monotonic', 'iterative']

    # initialize output to ignore label.
    output = torch.zeros_like(model_inputs['input_ids'])
    output = output.masked_fill(model_inputs['attention_mask'].bool(), -100)

    if decoding_strategy == 'parallel':
        # Simple argmax over arguments.
        predict_mask = model_inputs['predict_mask']
        logits, *_ = forward_w_triggers(model, model_inputs)
        preds = logits.argmax(dim=-1)
        output[predict_mask] = preds[predict_mask]

    elif decoding_strategy == 'monotonic':
        predict_mask = model_inputs['predict_mask'].clone()
        idx0 = torch.arange(predict_mask.size(0))
        input_ids = model_inputs['input_ids']
        iterations = predict_mask.sum(dim=-1).max().item()
        for i in range(iterations):
            logits, *_ = forward_w_triggers(model, model_inputs)
            row_mask = predict_mask.any(dim=-1)
            idx1 = torch.argmax(predict_mask.long(), dim=-1)
            combined_mask = torch.zeros_like(predict_mask)
            combined_mask[idx0, idx1] = row_mask
            pred = logits[combined_mask].argmax(dim=-1)
            input_ids[combined_mask] = pred
            output[combined_mask] = pred
            predict_mask[combined_mask] = False

    elif decoding_strategy == 'iterative':
        predict_mask = model_inputs['predict_mask'].clone()
        idx0 = torch.arange(predict_mask.size(0))
        input_ids = model_inputs['input_ids']
        iterations = predict_mask.sum().item()
        for i in range(iterations):
            # NOTE: We're going to be lazy and make the search for the most
            # likely prediction easier by setting the logits for any tokens
            # other than the candidates to a huge negative number.
            logits, *_ = forward_w_triggers(model, model_inputs)
            logits[~predict_mask] = -1e32
            top_scores, preds = torch.max(logits, dim=-1)
            row_mask = predict_mask.any(dim=-1)
            idx1 = torch.argmax(top_scores, dim=-1)
            combined_mask = torch.zeros_like(predict_mask)
            combined_mask[idx0, idx1] = row_mask
            pred = preds[combined_mask]
            input_ids[combined_mask] = pred
            output[combined_mask] = pred
            predict_mask[combined_mask] = False
    else:
        raise ValueError(
            'Something is really wrong with the control flow in this function'
        )

    return output


def main(args):
    if args.evaluation_strategy == 'exact-match':
        assert args.decoding_strategy is not None
    elif args.evaluation_strategy == 'multiple-choice':
        assert args.label_map is not None

    utils.set_seed(args.seed)

    # Handle multi-GPU setup
    world_size = os.getenv('WORLD_SIZE', -1)
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
        if world_size != -1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=args.local_rank,
                world_size=world_size,
            )
    is_main_process = args.local_rank in [-1, 0] or world_size == -1
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO if is_main_process else logging.WARN)
    logger.warning('Rank: %s - World Size: %s', args.local_rank, world_size)

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )

    # Load & preprocess trigger template and data.
    if args.label_map is not None:
        label_map = json.loads(args.label_map)
    else:
        label_map = None
    templatizer = utils.MultiTokenTemplatizer(
        template=args.template,
        tokenizer=tokenizer,
        label_field=args.label_field,
        label_map=label_map,
        add_padding=args.add_padding,
    )
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(
        args.train,
        templatizer=templatizer,
        train=True,
        preprocessor_key=args.preprocessor,
        limit=args.limit,
    )
    if world_size == -1:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=collator, sampler=train_sampler)
    dev_dataset = utils.load_trigger_dataset(
        args.dev,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
        limit=args.limit,
    )
    if world_size == -1:
        dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)
    else:
        dev_sampler = torch.utils.data.DistributedSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, collate_fn=collator, sampler=dev_sampler)
    test_dataset = utils.load_trigger_dataset(
        args.test,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
    )
    if world_size == -1:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    else:
        test_sampler = torch.utils.data.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    # Setup model
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config)
    model.embeds = utils.get_word_embeddings(model)
    model.lm_head = utils.get_lm_head(model)
    model.relation_embeds = torch.nn.Parameter(
        torch.randn(
            templatizer.num_trigger_tokens,
            model.embeds.weight.size(1),
            requires_grad=True,
        ), 
    )
    if args.initial_trigger is not None:
        logger.info('Overwriting embedding weights using initial trigger.')
        initial_trigger_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(args.initial_trigger)
        )
        if args.debug:
            detokenized = tokenizer.convert_ids_to_tokens(initial_trigger_ids)
            logger.debug(f'Initial trigger (detokenized): {detokenized}')
        initial_trigger_embeds = model.embeds(initial_trigger_ids)
        model.relation_embeds.data.copy_(initial_trigger_embeds)
        assert torch.equal(model.relation_embeds.data, initial_trigger_embeds)
    model.to(device)

    ckpt_path = args.ckpt_dir / 'pytorch_model.bin'
    if ckpt_path.exists():
        logger.info('Restoring checkpoint.')
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)

    # Setup optimizer
    if args.finetune_mode == 'all-but-trigger':
        params = []
    else:
        params = [{'params': [model.relation_embeds]}]
    if args.finetune_mode == 'partial' or args.finetune_mode == 'all-but-trigger': 
        params.append({
            'params': model.lm_head.parameters(),
            'lr': args.finetune_lr if args.finetune_lr else args.lr
        })
    elif args.finetune_mode == 'all' or args.finetune_mode == 'all-but-trigger':
        params.append({
            'params': [p for p in model.parameters() if not torch.equal(p, model.relation_embeds)],
            'lr': args.finetune_lr if args.finetune_lr else args.lr
        })
    optimizer = AdamW(
        params,
        lr=args.lr,
        weight_decay=1e-2,
        #betas=(0.9, 0.999),
        eps=1e-8
    )

    if world_size != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
        )
    evaluator = EVALUATORS[args.evaluation_strategy](
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        decoding_strategy=args.decoding_strategy,
    )
    if is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.ckpt_dir)

    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        if not args.disable_dropout:
            model.train()
        else:
            model.eval()
        if is_main_process and not args.quiet:
            iter_ = tqdm(train_loader)
        else:
            iter_ = train_loader
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        optimizer.zero_grad()
        for i, (model_inputs, labels) in enumerate(iter_):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            loss, correct, preds = evaluator(model_inputs, labels, train=True)
            loss /= args.accumulation_steps
            loss.backward()
            if (i % args.accumulation_steps) == (args.accumulation_steps - 1):
                logger.debug('Optimizer step.')
                if args.clip is not None:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                optimizer.step()
                optimizer.zero_grad()
            total_loss += loss.detach() * labels.size(0)
            total_correct += correct.detach()
            denom += labels.size(0)

            if args.debug and args.evaluation_strategy == 'generative':
                for label, pred, mask in zip(labels, preds, model_inputs['predict_mask']):
                    logger.info(
                        'Label: %s - Pred: %s',
                        tokenizer.decode(label[mask]),
                        tokenizer.decode(pred[mask])
                    )
            
            # NOTE: This loss/accuracy is only on the subset  of training data
            # in the main process.
            if is_main_process and not args.quiet:
                iter_.set_description(
                    f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
                    f'Accuracy: {total_correct / (denom + 1e-13): 0.4f}'
                )
        if world_size != -1:
            torch.distributed.reduce(total_loss, 0)
            torch.distributed.reduce(total_correct, 0)
            torch.distributed.reduct(denom, 0)
        if is_main_process:
            writer.add_scalar('Loss/train', (total_loss / (denom + 1e-13)).item(), epoch)
            writer.add_scalar('Accuracy/train', (total_correct / (denom + 1e-13)).item(), epoch)

        logger.info('Evaluating...')
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        if is_main_process and not args.quiet:
            iter_ = tqdm(dev_loader)
        else:
            iter_ = dev_loader
        with torch.no_grad():
            for model_inputs, labels in iter_:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                loss, correct, preds = evaluator(model_inputs, labels, train=False)
                total_loss += loss.detach() * labels.size(0)
                total_correct += correct.detach()
                denom += labels.size(0)

                if args.debug and args.evaluation_strategy == 'generative':
                    for label, pred, mask in zip(labels, preds, model_inputs['predict_mask']):
                        logger.info(
                            'Label: %s - Pred: %s',
                            tokenizer.decode(label[mask]),
                            tokenizer.decode(pred[mask])
                        )
                    

        if world_size != -1:
            torch.distributed.reduce(total_loss, 0)
            torch.distributed.reduce(total_correct, 0)
            torch.distributed.reduce(denom, 0)
        if is_main_process:
            writer.add_scalar('Loss/dev', (total_loss / (denom + 1e-13)).item(), epoch)
            writer.add_scalar('Accuracy/dev', (total_correct / (denom + 1e-13)).item(), epoch)

        logger.info(
            f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
            f'Accuracy: {total_correct / (denom + 1e-13): 0.4f}'
        )
        accuracy = total_correct / (denom + 1e-13)

        if is_main_process:
            if accuracy > best_accuracy:
                logger.info('Best performance so far.')
                best_accuracy = accuracy
                if not args.ckpt_dir.exists():
                    args.ckpt_dir.mkdir(parents=True)
                if world_size != -1:
                    state_dict = model.module.state_dict()
                else:
                    state_dict = model.state_dict()
                if is_main_process:
                    torch.save(state_dict, ckpt_path)
                tokenizer.save_pretrained(args.ckpt_dir)

    logger.info('Testing...')
    if ckpt_path.exists():
        logger.info('Restoring checkpoint.')
        state_dict = torch.load(ckpt_path, map_location=device)
        model.load_state_dict(state_dict)
    model.eval()
    total_correct = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            _, correct, preds = evaluator(model_inputs, labels, train=False)
            total_correct += correct.detach()
            denom += labels.size(0)
    if world_size != -1:
        torch.distributed.reduce(correct, 0)
        torch.distributed.reduce(denom, 0)

    if args.tmp:
        logger.info('Temporary mode enabled, deleting checkpoint.')
        os.remove(ckpt_path)

    accuracy = total_correct / (denom + 1e-13)
    if is_main_process:
        writer.add_scalar('Loss/test', (total_loss / (denom + 1e-13)).item(), 0)
        writer.add_scalar('Accuracy/test', (total_correct / (denom + 1e-13)).item(), 0)
    logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset & model paths
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name or path to the underlying MLM.')
    parser.add_argument('--train', type=Path, required=True,
                        help='Path to the training dataset.')
    parser.add_argument('--dev', type=Path, required=True,
                        help='Path to the development dataset.')
    parser.add_argument('--test', type=Path, required=True,
                        help='Path to the test dataset.')
    parser.add_argument('--ckpt-dir', type=Path, default=Path('ckpt/'),
                        help='Path to save/load model checkpoint.')

    # Model/training set up
    parser.add_argument('--template', type=str, required=True,
                        help='Template used to define the placement of instance '
                             'fields, triggers, and prediction tokens.')
    parser.add_argument('--label-map', type=str, default=None,
                        help='A json-formatted string defining how labels are '
                             'mapped to strings in the model vocabulary.')
    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None,
                        help='A list of tokens used to initialize the trigger '
                             'embeddings.')
    parser.add_argument('--label-field', type=str, default='label',
                        help='The name of label field in the instance '
                             'dictionary.')
    parser.add_argument('--add-padding', action='store_true',
                        help='Add padding to the label field. Used for WSC-like '
                             'training.')
    parser.add_argument('--preprocessor', type=str, default=None,
                        choices=PREPROCESSORS.keys(),
                        help='Data preprocessor. If unspecified a default '
                             'preprocessor will be selected based on filetype.')
    parser.add_argument('--evaluation-strategy', type=str, required=True,
                        choices=EVALUATORS.keys(),
                        help='Evaluation strategy. Options: '
                             'generative: For generative tasks,'
                             'classification: For prediction tasks with a fixed '
                             'set of labels.')
    parser.add_argument('--decoding-strategy', type=str, default=None,
                        choices=['parallel', 'monotonic', 'iterative'],
                        help='Decoding strategy for generative tasks. For more '
                             'details refer to the PET paper.')
    parser.add_argument('--finetune-mode', type=str, default='trigger',
                        choices=['trigger', 'partial', 'all', 'all-but-trigger'],
                        help='Approach used for finetuning. Options: '
                             'trigger: Only triggers are tuned. '
                             'partial: Top model weights additionally tuned. '
                             'all: All model weights are tuned.'
                             'all-but-trigger: All model weights apart from trigger weights are tuned. ')

    # Hyperparameters
    parser.add_argument('--bsz', type=int, default=32, help='Batch size.')
    parser.add_argument('--accumulation-steps', type=int, default=1,
                        help='Number of accumulation steps.')
    parser.add_argument('--epochs', type=int, default=10,
                        help='Number of training epochs.')
    parser.add_argument('--lr', type=float, default=1e-4,
                        help='Global learning rate.')
    parser.add_argument('--finetune-lr', type=float, default=None,
                        help='Optional learning rate used when optimizing '
                             'non-trigger weights')
    parser.add_argument('--disable-dropout', action='store_true',
                        help='Disable dropout during training.')
    parser.add_argument('--clip', type=float, default=None,
                        help='Gradient clipping value.')
    parser.add_argument('--limit', type=int, default=None,
                        help='Randomly limit train/dev sets to specified '
                             'number of datapoints.')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed.')

    # Additional options
    parser.add_argument('-f', '--force-overwrite', action='store_true',
                        help='Allow overwriting an existing model checkpoint.')
    parser.add_argument('--debug', action='store_true',
                        help='Enable debug-level logging messages.')
    parser.add_argument('--quiet', action='store_true',
                        help='Make tqdm shut up. Useful if storing logs.')
    parser.add_argument('--tmp', action='store_true',
                        help='Remove model checkpoint after evaluation. '
                             'Useful when performing many experiments.')
    parser.add_argument('--local_rank', type=int, default=-1,
                        help='For parallel/distributed training. Usually will '
                             'be set automatically.')

    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)

