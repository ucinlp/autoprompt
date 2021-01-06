"""
Script for running finetuning on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import json
import logging
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import (
    DataLoader, DistributedSampler, RandomSampler, SequentialSampler
)
from torch.optim.lr_scheduler import LambdaLR
import transformers
from transformers import (
    AdamW, AutoConfig, AutoModelForSequenceClassification, AutoTokenizer
)
from tqdm import tqdm

import autoprompt.utils as utils
from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def get_linear_schedule_with_warmup(optimizer, num_warmup_steps, num_training_steps, last_epoch=-1):
    """ Create a schedule with a learning rate that decreases linearly after
    linearly increasing during a warmup period.

    From:
        https://github.com/uds-lsv/bert-stable-fine-tuning/blob/master/src/transformers/optimization.py
    """

    def lr_lambda(current_step):
        if current_step < num_warmup_steps:
            return float(current_step) / float(max(1, num_warmup_steps))
        return max(
            0.0, float(num_training_steps - current_step) / float(max(1, num_training_steps - num_warmup_steps))
        )

    return LambdaLR(optimizer, lr_lambda, last_epoch)


def main(args):
    # Handle multi-GPU setup
    world_size = None
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
        torch.distributed.init_process_group(
            backend='nccl',
            init_method='env://',
        )
        world_size = torch.distributed.get_world_size()
    is_main_process = args.local_rank in [-1, 0]
    if args.debug:
        main_level = logging.DEBUG
        level = logging.DEBUG
    else:
        main_level = logging.INFO
        level = logging.WARN
    logging.basicConfig(level=main_level if is_main_process else level)
    logger.warning('Rank: %s - World Size: %s', args.local_rank, world_size)

    set_seed(args.seed)

    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    if args.adapter:
        model = transformers.AutoModelWithHeads.from_pretrained(
            args.model_name,
            config=config,
        )
        model.add_adapter(
            'adapter', 
            transformers.AdapterType.text_task,
            config='pfeiffer',
        )
        model.train_adapter(['adapter'])
        model.add_classification_head('adapter', num_labels=args.num_labels)
        model.set_active_adapters('adapter')
    else:
        model = AutoModelForSequenceClassification.from_pretrained(
            args.model_name,
            config=config,
        )
    model.to(device)
    if args.local_rank != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
            output_device=args.local_rank
        )

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    if args.label_map:
        label_map = json.loads(args.label_map)
    else:
        label_map = None
    train_dataset, label_map = utils.load_classification_dataset(
        args.train,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map,  # If None then will be learned
        limit=args.limit,
        preprocessor_key=args.preprocessor,
    )
    logger.info(f'Label map: {label_map}')
    if args.local_rank != -1:
        train_sampler = DistributedSampler(train_dataset, shuffle=True, seed=args.seed)
    else:
        train_sampler = RandomSampler(train_dataset)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.bsz,
        sampler=train_sampler,
        collate_fn=collator,
    )
    dev_dataset, _ = utils.load_classification_dataset(
        args.dev,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map,
        limit=args.limit,
        preprocessor_key=args.preprocessor,
    )
    if args.local_rank != -1:
        dev_sampler = DistributedSampler(dev_dataset)
    else:
        dev_sampler = SequentialSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, sampler=dev_sampler, collate_fn=collator)
    test_dataset, _ = utils.load_classification_dataset(
        args.test,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map,
        preprocessor_key=args.preprocessor,
    )
    if args.local_rank != -1:
        test_sampler = DistributedSampler(test_dataset)
    else:
        test_sampler = SequentialSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, sampler=test_sampler, collate_fn=collator)

    if args.bias_correction:
        betas = (0.9, 0.999)
    else:
        betas = (0.0, 0.000)

    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
        betas=betas
    )
    scaler = torch.cuda.amp.GradScaler(enabled=args.fp16)

    # Use suggested learning rate scheduler
    if args.warmup:
        num_training_steps = len(train_dataset) * args.epochs // args.bsz
        num_warmup_steps = num_training_steps // 10
        scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps,
                                                    num_training_steps)

    if not args.ckpt_dir.exists():
        logger.info(f'Making checkpoint directory: {args.ckpt_dir}')
        args.ckpt_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')

    if not args.skip_train:
        best_accuracy = 0
        for epoch in range(args.epochs):
            logger.info('Training...')
            model.train()
            if is_main_process and not args.quiet:
                iter_ = tqdm(train_loader)
            else:
                iter_ = train_loader
            for i, (model_inputs, labels) in enumerate(iter_):
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                with torch.cuda.amp.autocast(enabled=args.fp16):
                    logits, *_ = model(**model_inputs)
                    loss = F.cross_entropy(logits, labels.squeeze(-1))
                    loss /= args.accumulation_steps
                _, preds = logits.max(dim=-1)
                correct = (preds == labels.squeeze(-1)).sum()
                total = labels.size(0)
                accuracy = correct / (total + 1e-13)
                scaler.scale(loss).backward()
                if i % args.accumulation_steps == 0:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad()
                if args.warmup:
                    scheduler.step()
                if i % args.log_frequency == 0 and is_main_process and not args.quiet:
                    iter_.set_description(
                        f'loss: {loss: 0.4f}, '
                        f'lr: {optimizer.param_groups[0]["lr"]: .3e} '
                        f'acc: {accuracy: 0.4f}'
                    )

            logger.info('Evaluating...')
            model.eval()
            correct = torch.tensor(0.0, device=device)
            total = torch.tensor(0.0, device=device)
            with torch.no_grad():
                for model_inputs, labels in dev_loader:
                    model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                    labels = labels.to(device)
                    logits, *_ = model(**model_inputs)
                    _, preds = logits.max(dim=-1)
                    correct += (preds == labels.squeeze(-1)).sum().item()
                    total += labels.size(0)

            if args.local_rank != -1:
                torch.distributed.reduce(correct, 0)
                torch.distributed.reduce(total, 0)
            accuracy = correct / (total + 1e-13)
            logger.info(f'Accuracy: {accuracy : 0.4f}')

            if accuracy > best_accuracy:
                logger.info('Best performance so far.')
                if args.local_rank == -1:
                    model.save_pretrained(args.ckpt_dir)
                elif args.local_rank == 0:
                    model.module.save_pretrained(args.ckpt_dir)
                if is_main_process:
                    config.save_pretrained(args.ckpt_dir)
                    tokenizer.save_pretrained(args.ckpt_dir)
                best_accuracy = accuracy

    logger.info('Testing...')
    if not args.skip_train:
        if not args.adapter:
            model = model.from_pretrained(args.ckpt_dir)
            model.to(device)
        if args.tmp:
            logger.info('Removing checkpoint.')
            shutil.rmtree(args.ckpt_dir)
    model.eval()

    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            logits, *_ = model(**model_inputs)
            _, preds = logits.max(dim=-1)
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)

    if args.local_rank != -1:
        torch.distributed.reduce(correct, 0)
        torch.distributed.reduce(total, 0)
    accuracy = correct / (total + 1e-13)
    logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--dev', type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--field-a', type=str)
    parser.add_argument('--field-b', type=str, default=None)
    parser.add_argument('--label-field', type=str, default='label')
    parser.add_argument('--label-map', type=str, default=None)
    parser.add_argument('--preprocessor', type=str, default=None,
                        choices=PREPROCESSORS.keys())
    parser.add_argument('--ckpt-dir', type=Path, default=Path('ckpt/'))
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=3)
    parser.add_argument('--accumulation-steps', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-5)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('--bias-correction', action='store_true')
    parser.add_argument('--fp16', action='store_true')
    parser.add_argument('--log-frequency', type=int, default=100)
    parser.add_argument('-f', '--force-overwrite', action='store_true')
    parser.add_argument('--adapter', action='store_true')
    parser.add_argument('--warmup', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--skip-train', action='store_true')
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    main(args)

