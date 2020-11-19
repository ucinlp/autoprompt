"""
Script for running a linear probe on glue tasks.

Largely copied from:
    https://github.com/huggingface/transformers/blob/master/examples/text-classification/run_glue.py
"""
import argparse
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, WEIGHTS_NAME, CONFIG_NAME
from tqdm import tqdm

from autoprompt.popsicle import AutoPopsicle
import autoprompt.utils as utils

logger = logging.getLogger(__name__)


def main(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoPopsicle.from_pretrained(args.model_name, config=config)
    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset, label_map = utils.load_classification_dataset(
        args.train,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset, _ = utils.load_classification_dataset(
        args.dev,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    test_dataset, _ = utils.load_classification_dataset(
        args.test,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map
    )
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.Adam(model.classifier.parameters(), lr=args.lr, weight_decay=1e-6)

    if not args.ckpt_dir.exists():
        # logger.info(f'Making checkpoint directory: {args.ckpt_dir}')
        args.ckpt_dir.mkdir(parents=True)
    elif not args.force_overwrite:
        raise RuntimeError('Checkpoint directory already exists.')

    best_accuracy = 0
    try:
        for epoch in range(args.epochs):
            logger.info('Training...')
            model.eval()  # Just linear regression - don't want model outputs changing during training.
            avg_loss = utils.ExponentialMovingAverage()
            pbar = tqdm(train_loader)
            for model_inputs, labels in pbar:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                optimizer.zero_grad()
                logits, *_ = model(**model_inputs)
                loss = F.cross_entropy(logits, labels.squeeze(-1))
                loss.backward()
                optimizer.step()
                avg_loss.update(loss.item())
                pbar.set_description(f'loss: {avg_loss.get_metric(): 0.4f}')

            logger.info('Evaluating...')
            model.eval()
            correct = 0
            total = 0
            for model_inputs, labels in dev_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                logits, *_ = model(**model_inputs)
                _, preds = logits.max(dim=-1)
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)
            accuracy = correct / (total + 1e-13)
            logger.info(f'Accuracy: {accuracy : 0.4f}')

            if accuracy > best_accuracy:
                logger.info('Best performance so far. Saving...')
                # torch.save(model.state_dict(), args.ckpt_dir / WEIGHTS_NAME)
                # model.config.to_json_file(args.ckpt_dir / CONFIG_NAME)
                model.save_pretrained(args.ckpt_dir)
                tokenizer.save_pretrained(args.ckpt_dir)
                best_accuracy = accuracy

    except KeyboardInterrupt:
        logger.info('Training manually terminated.')

    logger.info('Testing...')
    checkpoint = torch.load(args.ckpt_dir / WEIGHTS_NAME)
    model.load_state_dict(checkpoint)
    model.eval()
    correct = 0
    total = 0
    for model_inputs, labels in test_loader:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        labels = labels.to(device)
        logits, *_ = model(**model_inputs)
        _, preds = logits.max(dim=-1)
        correct += (preds == labels.squeeze(-1)).sum().item()
        total += labels.size(0)
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
    parser.add_argument('--ckpt-dir', type=Path, default=Path('ckpt/'))
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=1e-3)
    parser.add_argument('-f', '--force-overwrite', action='store_true', default=True)
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--log_file', type=str, default='log.txt')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level, filename=args.log_file)

    main(args)
