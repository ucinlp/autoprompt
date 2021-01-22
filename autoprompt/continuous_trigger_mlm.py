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
from autoprompt.models import ContinuousTriggerMLM
from autoprompt.preprocessors import PREPROCESSORS
from autoprompt.evaluators import MLM_EVALUATORS


logger = logging.getLogger(__name__)


def check_args(args):
    """Checks for invalid argument combinations."""
    if args.evaluation_strategy == 'exact-match':
        assert args.decoding_strategy is not None
    if args.evaluation_strategy == 'classification':
        assert args.label_map is not None
    if args.evaluation_strategy == 'multiple-choice':
        assert args.bsz is None, 'Multiple choice uses custom batching, do not set `--bsz`.'


def serialize_args(args):
    """Serializes arguments to the checkpoint directory."""
    if not os.path.exists(args.ckpt_dir):
        logger.info(f'Making directory: {args.ckpt_dir}')
        os.makedirs(args.ckpt_dir)
    fname = os.path.join(args.ckpt_dir, 'args.json')
    with open(fname, 'w') as f:
        logger.info(f'Serializing CLI arguments to: {fname}')
        json.dump(vars(args), f)


def load_label_map(label_map):
    """Loads the label map."""
    if label_map is not None:
        return json.loads(args.label_map)


def load_transformers(model_name):
    """Loads transformers config, tokenizer, and model.""" 
    config = AutoConfig.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        model_name,
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )
    model = AutoModelForMaskedLM.from_pretrained(model_name, config=config)
    return config, tokenizer, model


def get_sampler(
    dataset,
    evaluation_strategy,
    distributed_config,
    train=False
):
    """Sets up the correct sampler for a data loader."""
    # Sampling is handled by data iterator for multiple choice problems.
    if evaluation_strategy != 'classification':
        return
    # Multi-GPU training
    if distributed_config.world_size != -1:
        return torch.utils.data.DistributedSampler(dataset, shuffle=train)
    # Defaults
    if train:
        return torch.utils.data.RandomSampler(dataset)
    else:
        return torch.utils.data.SequentialSampler(dataset)


def load_datasets(args, templatizer, distributed_config):
    """Loads the training, dev and test datasets."""
    dataset_constructor = utils.DATASET_CONSTRUCTORS[args.evaluation_strategy]
    collator = utils.Collator(pad_token_id=templatizer.pad_token_id)

    train_dataset = dataset_constructor(
        args.train,
        templatizer=templatizer,
        train=True,
        preprocessor_key=args.preprocessor,
        limit=args.limit,
    )
    train_sampler = get_sampler(train_dataset, args.evaluation_strategy, distributed_config, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=collator, sampler=train_sampler)

    dev_dataset = dataset_constructor(
        args.dev,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
        limit=args.limit,
    )
    dev_sampler = get_sampler(dev_dataset, args.evaluation_strategy, distributed_config, train=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, collate_fn=collator, sampler=dev_sampler)

    test_dataset = dataset_constructor(
        args.test,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
    )
    test_sampler = get_sampler(dev_dataset, args.evaluation_strategy, distributed_config, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, collate_fn=collator, sampler=test_sampler)

    return train_loader, dev_loader, test_loader


def get_initial_trigger_ids(initial_trigger, tokenizer):
    """Converts a list of trigger tokens to a tensor of trigger token ids."""
    if initial_trigger is None:
        return
    initial_trigger_ids = torch.tensor(
        tokenizer.convert_tokens_to_ids(args.initial_trigger)
    )
    detokenized = tokenizer.convert_ids_to_tokens(initial_trigger_ids)
    logger.debug(f'Initial trigger (detokenized): {detokenized}')
    return initial_trigger_ids


def get_optimizer(model, args):
    """Handles setting the optimizer up for different finetuning modes."""
    if args.finetune_mode == 'all-but-trigger':
        params = []
    else:
        params = [{'params': [model.trigger_embeddings]}]
    if args.finetune_mode == 'partial' or args.finetune_mode == 'all-but-trigger': 
        params.append({
            'params': model.lm_head.parameters(),
            'lr': args.finetune_lr if args.finetune_lr else args.lr
        })
    elif args.finetune_mode == 'all' or args.finetune_mode == 'all-but-trigger':
        params.append({
            'params': [p for p in model.parameters() if not torch.equal(p, model.trigger_embeddings)],
            'lr': args.finetune_lr if args.finetune_lr else args.lr
        })
    return AdamW(
        params,
        lr=args.lr,
        weight_decay=1e-2,
        eps=1e-8
    )


def to_device(data, device):
    if isinstance(data, dict):
        return {k: to_device(v, device) for k, v in data.items()}
    elif isinstance(data, list):
        return [to_device(x, device) for x in data]
    elif isinstance(data, torch.Tensor):
        return data.to(device)


def main(args):
    utils.set_seed(args.seed)

    # Initialization.
    check_args(args)
    serialize_args(args)
    distributed_config = utils.distributed_setup(args.local_rank)
    if not args.debug:
        logging.basicConfig(level=logging.INFO if distributed_config.is_main_process else logging.WARN)
        logger.info('Suppressing subprocess logging. If this is not desired enable debug mode.')
    if distributed_config.is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.ckpt_dir)
    config, tokenizer, base_model = load_transformers(args.model_name)

    # Load data.
    logger.info('Loading data.')
    label_map = load_label_map(args.label_map)
    templatizer = utils.MultiTokenTemplatizer(
        template=args.template,
        tokenizer=tokenizer,
        label_field=args.label_field,
        label_map=label_map,
        add_padding=args.add_padding,
    )
    train_loader, dev_loader, test_loader = load_datasets(
        args,
        templatizer=templatizer,
        distributed_config=distributed_config,
    )

    # Setup model
    logger.info('Initializing model.')
    initial_trigger_ids = get_initial_trigger_ids(args.initial_trigger, tokenizer)
    model = ContinuousTriggerMLM(
        base_model=base_model,
        num_trigger_tokens=templatizer.num_trigger_tokens,
        initial_trigger_ids=initial_trigger_ids,
    )
    model.to(distributed_config.device)

    # Restore existing checkpoint if available.
    ckpt_path = os.path.join(args.ckpt_dir, 'pytorch_model.bin')
    if os.path.exists(ckpt_path) and not args.force_overwrite:
        logger.info('Restoring checkpoint.')
        state_dict = torch.load(ckpt_path, map_location=distributed_config.device)
        model.load_state_dict(state_dict)

    # Setup optimizer
    optimizer = get_optimizer(model, args)

    if distributed_config.world_size != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
        )
    evaluator = MLM_EVALUATORS[args.evaluation_strategy](
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        decoding_strategy=args.decoding_strategy,
    )

    best_accuracy = 0
    if not args.skip_train:
        for epoch in range(args.epochs):
            logger.info(f'Epoch: {epoch}')
            logger.info('Training...')
            if not args.disable_dropout:
                model.train()
            else:
                model.eval()
            if distributed_config.is_main_process and not args.quiet:
                iter_ = tqdm(train_loader)
            else:
                iter_ = train_loader
            total_loss = torch.tensor(0.0, device=distributed_config.device)
            total_correct = torch.tensor(0.0, device=distributed_config.device)
            denom = torch.tensor(0.0, device=distributed_config.device)
            optimizer.zero_grad()
            for i, (model_inputs, labels) in enumerate(iter_):
                model_inputs = to_device(model_inputs, distributed_config.device)
                labels = to_device(labels, distributed_config.device)
                loss, correct, preds = evaluator(model_inputs, labels, train=True)
                loss /= args.accumulation_steps
                loss.backward()
                if (i % args.accumulation_steps) == (args.accumulation_steps - 1):
                    logger.debug('Optimizer step.')
                    if args.clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    optimizer.zero_grad()
                # TODO: Metric logging is clumsy/brittle.
                batch_size = 1.0 if args.evaluation_strategy == 'multiple-choice' else labels.size(0)
                total_loss += loss.detach() * batch_size
                total_correct += correct.detach()
                denom += batch_size
                
                # NOTE: This loss/accuracy is only on the subset  of training data
                # in the main process.
                if distributed_config.is_main_process and not args.quiet:
                    iter_.set_description(
                        f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
                        f'Accuracy: {total_correct / (denom + 1e-13): 0.4f}'
                    )
            if distributed_config.world_size != -1:
                torch.distributed.reduce(total_loss, 0)
                torch.distributed.reduce(total_correct, 0)
                torch.distributed.reduct(denom, 0)
            if distributed_config.is_main_process:
                writer.add_scalar('Loss/train', (total_loss / (denom + 1e-13)).item(), epoch)
                writer.add_scalar('Accuracy/train', (total_correct / (denom + 1e-13)).item(), epoch)

            if not args.skip_eval:
                logger.info('Evaluating...')
                model.eval()
                total_loss = torch.tensor(0.0, device=distributed_config.device)
                total_correct = torch.tensor(0.0, device=distributed_config.device)
                denom = torch.tensor(0.0, device=distributed_config.device)
                if distributed_config.is_main_process and not args.quiet:
                    iter_ = tqdm(dev_loader)
                else:
                    iter_ = dev_loader
                with torch.no_grad():
                    for model_inputs, labels in iter_:
                        model_inputs = to_device(model_inputs, distributed_config.device)
                        labels = to_device(labels, distributed_config.device)
                        loss, correct, preds = evaluator(model_inputs, labels, train=False)
                        batch_size = 1.0 if args.evaluation_strategy == 'multiple-choice' else labels.size(0)
                        total_loss += loss.detach() * batch_size
                        total_correct += correct.detach()
                        denom += batch_size

                if distributed_config.world_size != -1:
                    torch.distributed.reduce(total_loss, 0)
                    torch.distributed.reduce(total_correct, 0)
                    torch.distributed.reduce(denom, 0)
                if distributed_config.is_main_process:
                    writer.add_scalar('Loss/dev', (total_loss / (denom + 1e-13)).item(), epoch)
                    writer.add_scalar('Accuracy/dev', (total_correct / (denom + 1e-13)).item(), epoch)

                logger.info(
                    f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
                    f'Accuracy: {total_correct / (denom + 1e-13): 0.4f}'
                )
                accuracy = total_correct / (denom + 1e-13)

                if distributed_config.is_main_process:
                    if accuracy > best_accuracy:
                        logger.info('Best performance so far.')
                        best_accuracy = accuracy
                        if distributed_config.world_size != -1:
                            state_dict = model.module.state_dict()
                        else:
                            state_dict = model.state_dict()
                        if distributed_config.is_main_process:
                            torch.save(state_dict, ckpt_path)
                        tokenizer.save_pretrained(args.ckpt_dir)
                        config.save_pretrained(args.ckpt_dir)

    if not args.skip_test:
        logger.info('Testing...')
        if os.path.exists(ckpt_path) and not args.skip_eval:
            logger.info('Restoring checkpoint.')
            state_dict = torch.load(ckpt_path, map_location=distributed_config.device)
            model.load_state_dict(state_dict)
        output_fname = os.path.join(args.ckpt_dir, 'predictions')
        model.eval()
        total_correct = torch.tensor(0.0, device=distributed_config.device)
        denom = torch.tensor(0.0, device=distributed_config.device)
        with torch.no_grad(), open(output_fname, 'w') as f:
            for model_inputs, labels in test_loader:
                model_inputs = {k: v.to(distributed_config.device) for k, v in model_inputs.items()}
                labels = labels.to(distributed_config.device)
                _, correct, preds = evaluator(model_inputs, labels, train=False)
                total_correct += correct.detach()
                denom += labels.size(0)
                # Serialize output
                for pred in preds:
                    print(pred, file=f)
        if distributed_config.world_size != -1:
            torch.distributed.reduce(correct, 0)
            torch.distributed.reduce(denom, 0)

        if args.tmp:
            if os.path.exists(ckpt_path):
                logger.info('Temporary mode enabled, deleting checkpoint.')
                os.remove(ckpt_path)

        accuracy = total_correct / (denom + 1e-13)
        if distributed_config.is_main_process:
            writer.add_scalar('Loss/test', (total_loss / (denom + 1e-13)).item(), 0)
            writer.add_scalar('Accuracy/test', (total_correct / (denom + 1e-13)).item(), 0)
        logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()

    # Dataset & model paths
    parser.add_argument('--model-name', type=str, required=True,
                        help='Name or path to the underlying MLM.')
    parser.add_argument('--train', type=str, required=True,
                        help='Path to the training dataset.')
    parser.add_argument('--dev', type=str, required=True,
                        help='Path to the development dataset.')
    parser.add_argument('--test', type=str, required=True,
                        help='Path to the test dataset.')
    parser.add_argument('--ckpt-dir', type=str, default='ckpt/',
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
                        choices=MLM_EVALUATORS.keys(),
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

    # Skips
    parser.add_argument('--skip-train', action='store_true',
                        help='Skip training.')
    parser.add_argument('--skip-eval', action='store_true',
                        help='Skip evaluation loop during training. Good for cranking through '
                             'expensive multiple-choice experiments.')
    parser.add_argument('--skip-test', action='store_true',
                        help='Skip test.')

    # Hyperparameters
    parser.add_argument('--bsz', type=int, default=None, help='Batch size.')
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

