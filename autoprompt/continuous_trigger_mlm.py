"""Continuous triggers for MLM prompting."""
import argparse
import logging
import os

import torch
from transformers import AdamW
from tqdm import tqdm

from autoprompt.evaluators import MLM_EVALUATORS
from autoprompt.metrics import METRICS
from autoprompt.models import ContinuousTriggerMLM
from autoprompt.preprocessors import PREPROCESSORS
import autoprompt.data as data
import autoprompt.templatizers as templatizers
import autoprompt.utils as utils


logger = logging.getLogger(__name__)


def get_optimizer(model, args):
    """Handles setting the optimizer up for different finetuning modes."""
    params = [{'params': [model.trigger_embeddings]}]
    if args.finetune_mode == 'partial':
        params.append({
            'params': model.lm_head.parameters(),
            'lr': args.finetune_lr if args.finetune_lr else args.lr
        })
    elif args.finetune_mode == 'all':
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


def main(args):
    # pylint: disable=C0116,E1121,R0912,R0915
    utils.set_seed(args.seed)

    # Initialization.
    utils.check_args(args)
    utils.serialize_args(args)
    distributed_config = utils.distributed_setup(args.local_rank)
    if not args.debug:
        logging.basicConfig(level=logging.INFO if distributed_config.is_main_process else logging.WARN)
        logger.info('Suppressing subprocess logging. If this is not desired enable debug mode.')
    if distributed_config.is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args.ckpt_dir)
    config, tokenizer, base_model = utils.load_transformers(args.model_name)

    # Load data.
    logger.info('Loading data.')
    label_map = utils.load_label_map(args.label_map)
    templatizer = templatizers.MultiTokenTemplatizer(
        template=args.template,
        tokenizer=tokenizer,
        label_field=args.label_field,
        label_map=label_map,
        add_padding=args.add_padding,
    )
    train_loader, dev_loader, test_loader = data.load_datasets(
        args,
        templatizer=templatizer,
        distributed_config=distributed_config,
    )

    # Setup model
    logger.info('Initializing model.')
    initial_trigger_ids = utils.get_initial_trigger_ids(args.initial_trigger, tokenizer)
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
    score_fn = METRICS[args.evaluation_metric]

    best_score = 0
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
            total_loss = torch.FloatTensor(0.0, device=distributed_config.device)
            total_metrics = {}
            denom = torch.FloatTensor(0.0, device=distributed_config.device)

            optimizer.zero_grad()
            for i, (model_inputs, labels) in enumerate(iter_):
                model_inputs = utils.to_device(model_inputs, distributed_config.device)
                labels = utils.to_device(labels, distributed_config.device)
                loss, metrics, preds = evaluator(model_inputs, labels, train=True,
                                        evaluation_metric=args.evaluation_metric)
                loss /= args.accumulation_steps
                loss.backward()
                if (i % args.accumulation_steps) == (args.accumulation_steps - 1):
                    logger.debug('Optimizer step.')
                    if args.clip is not None:
                        torch.nn.utils.clip_grad_norm_(model.parameters(), args.clip)
                    optimizer.step()
                    optimizer.zero_grad()

                batch_size = 1.0 if args.evaluation_strategy == 'multiple-choice' else labels.size(0)
                total_loss += loss.detach() * batch_size
                for metric in metrics:
                    if metric in total_metrics:
                        total_metrics[metric] += metrics[metric].detach()
                    else:
                        total_metrics[metric] = metrics[metric].detach()
                denom += batch_size

                # NOTE: This loss/accuracy is only on the subset of training data
                # in the main process.
                if distributed_config.is_main_process and not args.quiet:
                    score = score_fn(total_metrics, denom)
                    iter_.set_description(
                        f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
                        f'Metric: {score: 0.4f}'
                    )

            if distributed_config.world_size != -1:
                torch.distributed.reduce(total_loss, 0)
                for metric in total_metrics:
                    torch.distributed.reduce(total_metrics[metric], 0)
                torch.distributed.reduce(denom, 0)
            if distributed_config.is_main_process:
                writer.add_scalar('Loss/train', (total_loss / (denom + 1e-13)).item(), epoch)
                score = score_fn(total_metrics, denom)
                writer.add_scalar(f'{args.evaluation_metric.capitalize()}/train', score.item(), epoch)

            if not args.skip_eval:
                logger.info('Evaluating...')
                model.eval()
                total_loss = torch.FloatTensor(0.0, device=distributed_config.device)
                total_metrics = {}
                denom = torch.FloatTensor(0.0, device=distributed_config.device)
                if distributed_config.is_main_process and not args.quiet:
                    iter_ = tqdm(dev_loader)
                else:
                    iter_ = dev_loader
                with torch.no_grad():
                    for model_inputs, labels in iter_:
                        model_inputs = utils.to_device(model_inputs, distributed_config.device)
                        labels = utils.to_device(labels, distributed_config.device)
                        loss, metrics, preds = evaluator(model_inputs, labels,
                            train=False, evaluation_metric=args.evaluation_metric)
                        batch_size = 1.0 if args.evaluation_strategy == 'multiple-choice' else labels.size(0)
                        total_loss += loss.detach() * batch_size
                        for metric in metrics:
                            if metric in total_metrics:
                                total_metrics[metric] += metrics[metric].detach()
                            else:
                                total_metrics[metric] = metrics[metric].detach()
                        denom += batch_size

                        if distributed_config.world_size != -1:
                            torch.distributed.reduce(total_loss, 0)
                            for metric in total_metrics:
                                torch.distributed.reduce(total_metrics[metric], 0)
                            torch.distributed.reduce(denom, 0)
                        if distributed_config.is_main_process and not args.quiet:
                            score = score_fn(total_metrics, denom)
                            iter_.set_description(
                                f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
                                f'Metric: {score: 0.4f}'
                            )
                if distributed_config.is_main_process:
                    writer.add_scalar('Loss/dev', (total_loss / (denom + 1e-13)).item(), epoch)
                    score = score_fn(total_metrics, denom)
                    writer.add_scalar(f'{args.evaluation_metric.capitalize()}/dev', score.item(), epoch)

                    if score > best_score:
                        logger.info('Best performance so far.')
                        best_score = score
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
        total_metrics = {}
        denom = torch.FloatTensor(0.0, device=distributed_config.device)
        with torch.no_grad(), open(output_fname, 'w') as f:
            for model_inputs, labels in test_loader:
                model_inputs = {k: v.to(distributed_config.device) for k, v in model_inputs.items()}
                labels = labels.to(distributed_config.device)
                _, metrics, preds = evaluator(model_inputs, labels, train=False,
                                        evaluation_metric=args.evaluation_metric)
                batch_size = 1.0 if args.evaluation_strategy == 'multiple-choice' else labels.size(0)
                for metric in metrics:
                    if metric in total_metrics:
                        total_metrics[metric] += metrics[metric].detach()
                    else:
                        total_metrics[metric] = metrics[metric].detach()
                denom += batch_size
                # Serialize output
                for pred in preds:
                    print(pred, file=f)
        if distributed_config.world_size != -1:
            for metric in total_metrics:
                torch.distributed.reduce(metrics[metric], 0)
            torch.distributed.reduce(denom, 0)

        if args.tmp:
            if os.path.exists(ckpt_path):
                logger.info('Temporary mode enabled, deleting checkpoint.')
                os.remove(ckpt_path)

        score = score_fn(metrics, denom)
        writer.add_scalar(f'{args.evaluation_metric.capitalize()}/test', score.item(), epoch)
        logger.info(f'Metric: {score: 0.4f}')


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
    parser.add_argument('--evaluation-metric', type=str, default='accuracy',
                        choices=list(METRICS.keys()),
                        help='Evaluation metric to use.')

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
