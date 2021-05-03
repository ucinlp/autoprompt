"""Finetuning for classification."""
import argparse
import contextlib
import io
import logging
import os

# from checklist.test_suite import TestSuite
import numpy as np
import transformers
import torch
from torch.optim.lr_scheduler import LambdaLR
from tqdm import tqdm

from autoprompt.metrics import METRICS
from autoprompt.models import ContinuousTriggerMLM, LinearComboMLM
from autoprompt.optimizers import L1SGD
from autoprompt.preprocessors import PREPROCESSORS
from autoprompt.trainers import Trainer
import autoprompt.data as data
import autoprompt.templatizers as templatizers
import autoprompt.utils as utils


logger = logging.getLogger(__name__)


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


def get_optimizer(model, args):
    """Handles setting the optimizer up for different finetuning modes."""
    if 'bitfit' in args['finetune_mode'] and 'layernorm' in args['finetune_mode']:
        raise ValueError('Cannot finetune both bitfit and layernorm due to overlapping '
                         'parameters.')
    if args['finetune_mode']:
        choices={'bitfit', 'layernorm', 'adapter'}
        for x in args['finetune_mode']:
            if x not in choices:
                raise ValueError(f'Unsupported finetune mode: {x}')

    # Default optimizer and kwargs
    optimizer = transformers.AdamW
    kwargs = {
        'weight_decay':1e-2,
        'eps': 1e-8,
    }

    # Finetune all by default
    if not args['finetune_mode']:
        return optimizer(
            model.parameters(),
            lr=args['lr'],
            **kwargs
        )

    params = []
    verboten_modules = []
    if 'adapter' in args['finetune_mode']:
        p = []
        for module in model.modules():
            if isinstance(module, transformers.adapter_modeling.Adapter):
                p.extend(module.parameters())
                verboten_modules.extend(module.modules())
        params.append({
            'params': p,
            'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
        })
    if 'bitfit' in args['finetune_mode']:
        for module in model.modules():
            if module in verboten_modules:
                continue
            if isinstance(module, (torch.nn.LayerNorm, torch.nn.Linear)):
                params.append({
                    'params': [module.bias],
                    'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
                })
    if 'layernorm' in args['finetune_mode']:
        for module in model.modules():
            if module in verboten_modules:
                continue
            if isinstance(module, torch.nn.LayerNorm):
                params.append({
                    'params': module.parameters(),
                    'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
                })
    return optimizer(
        params,
        lr=args['lr'],
        **kwargs
    )


class FinetuneTrainer(Trainer):
    def train(self, train_loader, dev_loader):

        # TODO(rloganiv): This is lazy.
        args = self.args

        if not os.path.exists(args['ckpt_dir']):
            os.makedirs(args['ckpt_dir'])
        # Setup model
        logger.info('Initializing model.')
        model = transformers.AutoModelForSequenceClassification.from_pretrained(args['model_name'],
                                                                                config=self.config)
        if 'adapter' in args['finetune_mode']:
            logger.info('Adding adapter.')
            assert isinstance(
                model,
                (
                    transformers.BertPreTrainedModel,
                    transformers.modeling_roberta.RobertaPreTrainedModel,
                )
            )
            adapter_config = transformers.AdapterConfig.load(
                'houlsby',
                reduction_factor=args['reduction_factor'],
            )
            model.add_adapter(
                'adapter',
                adapter_type=transformers.AdapterType.text_task,
                config=adapter_config,
            )
            model.train_adapter(['adapter'])
            model.set_active_adapters(['adapter'])
            for parameter in model.parameters():
                parameter.requires_grad = True

        model.to(self.distributed_config.device)

        # Restore existing checkpoint if available.
        ckpt_path = os.path.join(args['ckpt_dir'], 'pytorch_model.bin')
        if os.path.exists(ckpt_path) and not args['force_overwrite']:
            logger.info('Restoring checkpoint.')
            state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
            model.load_state_dict(state_dict)

        # Setup optimizer
        optimizer = get_optimizer(model, args)
        total_steps = args['train_size'] * args['epochs'] / (args['bsz'] * args['accumulation_steps'])
        warmup_steps = 0.1 * total_steps
        scheduler = get_linear_schedule_with_warmup(optimizer, warmup_steps, total_steps)
 
        # TODO(ewallace): The count for partial will be inaccurate since we count *all* of the LM head
        # params, whereas we are actually only updating the few that correspond to the label token names.
        total = 0
        for param_group in optimizer.param_groups:
             for tensor in param_group['params']:
                 total += tensor.numel()
        logger.info(f'Using finetuning mode: {args["finetune_mode"]}')
        logger.info(f'Updating {total} / {sum(p.numel() for p in model.parameters())} params.')

        if self.distributed_config.world_size != -1:
            model = torch.nn.parallel.DistributedDataParallel(
                model,
                device_ids=[args['local_rank']],
            )
        metric = METRICS[args['evaluation_metric']](
            label_map=self.label_map
        )

        best_score = -float('inf')
        best_metric_dict = {}
        if not args['skip_train']:
            for epoch in range(args['epochs']):
                logger.info(f'Epoch: {epoch}')
                logger.info('Training...')
                if not args['disable_dropout']:
                    model.train()
                else:
                    model.eval()
                if self.distributed_config.is_main_process and not args['quiet']:
                    iter_ = tqdm(train_loader)
                else:
                    iter_ = train_loader
                total_loss = torch.tensor(0.0, device=self.distributed_config.device)
                denom = torch.tensor(0.0, device=self.distributed_config.device)
                metric.reset()

                optimizer.zero_grad()
                for i, (model_inputs, labels) in enumerate(iter_):
                    model_inputs = utils.to_device(model_inputs, self.distributed_config.device)
                    labels = utils.to_device(labels, self.distributed_config.device)
                    loss, logits, *_ = model(**model_inputs, labels=labels)
                    preds = logits.argmax(dim=-1)
                    metric.update(labels, preds)
                    loss /= args['accumulation_steps']
                    loss.backward()
                    if (i % args['accumulation_steps']) == (args['accumulation_steps'] - 1):
                        if args['clip'] is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
                        optimizer.step()
                        optimizer.zero_grad()
                        scheduler.step()

                    batch_size = 1.0 if args['evaluation_strategy'] == 'multiple-choice' else labels.size(0)
                    total_loss += loss.detach() * batch_size
                    denom += batch_size

                    # NOTE: This loss/accuracy is only on the subset of training data
                    # in the main process.

                    if self.distributed_config.is_main_process and not args['quiet']:
                        metric_dict = metric.get()
                        metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
                        iter_.set_description(
                            f'loss: {total_loss / (denom + 1e-13): 0.4f}, ' +
                            metric_string
                        )

                if self.distributed_config.world_size != -1:
                    torch.distributed.reduce(total_loss, 0)
                    torch.distributed.reduce(denom, 0)
                    metric.reduce()
                if self.distributed_config.is_main_process:
                    self.writer.add_scalar('Loss/train', (total_loss / (denom + 1e-13)).item(), epoch)
                    metric_dict = metric.get()
                    for key, value in metric_dict.items():
                        self.writer.add_scalar(f'{key.capitalize()}/train', value, epoch)

                if not args['skip_eval']:
                    logger.info('Evaluating...')
                    model.eval()
                    total_loss = torch.tensor(0.0, device=self.distributed_config.device)
                    denom = torch.tensor(0.0, device=self.distributed_config.device)
                    metric.reset()

                    if self.distributed_config.is_main_process and not args['quiet']:
                        iter_ = tqdm(dev_loader)
                    else:
                        iter_ = dev_loader
                    with torch.no_grad():
                        for model_inputs, labels in iter_:
                            model_inputs = utils.to_device(model_inputs, self.distributed_config.device)
                            labels = utils.to_device(labels, self.distributed_config.device)
                            loss, logits, *_ = model(**model_inputs, labels=labels)
                            preds = logits.argmax(dim=-1)
                            metric.update(labels, preds)
                            batch_size = 1.0 if args['evaluation_strategy'] == 'multiple-choice' else labels.size(0)
                            total_loss += loss.detach() * batch_size
                            denom += batch_size

                            if self.distributed_config.world_size != -1:
                                torch.distributed.reduce(total_loss, 0)
                                torch.distributed.reduce(denom, 0)
                                metric.reduce()
                            if self.distributed_config.is_main_process and not args['quiet']:
                                metric_dict = metric.get()
                                metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
                                iter_.set_description(
                                    f'loss: {total_loss / (denom + 1e-13): 0.4f}, ' +
                                    metric_string
                                )
                    if self.distributed_config.is_main_process:
                        # !!!
                        self.writer.add_scalar('Loss/dev', (total_loss / (denom + 1e-13)).item(), epoch)
                        metric_dict = metric.get()
                        for key, value in metric_dict.items():
                            self.writer.add_scalar(f'{key.capitalize()}/dev', value, epoch)


                        score = metric_dict[metric.score_key]
                        if score > best_score:
                            logger.info('Best performance so far.')
                            best_score = score
                            best_metric_dict = metric_dict
                            if self.distributed_config.world_size != -1:
                                model_to_save = model.module
                            else:
                                model_to_save = model

                            if self.distributed_config.is_main_process:
                                state_dict = model_to_save.state_dict()
                                torch.save(state_dict, ckpt_path)
                            self.tokenizer.save_pretrained(args['ckpt_dir'])
                            self.config.save_pretrained(args['ckpt_dir'])
                            if 'adapter' in args['finetune_mode']:
                                model_to_save.base_model.save_adapter(
                                    args['ckpt_dir'],
                                    adapter_name='adapter',
                                )

            if os.path.exists(ckpt_path) and not args['skip_eval']:
                logger.info('Restoring checkpoint.')
                if self.distributed_config.world_size != -1:
                    model_to_load = model.module
                else:
                    model_to_load = model
                state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
                model_to_load.load_state_dict(state_dict)
                if 'adapter' in args['finetune_mode']:
                    model_to_load.base_model.load_adapter(
                        args['ckpt_dir'],
                        adapter_type=transformers.AdapterType.text_task,
                        config=adapter_config,
                    )

        return model, best_metric_dict

    def test(self, model, test_loader):

        # TODO(rloganiv): This is lazy.
        args = self.args

        if not args['skip_test']:
            ckpt_path = os.path.join(args['ckpt_dir'], 'pytorch_model.bin')
            metric = METRICS[args['evaluation_metric']](
                label_map=self.label_map,
            )
            output_fname = os.path.join(args['ckpt_dir'], 'predictions')
            model.eval()

            with torch.no_grad(), open(output_fname, 'w') as f:
                for model_inputs, labels in test_loader:
                    model_inputs = {k: v.to(self.distributed_config.device) for k, v in model_inputs.items()}
                    labels = labels.to(self.distributed_config.device)
                    loss, logits, *_ = model(**model_inputs, labels=labels)
                    preds = logits.argmax(dim=-1)
                    metric.update(labels, preds)

                    # Serialize output
                    for pred in preds:
                        print(pred, file=f)

            # TODO: Update metric...
            if self.distributed_config.world_size != -1:
                metric.reduce()

            if args['tmp']:
                if os.path.exists(ckpt_path):
                    logger.info('Temporary mode enabled, deleting checkpoint.')
                    os.remove(ckpt_path)
                    if 'adapter' in args['finetune_mode']:
                        adapter_path = os.path.join(args['ckpt_dir'], 'pytorch_adapter.bin')
                        adapter_head_path = os.path.join(args['ckpt_dir'], 'pytorch_model_head.bin')
                        if os.path.exists(adapter_path):
                            os.remove(adapter_path)
                        if os.path.exists(adapter_head_path):
                            os.remove(adapter_head_path)

            metric_dict = metric.get()
            for key, value in metric_dict.items():
                self.writer.add_scalar(f'{key.capitalize()}/test', value, 0)
            metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
            logger.info(metric_string)

            return metric_dict


def main(args):
    # pylint: disable=C0116,E1121,R0912,R0915
    utils.set_seed(args['seed'])
    utils.serialize_args(args)
    args['evaluation_strategy'] = 'classification'  # Hard-coded

    distributed_config = utils.distributed_setup(args['local_rank'])
    if not args['debug']:
        logging.basicConfig(level=logging.INFO if distributed_config.is_main_process else logging.WARN)
        logger.info('Suppressing subprocess logging. If this is not desired enable debug mode.')
    if distributed_config.is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args['ckpt_dir'])
    else:
        writer = utils.NullWriter()
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args['model_name'],
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )
    templatizer = templatizers.FinetuneTemplatizer(
        tokenizer=tokenizer,
        input_field_a=args['input_field_a'],
        input_field_b=args['input_field_b'],
        label_field=args['label_field'],
    )

    logger.info('Loading data.')
    train_loader, dev_loader, test_loader, checklist_test_loader = data.load_datasets(
        args,
        templatizer=templatizer,
        distributed_config=distributed_config,
    )
    config = transformers.AutoConfig.from_pretrained(args['model_name'], num_labels=len(templatizer._label_map))
    trainer = FinetuneTrainer(
        args=args,
        config=config,
        tokenizer=tokenizer,
        templatizer=templatizer,
        label_map=templatizer._label_map,
        distributed_config=distributed_config,
        writer=writer,
    )
    model, _ = trainer.train(train_loader, dev_loader)
    trainer.test(model, test_loader=test_loader)


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
    parser.add_argument('--checklist', type=str, default=None,
                        help='Path to a checklist test set.')
    parser.add_argument('--ckpt-dir', type=str, default='ckpt/',
                        help='Path to save/load model checkpoint.')

    # Model/training set up
    parser.add_argument('--input-field-a', type=str, required=True)
    parser.add_argument('--input-field-b', type=str, default=None)
    parser.add_argument('--label-field', type=str, default='label',
                        help='The name of label field in the instance '
                             'dictionary.')
    parser.add_argument('--preprocessor', type=str, default=None,
                        choices=PREPROCESSORS.keys(),
                        help='Data preprocessor. If unspecified a default '
                             'preprocessor will be selected based on filetype.')
    parser.add_argument('--finetune-mode', type=str, nargs='*',
                        help='Components of model to finetune (multiple can be specified). If '
                             'nothing is specified then all parameters will be tuned. '
                             'Options: '
                             'trigger: trigger embeddings. '
                             'top-layer: top model weights. '
                             'bitfit: bias terms. '
                             'layernorm: layer norm params. ')
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
    parser.add_argument('--reduction-factor', type=int, default=16,
                        help='Reduction factor if using adapters')
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

    args = vars(parser.parse_args())

    if args['debug']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
