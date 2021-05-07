"""Continuous triggers for MLM prompting."""
import argparse
import contextlib
import io
import logging
import os

# from checklist.test_suite import TestSuite
import numpy as np
import transformers
import torch
from tqdm import tqdm

from autoprompt.evaluators import MLM_EVALUATORS
from autoprompt.metrics import METRICS
from autoprompt.models import ContinuousTriggerMLM, LinearComboMLM
from autoprompt.optimizers import L1SGD
from autoprompt.preprocessors import PREPROCESSORS
from autoprompt.trainers import Trainer
import autoprompt.data as data
import autoprompt.templatizers as templatizers
import autoprompt.utils as utils


logger = logging.getLogger(__name__)


def get_optimizer(model, args):
    """Handles setting the optimizer up for different finetuning modes."""
    if 'bitfit' in args['finetune_mode'] and 'layernorm' in args['finetune_mode']:
        raise ValueError('Cannot finetune both bitfit and layernorm due to overlapping '
                         'parameters.')
    if args['finetune_mode']:
        choices={'trigger', 'top-layer', 'bitfit', 'layernorm', 'adapter', 'calibration'}
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
    if 'trigger' in args['finetune_mode']:
        if isinstance(model, ContinuousTriggerMLM):
            params.append({'params': [model.trigger_embeddings]})
        elif isinstance(model, LinearComboMLM):
            params.append({
                'params': [model.trigger_projection],
                'l1decay': args['l1decay'],
                'theta': args['theta']
            })
            optimizer = L1SGD
            kwargs = {}
    if 'top-layer' in args['finetune_mode']:
        params.append({
            'params': model.lm_head.parameters(),
            'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
        })
        verboten_modules.extend(model.lm_head.modules())
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
    if 'calibration' in args['finetune_mode']:
        label_map = utils.load_label_map(args['label_map'])
        num_labels = len(list(label_map.keys()))
        model.calibration_layer = torch.nn.Linear(num_labels, num_labels)
        # initialize so the layer is the identity
        model.calibration_layer.weight = torch.nn.Parameter(torch.eye(num_labels))
        model.calibration_layer.bias = torch.nn.Parameter(torch.zeros(num_labels))
        model.calibration_layer.to(model.word_embeddings.weight.device)
        params.append({
            'params': model.calibration_layer.parameters(),
            'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
        })

    return optimizer(
        params,
        lr=args['lr'],
        **kwargs
    )


class ContinuousMLMTrainer(Trainer):
    def train(self, train_loader, dev_loader):

        # TODO(rloganiv): This is lazy.
        args = self.args

        if not os.path.exists(args['ckpt_dir']):
            os.makedirs(args['ckpt_dir'])
        # Setup model
        logger.info('Initializing model.')
        base_model = transformers.AutoModelForMaskedLM.from_pretrained(args['model_name'],
                                                                       config=self.config)
        if 'adapter' in args['finetune_mode']:
            logger.info('Adding adapter.')
            assert isinstance(
                base_model,
                (
                    transformers.BertPreTrainedModel,
                    transformers.modeling_roberta.RobertaPreTrainedModel,
                )
            )
            adapter_config = transformers.AdapterConfig.load(
                'houlsby',
                reduction_factor=args['reduction_factor'],
            )
            base_model.add_adapter(
                'adapter',
                adapter_type=transformers.AdapterType.text_task,
                config=adapter_config,
            )
            base_model.train_adapter(['adapter'])
            base_model.set_active_adapters(['adapter'])
            for parameter in base_model.parameters():
                parameter.requires_grad = True

        initial_trigger_ids = utils.get_initial_trigger_ids(args['initial_trigger'], self.tokenizer)
        if args['linear']:
            model = LinearComboMLM(
                base_model=base_model,
                num_trigger_tokens=self.templatizer.num_trigger_tokens,
                initial_trigger_ids=initial_trigger_ids,
            )
        else:
            model = ContinuousTriggerMLM(
                base_model=base_model,
                num_trigger_tokens=self.templatizer.num_trigger_tokens,
                initial_trigger_ids=initial_trigger_ids,
            )
        model.to(self.distributed_config.device)

        # Restore existing checkpoint if available.
        ckpt_path = os.path.join(args['ckpt_dir'], 'pytorch_model.bin')
        if os.path.exists(ckpt_path) and not args['force_overwrite']:
            logger.info('Restoring checkpoint.')
            state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
            model.load_state_dict(state_dict)

        # Setup optimizer
        optimizer = get_optimizer(model, args)
 
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
        evaluator = MLM_EVALUATORS[args['evaluation_strategy']](
            model=model,
            tokenizer=self.tokenizer,
            label_map=self.label_map,
            decoding_strategy=args['decoding_strategy'],
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
                    loss, preds = evaluator(model_inputs, labels, metric=metric, train=True)
                    loss /= args['accumulation_steps']
                    loss.backward()
                    if (i % args['accumulation_steps']) == (args['accumulation_steps'] - 1):
                        if args['clip'] is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
                        optimizer.step()
                        optimizer.zero_grad()

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
                            loss, preds = evaluator(model_inputs, labels, metric=metric, train=False)
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

                        if args['linear']:
                            zero_frac = model.trigger_projection.eq(0.0).sum() / torch.numel(model.trigger_projection)
                            logger.info(f'Fraction of Zero Weights: {zero_frac}')

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
            evaluator = MLM_EVALUATORS[args['evaluation_strategy']](
                model=model,
                tokenizer=self.tokenizer,
                label_map=self.label_map,
                decoding_strategy=args['decoding_strategy'],
            )
            metric = METRICS[args['evaluation_metric']](
                label_map=self.label_map,
            )
            output_fname = os.path.join(args['ckpt_dir'], 'predictions')
            model.eval()

            with torch.no_grad(), open(output_fname, 'w') as f:
                for model_inputs, labels in test_loader:
                    model_inputs = {k: v.to(self.distributed_config.device) for k, v in model_inputs.items()}
                    labels = labels.to(self.distributed_config.device)
                    _, preds = evaluator(model_inputs, labels, metric=metric, train=False)

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

            if args['linear']:
                zero_frac = model.trigger_projection.eq(0.0).sum() / torch.numel(model.trigger_projection)
                logger.info(f'Fraction of Zero Weights: {zero_frac}')

            metric_dict = metric.get()
            for key, value in metric_dict.items():
                self.writer.add_scalar(f'{key.capitalize()}/test', value, 0)
            metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
            logger.info(metric_string)

            return metric_dict


def main(args):
    # pylint: disable=C0116,E1121,R0912,R0915
    utils.set_seed(args['seed'])
    utils.check_args(args)
    utils.serialize_args(args)
    distributed_config = utils.distributed_setup(args['local_rank'])
    if not args['debug']:
        logging.basicConfig(level=logging.INFO if distributed_config.is_main_process else logging.WARN)
        logger.info('Suppressing subprocess logging. If this is not desired enable debug mode.')
    if distributed_config.is_main_process:
        writer = torch.utils.tensorboard.SummaryWriter(log_dir=args['ckpt_dir'])
    else:
        writer = utils.NullWriter()
    config = transformers.AutoConfig.from_pretrained(args['model_name'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args['model_name'],
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )

    logger.info('Loading data.')
    label_map = utils.load_label_map(args['label_map'])
    templatizer = templatizers.MultiTokenTemplatizer(
        template=args['template'],
        tokenizer=tokenizer,
        label_field=args['label_field'],
        label_map=label_map,
        add_padding=args['add_padding'],
    )
    train_loader, dev_loader, test_loader, checklist_test_loader = data.load_datasets(
        args,
        templatizer=templatizer,
        distributed_config=distributed_config,
    )
    trainer = ContinuousMLMTrainer(
        args=args,
        config=config,
        tokenizer=tokenizer,
        templatizer=templatizer,
        label_map=label_map,
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
    parser.add_argument('--template', type=str, required=True,
                        help='Template used to define the placement of instance '
                             'fields, triggers, and prediction tokens.')
    parser.add_argument('--label-map', type=str, default=None,
                        help='A json-formatted string defining how labels are '
                             'mapped to strings in the model vocabulary.')
    parser.add_argument('--initial-trigger', nargs='*', type=str,
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
    parser.add_argument('--finetune-mode', type=str, nargs='*', default=[],
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
    parser.add_argument('--linear', action='store_true', help='Enables using linear combo MLM')

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

    # Sparse loss params
    parser.add_argument('--l1decay', type=float, default=0.0,
                        help='L1 regularization weight (if using linear combination MLM)')
    parser.add_argument('--theta', type=float, default=1e32,
                        help='L1 regularization weight (if using linear combination MLM)')

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
