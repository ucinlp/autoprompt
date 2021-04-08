"""Discrete triggers for MLM prompting."""
import argparse
import logging
import os
import random

import torch
import transformers
from tqdm import tqdm

from autoprompt.evaluators import MLM_EVALUATORS
from autoprompt.metrics import METRICS
from autoprompt.models import DiscreteTriggerMLM
from autoprompt.preprocessors import PREPROCESSORS
from autoprompt.trainers import Trainer
import autoprompt.data as data
import autoprompt.templatizers as templatizers
import autoprompt.utils as utils


logger = logging.getLogger(__name__)


def get_cinf_optimizer(model, args):
    """Handles setting the optimizer up for different finetuning modes."""
    params = []
    if args['finetune_mode'] == 'all':
        params.append({
            'params': model.parameters(),
            'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
        })
    elif args['finetune_mode'] == 'partial':
        params.append({
            'params': model.lm_head.parameters(),
            'lr': args['finetune_lr'] if args['finetune_lr'] else args['lr']
        })
    return transformers.AdamW(
        params,
        lr=args['lr'],
        weight_decay=1e-2,
        eps=1e-8
    )


# TODO(rloganiv): Right now this just collects gradients and outputs candidates.  If we really
# wanted to emulate torch's optimizers this should also replace trigger tokens, but because of the
# interleaved candidate evaluation steps this can't be cleanly done without having the optimizer
# consume data from the iterator.
class DiscreteOptimizer:
    """Optimizes discrete prompts."""
    def __init__(
        self,
        model,
        num_candidates,
        candidate_filter=None,
    ):
        self._model = model
        self._num_candidates = num_candidates
        self._filter = candidate_filter

        self._stored_gradient = None
        self._total_gradient = None
        model.word_embeddings.register_backward_hook(self._backwards_hook)

    def _backwards_hook(self, module, grad_in, grad_out):
        # pylint: disable=missing-function-docstring,unused-argument
        self._stored_gradient = grad_out[0]

    def update(self, trigger_mask):
        """Update the prompt gradients.

        Should be run in conjunction w/ model.backward().
        """
        bsz = trigger_mask.size(0)
        num_trigger_tokens = self._model.trigger_ids.size(0)
        trigger_grad = self._stored_gradient[trigger_mask]
        trigger_grad = trigger_grad.view(bsz, num_trigger_tokens, -1)
        if self._total_gradient is None:
            self._total_gradient = trigger_grad.sum(dim=0)
        else:
            self._total_gradient += trigger_grad.sum(dim=0)

    def get_candidates(self, idx):
        """Get candidates for trigger token replacement."""
        with torch.no_grad():
            gradient_dot_embedding_matrix = torch.matmul(
                self._model.word_embeddings.weight,
                self._total_gradient[idx],
            )
            if self._filter is not None:
                gradient_dot_embedding_matrix -= self._filter
            gradient_dot_embedding_matrix *= -1
            _, top_k_ids = gradient_dot_embedding_matrix.topk(self._num_candidates)
        return top_k_ids

    def reset(self):
        """Resets the discrete optimizer.

        Should be run in conjunction w/ model.zero_grad()
        """
        self._stored_gradient = None
        self._total_gradient = None


class DiscreteMLMTrainer(Trainer):
    def train(self, train_loader, dev_loader):

        # TODO(rloganiv): This is lazy.
        args = self.args

        # Setup model
        logger.info('Initializing model.')
        base_model = transformers.AutoModelForMaskedLM.from_pretrained(args['model_name'],
                                                                       config=self.config)
        initial_trigger_ids = utils.get_initial_trigger_ids(args['initial_trigger'], self.tokenizer)
        if initial_trigger_ids is None:
            initial_trigger_ids = torch.full(
                (self.templatizer.num_trigger_tokens,),
                fill_value=self.tokenizer.mask_token_id,
                dtype=torch.int64,
            )
        model = DiscreteTriggerMLM(base_model, initial_trigger_ids)
        model.to(self.distributed_config.device)

        # Restore existing checkpoint if available.
        ckpt_path = os.path.join(args['ckpt_dir'], 'pytorch_model.bin')
        if os.path.exists(ckpt_path) and not args['force_overwrite']:
            logger.info('Restoring checkpoint.')
            state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
            model.load_state_dict(state_dict)

        # Setup optimizers
        if args['finetune_mode'] != 'trigger':
            cinf_optimizer = get_cinf_optimizer(model, args)
        else:
            cinf_optimizer = None
        discrete_optimizer = DiscreteOptimizer(model, args['num_candidates'])

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

        # The multi-candidate evaluation does not play nicely with having a
        # single metric, so we instantiate them whenever we need them instead
        # of carrying just one around.
        metric_cls = METRICS[args['evaluation_metric']]

        best_score = 0
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
                iter_ = iter(iter_)
                total_loss = torch.tensor(0.0, device=self.distributed_config.device)
                denom = torch.tensor(0.0, device=self.distributed_config.device)

                if cinf_optimizer is not None:
                    cinf_optimizer.zero_grad()
                discrete_optimizer.reset()

                running = True
                while running:
                    # Get candidates
                    logger.debug('Measuring loss...')
                    metric = metric_cls(label_map=self.label_map)
                    for _ in range(args['accumulation_steps']):
                        try:
                            model_inputs, labels = next(iter_)
                        except StopIteration:
                            running = False
                            break
                        else:
                            model_inputs = utils.to_device(model_inputs, self.distributed_config.device)
                            labels = utils.to_device(labels, self.distributed_config.device)
                            loss, *_ = evaluator(
                                model_inputs,
                                labels,
                                metric=metric,
                                train=True,
                            )
                            loss /= args['accumulation_steps']
                            loss.backward()
                            discrete_optimizer.update(model_inputs['trigger_mask'])
                    # If terminated early just break
                    if not running:
                        logger.debug('Breaking early...')
                        break
                    update_idx = random.randrange(self.templatizer.num_trigger_tokens)
                    candidates = discrete_optimizer.get_candidates(update_idx)

                    # Evaluate candidates
                    logger.debug('Evaluating candidates...')
                    current_metric = metric_cls(label_map=self.label_map)
                    candidate_metrics = [metric_cls(label_map=self.label_map) for _ in range(args['num_candidates'])]
                    denom = torch.tensor(0.0, device=self.distributed_config.device)
                    for _ in range(args['accumulation_steps']):
                        try:
                            model_inputs, labels = next(iter_)
                        except StopIteration:
                            running = False
                            break
                        else:
                            model_inputs = utils.to_device(model_inputs, self.distributed_config.device)
                            labels = utils.to_device(labels, self.distributed_config.device)
                            batch_size = 1.0 if args['evaluation_strategy'] == 'multiple-choice' else labels.size(0)
                            denom += batch_size

                            # Current metric update
                            with torch.no_grad():
                                evaluator(
                                    model_inputs,
                                    labels,
                                    metric=current_metric,
                                    train=True,
                                )

                            # Candidate metrics update
                            for candidate, candidate_metric in zip(candidates, candidate_metrics): 
                                with torch.no_grad():
                                    candidate_trigger_ids = model.trigger_ids.clone()
                                    candidate_trigger_ids[update_idx] = candidate
                                    evaluator(
                                        model_inputs,
                                        labels,
                                        metric=candidate_metric,
                                        train=True,
                                        trigger_ids=candidate_trigger_ids,
                                    )

                    # If terminated early just break
                    if not running:
                        logger.debug('Breaking early...')
                        break

                    # TODO(rloganiv): Make a method to clean this up.
                    current_score = torch.tensor(
                        current_metric.get()[current_metric.score_key],
                        device=self.distributed_config.device,
                    )
                    candidate_scores = torch.tensor(
                        [c.get()[c.score_key] for c in candidate_metrics],
                        device=self.distributed_config.device,
                    )
                    best_candidate_score = candidate_scores.max()
                    best_candidate_idx = candidate_scores.argmax()
                    if best_candidate_score > current_score:
                        logger.debug('Better candidate detected.')
                        new_trigger_ids = model.trigger_ids.clone()
                        replacement_token = candidates[best_candidate_idx]
                        new_trigger_ids[update_idx] = replacement_token
                        logger.debug(f'New trigger: {new_trigger_ids}')
                        model.trigger_ids = new_trigger_ids

                    # Step continuous optimizer
                    if cinf_optimizer is not None:
                        logger.debug('Optimizer step.')
                        if args['clip'] is not None:
                            torch.nn.utils.clip_grad_norm_(model.parameters(), args['clip'])
                        cinf_optimizer.step()

                    # Make sure to reset the gradient accumulation.
                    model.zero_grad()
                    discrete_optimizer.reset()
                
                if not args['skip_eval']:
                    logger.info('Evaluating...')
                    model.eval()
                    total_loss = torch.tensor(0.0, device=self.distributed_config.device)
                    denom = torch.tensor(0.0, device=self.distributed_config.device)
                    metric = metric_cls(label_map=self.label_map)

                    if self.distributed_config.is_main_process and not args['quiet']:
                        iter_ = tqdm(dev_loader)
                    else:
                        iter_ = dev_loader
                    with torch.no_grad():
                        for model_inputs, labels in iter_:
                            model_inputs = utils.to_device(model_inputs, self.distributed_config.device)
                            labels = utils.to_device(labels, self.distributed_config.device)
                            loss, *_ = evaluator(
                                model_inputs,
                                labels,
                                metric=metric,
                                train=False,
                            )
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
                        self.writer.add_scalar('Loss/dev', (total_loss / (denom + 1e-13)).item(), epoch)
                        metric_dict = metric.get()
                        for key, value in metric_dict.items():
                            self.writer.add_scalar(f'{key.capitalize()}/train', value, epoch)

                        score = metric_dict[metric.score_key]
                        if score > best_score:
                            logger.info('Best performance so far.')
                            best_score = score
                            best_metric_dict = metric_dict
                            if self.distributed_config.world_size != -1:
                                state_dict = model.module.state_dict()
                            else:
                                state_dict = model.state_dict()
                            if self.distributed_config.is_main_process:
                                torch.save(state_dict, ckpt_path)
                            self.tokenizer.save_pretrained(args['ckpt_dir'])
                            self.config.save_pretrained(args['ckpt_dir'])

            if os.path.exists(ckpt_path) and not args['skip_eval']:
                logger.info('Restoring checkpoint.')
                state_dict = torch.load(ckpt_path, map_location=self.distributed_config.device)
                model.load_state_dict(state_dict)

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
                    _, preds = evaluator(
                        model_inputs,
                        labels,
                        metric=metric,
                        train=False,
                    )

                    # Serialize output
                    for pred in preds:
                        print(pred, file=f)

            if self.distributed_config.world_size != -1:
                metric.reduce()

            if args['tmp']:
                if os.path.exists(ckpt_path):
                    logger.info('Temporary mode enabled, deleting checkpoint.')
                    os.remove(ckpt_path)

            metric_dict = metric.get()
            for key, value in metric_dict.items():
                self.writer.add_scalar(f'{key.capitalize()}/test', value, 0)
            metric_string = ' '.join(f'{k}: {v:0.4f}' for k, v in metric_dict.items())
            logger.info(metric_string)

            return score.item()


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
    train_loader, dev_loader, test_loader, _ = data.load_datasets(
        args,
        templatizer=templatizer,
        distributed_config=distributed_config,
    )
    trainer = DiscreteMLMTrainer(
        args=args,
        config=config,
        tokenizer=tokenizer,
        templatizer=templatizer,
        label_map=label_map,
        distributed_config=distributed_config,
        writer=writer,
    )
    model, _ = trainer.train(train_loader, dev_loader)
    trainer.test(model, test_loader)


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
                        choices=['trigger', 'partial', 'all'],
                        help='Approach used for finetuning. Options: '
                             'trigger: Only triggers are tuned. '
                             'partial: Top model weights additionally tuned. '
                             'all: All model weights are tuned.')
    parser.add_argument('--evaluation-metric', type=str, default='accuracy',
                        choices=list(METRICS.keys()),
                        help='Evaluation metric to use.')
    parser.add_argument('--num-candidates', type=int, default=10,
                        help='Number of candidate trigger token replacements to consider each step.')

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

    args = vars(parser.parse_args())

    if args['debug']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
