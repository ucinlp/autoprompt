import argparse
import csv
import itertools
import logging
import pathlib
import statistics
import sys
import tempfile

import torch
import transformers
import yaml

import autoprompt.data as data
import autoprompt.templatizers as templatizers
import autoprompt.trainers as trainers
import autoprompt.utils as utils


logger = logging.getLogger(__name__)


def generate_args(proto_config):
    keys = proto_config['parameters'].keys()
    values = proto_config['parameters'].values()
    for instance in itertools.product(*values):
        parameters = dict(zip(keys, instance))
        yield {**proto_config['args'], **parameters}


def kfold(args, trainer_class, num_folds):
    utils.set_seed(args['seed'])
    utils.check_args(args)
    distributed_config = utils.distributed_setup(local_rank=-1)
    config = transformers.AutoConfig.from_pretrained(args['model_name'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args['model_name'],
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )
    label_map = utils.load_label_map(args['label_map'])
    templatizer = templatizers.MultiTokenTemplatizer(
        template=args['template'],
        tokenizer=tokenizer,
        label_field=args['label_field'],
        label_map=label_map,
        add_padding=args['add_padding'],
    )
    writer = utils.NullWriter()
    scores = []
    splits = data.generate_splits(
        args,
        num_folds,
        templatizer,
        distributed_config,
    )
    trainer = trainer_class(
        args=args,
        config=config,
        tokenizer=tokenizer,
        templatizer=templatizer,
        label_map=label_map,
        distributed_config=distributed_config,
        writer=writer,
    )
    for train_loader, dev_loader in splits:
        # Use a temporary directory for k-fold experiments to ensure proper cleanup.
        with tempfile.TemporaryDirectory() as tmpdir:
            args['ckpt_dir'] = tmpdir  # NOTE: This is implicitly modifying the trainer. Sorry.
            model, score = trainer.train(train_loader, dev_loader)
        scores.append(score)

        del model
        torch.cuda.empty_cache()

    return scores


def evaluate(args, trainer_class):
    # TODO(rloganiv): Instead of copying this over and over, replace w/ a more robust
    # config/initialization thingy.
    utils.set_seed(args['seed'])
    utils.check_args(args)
    distributed_config = utils.distributed_setup(local_rank=-1)
    config = transformers.AutoConfig.from_pretrained(args['model_name'])
    tokenizer = transformers.AutoTokenizer.from_pretrained(
        args['model_name'],
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )
    label_map = utils.load_label_map(args['label_map'])
    templatizer = templatizers.MultiTokenTemplatizer(
        template=args['template'],
        tokenizer=tokenizer,
        label_field=args['label_field'],
        label_map=label_map,
        add_padding=args['add_padding'],
    )
    writer = utils.NullWriter()
    scores = []

    train_loader, dev_loader, test_loader, _ = data.load_datasets(
        args,
        templatizer=templatizer,
        distributed_config=distributed_config
    )

    trainer = trainer_class(
        args=args,
        config=config,
        tokenizer=tokenizer,
        templatizer=templatizer,
        label_map=label_map,
        distributed_config=distributed_config,
        writer=writer
    )
    model, _ = trainer.train(train_loader, dev_loader)
    final_score = trainer.test(model, test_loader)
    return model, final_score


def main(args):
    logger.info('Loading jobs from: %s', args.input)
    with open(args.input, 'r') as f:
        proto_config = yaml.load(f, Loader=yaml.SafeLoader)
    if proto_config['trainer'] == 'continuous_mlm':
        trainer_class = trainers.ContinuousMLMTrainer
    elif proto_config['trainer'] == 'discrete_mlm':
        trainer_class = trainers.DiscreteMLMTrainer

    # K-Fold Step
    best_score = float('-inf')
    best_args = None
    # Serialize result
    with open(args.dir / 'cross_validation_results.csv', 'w') as f:
        # TODO(rloganiv): Is there a less ugly way of writing this?
        fieldnames = list(proto_config['args'].keys()) + \
            list(proto_config['parameters'].keys()) + \
            ['mean', 'stdev']
        writer = csv.DictWriter(f, fieldnames, delimiter=',')
        writer.writeheader()
        for train_args in generate_args(proto_config):
            # Train
            logger.debug(f'Args: {train_args}')
            scores = kfold(train_args, trainer_class, args.num_folds)
            mean = statistics.mean(scores)
            stdev = statistics.stdev(scores)
            logger.debug(f'Mean: {mean}, Std Dev: {stdev}')

            row = {k: v for k, v in train_args.items() if k in fieldnames}
            row['mean'] = mean
            row['stdev'] = stdev
            writer.writerow(row)

            # Update if best
            if mean > best_score:
                logger.debug('Best model so far.')
                best_score = mean
                best_args = train_args

    # Retrain and evaluate best model w/ multiple random seeds
    logger.info(f'Serializing Best config: {best_args}')
    config = proto_config.copy()
    config['args'] = best_args
    del config['parameters']
    with open(args.dir / 'best_config.yaml', 'w') as f:
        yaml.dump(config, f)

    logger.info('Evaluating')
    with open(args.dir / 'best_model_scores.csv', 'w') as f:
        writer = csv.DictWriter(f, fieldnames=['seed', 'score'])
        writer.writeheader()
        for i in range(args.num_seeds):
            best_args['seed'] = i
            model, score = evaluate(best_args, trainer_class)
            del model
            torch.cuda.empty_cache()
            logger.info(f'Score for seed {i}: {score}')
            writer.writerow({'seed': i, 'score': score})



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('input', type=pathlib.Path, help='JSONL file containing jobs.')
    parser.add_argument('-k', '--num_folds', type=int, default=4)
    parser.add_argument('-n', '--num_seeds', type=int, default=10)
    parser.add_argument('--logdir', type=pathlib.Path, default='results/')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('-f', '--force_overwrite', action='store_true')
    args = parser.parse_args()
    args.dir = args.logdir / args.input.stem

    if args.dir.exists() and not args.force_overwrite:
        raise RuntimeError(
            'A result directory already exists for this configuration file. Either the experiment '
            'has already been performed, or you are re-using a filename. If you really want to run '
            'this use the -f flag.'
        )

    # Setup logging
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    root_logger = logging.getLogger()
    root_logger.setLevel(logging.DEBUG)

    ch = logging.StreamHandler()
    ch.setLevel(logging.DEBUG if args.debug else logging.INFO)
    ch.setFormatter(formatter)
    root_logger.addHandler(ch)

    if not args.dir.exists():
        logging.debug(f'Creating directory: {args.dir}')
        args.dir.mkdir(parents=True)
    fh = logging.FileHandler(args.dir / 'debug.log', mode='w')
    fh.setLevel(logging.DEBUG)
    fh.setFormatter(formatter)
    root_logger.addHandler(fh)

    main(args)
