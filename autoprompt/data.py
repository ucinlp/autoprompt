"""Data loading utilities."""
import logging
import warnings
import random

import torch
from torch.nn.utils.rnn import pad_sequence
from torch.utils.data import DataLoader, IterableDataset

from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    if sequence[0].size() == (1,):  # Labels aren't a token sequence
        return torch.cat(sequence, dim=0)
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


class Collator:
    """
    Collates transformer outputs.
    """
    def __init__(self, pad_token_id=0):
        self._pad_token_id = pad_token_id

    def __call__(self, features):
        # Separate the list of inputs and labels
        model_inputs, labels = list(zip(*features))
        # Assume that all inputs have the same keys as the first
        proto_input = model_inputs[0]
        keys = list(proto_input.keys())
        padded_inputs = {}
        for key in keys:
            if key == 'input_ids':
                padding_value = self._pad_token_id
            else:
                padding_value = 0
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        labels = pad_squeeze_sequence(labels, batch_first=True, padding_value=-100)
        return padded_inputs, labels


# TODO(rloganiv): Better name? Consistency with other MLM datasets?
def load_classification_dataset(
    fname,
    tokenizer,
    input_field_a,
    input_field_b=None,
    label_field='label',
    label_map=None,
    limit=None,
    preprocessor_key=None,
):
    """
    Loads a sequence classification dataset.

    Parameters
    ==========
    tokenizer : transformers.PretrainedTokenizer
        Maps text to id tensors.
    sentence1 :
    """
    instances = []
    label_map = label_map or {}
    if preprocessor_key is None:
        preprocessor = PREPROCESSORS[fname.split('.')[-1]]
    else:
        preprocessor = PREPROCESSORS[preprocessor_key]
    for instance in preprocessor(fname):
        logger.debug(instance)
        model_inputs = tokenizer(
            instance[input_field_a],
            instance[input_field_b] if input_field_b else None,
            add_special_tokens=True,
            truncation=True,
            padding='max_length',
            max_length=64,  # TODO: Don't hardcode, base on # triggers.
            return_tensors='pt'
        )
        logger.debug(model_inputs)
        label = instance[label_field]
        if label not in label_map:
            label_map[label] = len(label_map)
        label_id = label_map[label]
        label_id = torch.tensor([[label_id]])  # To make collator expectation
        logger.debug(f'Label id: {label_id}')
        instances.append((model_inputs, label_id))
    if limit:
        limit = min(len(instances), limit)
        instances = random.sample(instances, limit)
    return instances, label_map


def prime(model_inputs, label_id, priming_dataset, model_max_length):

    to_concat = {k: [v] for k, v in model_inputs.items()}

    current_length = model_inputs['input_ids'].size(1)

    for prepended_inputs, label_id_ in priming_dataset:


        new_length = current_length + prepended_inputs['input_ids'].size(1)

        if new_length < model_max_length:

            # Clone prepended inputs so that we don't accidentally overwrite
            # data we care about.
            prepended_inputs = {k: v.clone() for k, v in prepended_inputs.items()}
            
            # Now fill in label and remove predict mask
            # TODO(rloganiv): Does something bad happen if we have multiple
            # trigger tokens? Or is there broadcast magic?
            input_ids = prepended_inputs['input_ids']
            predict_mask = prepended_inputs['predict_mask']
            input_ids[predict_mask] = label_id_[predict_mask]
            predict_mask.zero_()

            for k in prepended_inputs:
                to_concat[k].insert(0, prepended_inputs[k])

            current_length = new_length

        else:
            break

    model_inputs = {k: torch.cat(v, dim=1) for k, v in to_concat.items()}
    new_label_id = torch.zeros_like(model_inputs['input_ids']).fill_(-100)
    new_label_id[:,-label_id.size(1):] = label_id

    return model_inputs, new_label_id


def load_trigger_dataset(
    fname,
    templatizer,
    limit=None,
    train=False,
    preprocessor_key=None,
    priming_dataset=None,
):
    """
    Loads a MLM classification dataset.

    Parameters
    ==========
    fname : str
        The filename.
    templatizer : Templatizer
        Maps instances to cloze-style model inputs.
    limit : int
        (optional) Limit the amount of data loaded.
    train : bool
        Whether the data is used for training. Default: False.
    preprocessor_key : str
        Key used to lookup preprocessor for data.
    """
    if preprocessor_key is None:
        preprocessor = PREPROCESSORS[fname.split('.')[-1]]
    else:
        preprocessor = PREPROCESSORS[preprocessor_key]
    instances = []
    for x in preprocessor(fname, train=train):
        try:
            model_inputs, label_id = templatizer(x, train=train)

            if priming_dataset is not None:
                model_inputs, label_id = prime(
                    model_inputs,
                    label_id,
                    priming_dataset,
                    model_max_length=templatizer._tokenizer.model_max_length,
                )

        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            instances.append((model_inputs, label_id))
    if limit:
        limit = min(len(instances), limit)
        return random.sample(instances, limit)
    return instances


class MultipleChoiceDataset(IterableDataset):
    """
    Dataset for multiple choice style problems.

    Since I am feeling lazy and large inputs aren't really supported anyways, this yields (pos,neg)
    pairs one at a time during training, and
    """
    # pylint: disable=abstract-method
    def __init__(
            self,
            instances,
            templatizer,
            limit=None,
            train=False,
    ):
        self._instances = instances
        self._templatizer = templatizer
        self._limit = limit
        self._train = train

    @classmethod
    def load(
            cls,
            fname,
            templatizer,
            limit=None,
            train=False,
            preprocessor_key=None,
    ):
        """
        Loads the dataset from a file.

        Parameters
        ==========
        fname : str
            The filename.
        templatizer : Templatizer
            Maps instances to cloze-style model inputs.
        limit : int
            (optional) Limit the amount of data loaded.
        train : bool
            Whether the data is used for training. Default: False.
        preprocessor_key : str
            Key used to lookup preprocessor for data.
        """
        if limit is not None:
            raise NotImplementedError('Limit not supported for MultipleChoiceDataset.')
        if preprocessor_key is None:
            raise ValueError('MultipleChoiceDataset cannot use default preprocessors.')
        preprocessor = PREPROCESSORS[preprocessor_key]
        instances = list(preprocessor(fname, train=train))
        return cls(instances, templatizer, limit, train)

    def __iter__(self):
        instances = self._instances
        if self._train:
            random.shuffle(instances)  # WARNING: Side effects.
        for instance in instances:
            instance = instance.copy()  # Non destructive.
            labels = instance.pop('labels')
            labels.sort(key=lambda x: x[1], reverse=True)
            if self._train:
                positive_labels = [label for label, is_positive in labels if is_positive]
                positive_label = random.choice(positive_labels)
                positive_instance = instance.copy()
                positive_instance['label'] = positive_label
                positive_output = self._templatizer(positive_instance, train=True)

                negative_labels = [label for label, is_positive in labels if not is_positive]
                negative_label = random.choice(negative_labels)
                negative_instance = instance.copy()
                negative_instance['label'] = negative_label
                negative_output = self._templatizer(negative_instance, train=True)
                yield positive_output, negative_output

            else:
                outputs = []
                for label, _ in labels:
                    sub_instance = instance.copy()
                    sub_instance['label'] = label
                    output = self._templatizer(sub_instance)
                    outputs.append(output)
                yield outputs


class GenerativeDataset(IterableDataset):
    """
    Dataset for generative style problems.
    """
    # pylint: disable=abstract-method
    def __init__(
        self,
        instances,
        templatizer,
        limit=None,
        train=False,
    ):
        self._instances = instances
        self._templatizer = templatizer
        self._limit = limit
        self._train = train

    @classmethod
    def load(
        cls,
        fname,
        templatizer,
        limit=None,
        train=False,
        preprocessor_key=None,
    ):
        """
        Loads the dataset from a file.

        Parameters
        ==========
        fname : str
            The filename.
        templatizer : Templatizer
            Maps instances to cloze-style model inputs.
        limit : int
            (optional) Limit the amount of data loaded.
        train : bool
            Whether the data is used for training. Default: False.
        preprocessor_key : str
            Key used to lookup preprocessor for data.
        """
        if limit is not None:
            raise NotImplementedError('Limit not supported for generative datasets.')
        if preprocessor_key is None:
            raise ValueError('MultipleChoiceDataset cannot use default preprocessors.')
        preprocessor = PREPROCESSORS[preprocessor_key]
        instances = list(preprocessor(fname, train=train))
        return cls(instances, templatizer, limit, train)

    def __iter__(self):
        instances = self._instances
        if self._train:
            random.shuffle(instances)  # WARNING: Side effects.
        for instance in instances:
            instance = instance.copy()
            yield self._templatizer(instance, train=self._train)


DATASET_CONSTRUCTORS = {
    'classification': load_trigger_dataset,
    'generative': GenerativeDataset.load,
    'multiple-choice': MultipleChoiceDataset.load,
}


def get_sampler(
    dataset,
    evaluation_strategy,
    distributed_config,
    train=False,
):
    """Sets up the metrics sampler for a data loader."""
    # Sampling is handled by data iterator for multiple choice problems.
    if evaluation_strategy != 'classification':
        return None
    # Multi-GPU training
    if distributed_config.world_size != -1:
        return torch.utils.data.DistributedSampler(dataset, shuffle=train)
    # Defaults
    if train:
        return torch.utils.data.RandomSampler(dataset)
    return torch.utils.data.SequentialSampler(dataset)


def generate_splits(args, num_folds, templatizer, distributed_config):
    # TODO(rloganiv): Need to work something out for multiple choice...
    if args['evaluation_strategy'] != 'classification':
        raise NotImplementedError('Ugh...')

    assert num_folds > 0, 'Number of folds must be a positive integer.'

    dataset_constructor = DATASET_CONSTRUCTORS[args['evaluation_strategy']]
    collator = Collator(pad_token_id=templatizer.pad_token_id)

    train_dataset = dataset_constructor(
        args['train'],
        templatizer=templatizer,
        train=True,
        preprocessor_key=args['preprocessor'],
        limit=args['limit'],
    )
    dev_dataset = dataset_constructor(
        args['dev'],
        templatizer=templatizer,
        preprocessor_key=args['preprocessor'],
        limit=args['limit'],
    )
    combined = train_dataset + dev_dataset
    logger.debug(f'Combined Size: {len(combined)}')
    chunk_size = len(combined) // num_folds
    if len(combined) % num_folds != 0:
        warnings.warn(
            'Folds do not evenly divide data. Ignoring last incomplete fold.',
            RuntimeWarning
        )
        num_folds -= 1

    for k in range(num_folds):
        train_dataset = combined[:k*chunk_size] + combined[(k+1)*chunk_size:]
        train_sampler = get_sampler(train_dataset, args['evaluation_strategy'], distributed_config, train=True)
        train_loader = DataLoader(train_dataset, batch_size=args['bsz'], collate_fn=collator, sampler=train_sampler)
        
        args['train_size'] = len(train_dataset)  # TODO(rloganiv): This is terrible

        dev_dataset = combined[k*chunk_size:(k+1)*chunk_size]
        dev_sampler = get_sampler(dev_dataset, args['evaluation_strategy'], distributed_config, train=False)
        dev_loader = DataLoader(dev_dataset, batch_size=args['bsz'], collate_fn=collator, sampler=dev_sampler)

        logger.debug(f'K: {k}, Train size: {len(train_dataset)}, Dev size: {len(dev_dataset)}')

        yield train_loader, dev_loader


def load_datasets(args, templatizer, distributed_config):
    """Loads the training, dev and test datasets."""
    dataset_constructor = DATASET_CONSTRUCTORS[args['evaluation_strategy']]
    collator = Collator(pad_token_id=templatizer.pad_token_id)

    train_dataset = dataset_constructor(
        args['train'],
        templatizer=templatizer,
        train=True,
        preprocessor_key=args['preprocessor'],
        limit=args['limit'],
    )
    train_sampler = get_sampler(train_dataset, args['evaluation_strategy'], distributed_config, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args['bsz'], collate_fn=collator, sampler=train_sampler)

    # TODO(rloganiv): Is there a cleaner way?
    args['train_size'] = len(train_dataset)

    dev_dataset = dataset_constructor(
        args['dev'],
        templatizer=templatizer,
        preprocessor_key=args['preprocessor'],
        limit=args['limit'],
    )
    dev_sampler = get_sampler(dev_dataset, args['evaluation_strategy'], distributed_config, train=False)
    dev_loader = DataLoader(dev_dataset, batch_size=args['bsz'], collate_fn=collator, sampler=dev_sampler)

    test_dataset = dataset_constructor(
        args['test'],
        templatizer=templatizer,
        preprocessor_key=args['preprocessor'],
        priming_dataset=train_dataset if args['prime'] else None
    )
    test_sampler = get_sampler(test_dataset, args['evaluation_strategy'], distributed_config, train=False)
    test_loader = DataLoader(test_dataset, batch_size=args['bsz'], collate_fn=collator, sampler=test_sampler)

    if args.get('checklist', False):
        checklist_test_dataset = dataset_constructor(
            args['checklist'],
            templatizer=templatizer,
            preprocessor_key=args['preprocessor'],
        )
        checklist_test_sampler = get_sampler(checklist_test_dataset, args['evaluation_strategy'], distributed_config, train=False)
        checklist_test_loader = DataLoader(checklist_test_dataset, batch_size=args['bsz'], collate_fn=collator, sampler=checklist_test_sampler)
    else:
        checklist_test_loader = None

    return train_loader, dev_loader, test_loader, checklist_test_loader


