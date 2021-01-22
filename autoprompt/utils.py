import csv
import collections
import json
import logging
import os
import random

import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers.modeling_albert import AlbertPreTrainedModel
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_roberta import RobertaPreTrainedModel

from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


def _get_special_ids(tokenizer):
    """Gets the ids of special [T] and [P] tokens."""
    trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    if trigger_token_id == tokenizer.unk_token_id:
        raise ValueError('Tokenizer does not have special [T] token.')
    predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    if predict_token_id == tokenizer.unk_token_id:
        raise ValueError('Tokenizer does not have special [P] token.')
    return trigger_token_id, predict_token_id


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        self._stored_output = output

    def get(self):
        return self._stored_output


class ExponentialMovingAverage:
    def __init__(self, weight=0.3):
        self._weight = weight
        self.reset()

    def update(self, x):
        self._x += x
        self._i += 1

    def reset(self):
        self._x = 0
        self._i = 0

    def get_metric(self):
        return self._x  / (self._i + 1e-13)


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


def encode_label(tokenizer, label, tokenize=False):
    """
    Helper function for encoding labels. Deals with the subtleties of handling
    multiple labels/tokens.
    """
    if isinstance(label, str):
        if tokenize:
            # Ensure label is properly tokenized, and only retain first token
            # if it gets split into multiple tokens.
            # TODO: Make sure this is desired behavior.
            tokens = tokenizer.tokenize(label, add_prefix_space=True)
            if len(tokens) > 1:
                logger.warning('Label "%s" gets split into multiple tokens: %s', label, tokens)
            if tokens[0] == tokenizer.unk_token:
                raise ValueError(f'Label "{label}" gets mapped to unk.')
            label = tokens[0]
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
    elif isinstance(label, int):
        encoded = torch.tensor([[label]])
    return encoded


class TriggerTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    add_special_tokens : bool
        Whether or not to add special tokens when encoding. Default: False.
    """
    def __init__(self,
                 template,
                 tokenizer,
                 label_field='label',
                 label_map=None,
                 tokenize_labels=False):
        self._template = template
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._tokenize_labels = tokenize_labels

        trigger_token_id, predict_token_id = _get_special_ids(tokenizer)
        self._trigger_token_id = trigger_token_id
        self._predict_token_id = predict_token_id

    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    def __call__(self, format_kwargs, **kwargs):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        label = format_kwargs.pop(self._label_field)
        text = self._template.format(**format_kwargs)
        if label is None:
            raise Exception(f'Bad data: {text}')

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt'
        )
        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._trigger_token_id)
        predict_mask = input_ids.eq(self._predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        # Encode the label(s)
        if self._label_map is not None:
            label = self._label_map[label]
        label_id = encode_label(
            tokenizer=self._tokenizer,
            label=label,
            tokenize=self._tokenize_labels
        )

        return model_inputs, label_id


class MultiTokenTemplatizer:
    """
    An object to facilitate creating transformers-friendly triggers inputs from
    a template.

    Parameters
    ==========
    template : str
        The template string, comprised of the following tokens:
            [T] to mark a trigger placeholder.
            [P] to mark a prediction placeholder.
            {fields} arbitrary fields instantiated from the dataset instances.
        For example a NLI template might look like:
            "[T] [T] [T] {premise} [P] {hypothesis}"
    tokenizer : PretrainedTokenizer
        A HuggingFace tokenizer. Must have special trigger and predict tokens.
    label_field : str
        The field in the input dictionary that corresponds to the label.
    """
    def __init__(
        self,
        template,
        tokenizer,
        label_field='label',
        label_map=None,
        add_padding=False,
    ):
        self._template = template
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._add_padding = add_padding

        trigger_token_id, predict_token_id = _get_special_ids(tokenizer)
        self._trigger_token_id = trigger_token_id
        self._predict_token_id = predict_token_id

        if label_map is not None:
            logger.debug(
                'Label map (detokenized): %s',
                {k: tokenizer.decode(tokenizer.encode(v, add_special_tokens=False)) for k, v in label_map.items()}
            )

    # TODO(rloganiv): If there is any more shared code between tokenizers,
    # consider creating a base class.
    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id

    def _maybe_truncate(self, format_kwargs, approximate_length):
        """
        If instantiated template would exceed maximum sequence length then
        reduce the input sizes.
        """
        format_kwargs = format_kwargs.copy()
        budget = self._tokenizer.model_max_length
        budget -= approximate_length
        budget -= self.num_trigger_tokens
        while True:
            field_lengths = {k: len(self._tokenizer.encode(v, add_special_tokens=False)) for k, v in format_kwargs.items()}
            instance_length = sum(field_lengths.values())
            gap = budget - instance_length
            if gap < 0:
                longest_field = max(field_lengths.items(), key=lambda x: x[1])[0]
                encoded = self._tokenizer.encode(format_kwargs[longest_field], add_special_tokens=False)
                truncated = encoded[:(gap - 1)]
                format_kwargs[longest_field] = self._tokenizer.decode(truncated)
            else:
                break
        return format_kwargs

    def __call__(self, format_kwargs, train=False, **kwargs):
        """
        Combines the template with instance specific inputs.

        Parameters
        ==========
        format_kwargs : Dict[str, str]
            A dictionary whose keys correspond to the fields in the template,
            and whose values are the strings that should instantiate those
            fields. The dictionary must contain the label as well (which
            requires special processing).
        """
        # Tokenize label
        label = format_kwargs.pop(self._label_field)
        if label is None:
            raise Exception(
                f'No label detected for instance: {format_kwargs}.'
                f'Double check that label field is correct: {self._label_field}.'
            )
        if self._label_map is not None:
            label = self._label_map[label]
        label_tokens = self._tokenizer.encode(
            label,
            add_special_tokens=False,  # Don't want to add [CLS] mid-sentence
            return_tensors='pt',
        )
        label_size = label_tokens.size(1)

        # Add padding, by initializing a longer tensor of <pad> tokens and
        # replacing the fron with the label tokens. Magic numbers below come
        # from PET WSC settings.
        if self._add_padding:
            # CHECK BACK ON THIS; PROBABLY REMOVE.
            # if 'padded_label_size' in kwargs:  # Manually set padding for multiple-choice.
                # padded_label_size = kwargs['padded_label_size']
            # else:
            if train:
                pad_length = random.randint(0, 3)
            else:
                pad_length = 1
            padded_label_size = label_size + pad_length
            padded_label_tokens = label_tokens.new_zeros(1, padded_label_size)
            # CHECK BACK ON THIS; PROBABLY REMOVE. For multiple choice training, input is padded w/
            # ignored mask tokens. I guess this is to avoid cheating by counting word tokens?
            # if 'ignore_padding' in kwargs:
                # pad_token_id = -100 if kwargs['ignore_padding'] else self._tokenizer.pad_token_id
            # else:
                # pad_token_id = self._tokenizer.pad_token_id
            padded_label_tokens.fill_(self._tokenizer.pad_token_id)
            padded_label_tokens[:,:label_size] = label_tokens
        else:
            padded_label_size = label_size
            padded_label_tokens = label_tokens

        # For sake of convenience we're just going to replace [P] with multiple
        # [P]s to help with the bookkeeping.
        template = self._template.replace(
            '[P]', 
            ' '.join(['[P]'] * padded_label_size),
        )
        approximate_length = self._tokenizer(
            template,
            add_special_tokens=True,
            return_length=True
        )['length']

        # Instantiate & tokenize the template
        format_kwargs = self._maybe_truncate(
            format_kwargs,
            approximate_length=approximate_length,
        )
        text = template.format(**format_kwargs)
        model_inputs = self._tokenizer(
            text,
            add_special_tokens=True,
            return_tensors='pt',
        )
        input_ids = model_inputs['input_ids']

        # Trigger & predict token bookkeeping. Unlike in other templatizer,
        # triggers are replaced by [MASK] tokens by default. This is to avoid
        # unnecc. post-processing for continuous triggers (using OOV tokens
        # during the forward pass is the recipe for a serious headache).
        # TODO: Consider making this the default behavior across templatizers.
        trigger_mask = input_ids.eq(self._trigger_token_id)
        input_ids[trigger_mask] = self._tokenizer.mask_token_id
        predict_mask = input_ids.eq(self._predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        # EXPERIMENTAL: Handle sep mask.
        if 'token_type_ids' in model_inputs:
            sep_mask = input_ids.eq(self._tokenizer.sep_token_id)
            model_inputs['token_type_ids'][:,1:] = torch.cumsum(
                sep_mask,
                dim=-1,
                dtype=torch.long
            )[:,:-1]

        # For sake of convenience, we're going to use HuggingFace's built-in
        # loss computation for computing cross-entropy. See the description of
        # the `labels` argument here for more details:
        #   https://huggingface.co/transformers/glossary.html#labels
        labels = torch.zeros_like(model_inputs['input_ids'])
        labels.fill_(-100)  # -100 is the default "ignore" value
        labels[predict_mask] = padded_label_tokens

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        return model_inputs, labels


class MultipleChoiceDataset(torch.utils.data.IterableDataset):
    """
    Dataset for multiple choice style problems.
    
    Since I am feeling lazy and large inputs aren't really supported anyways, this yields (pos,neg)
    pairs one at a time during training, and
    """
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
        if preprocessor_key is None:
            raise ValueError('MultipleChoiceDataset cannot use default preprocessors.')
        preprocessor = PREPROCESSORS[preprocessor_key]
        instances = list(preprocessor(fname, train=train))
        return cls(instances, templatizer, limit, train)

    def __iter__(self):
        instances = self._instances
        if self._train:
            random.shuffle(instances)  # WARNING: Side effects.
        if self._limit is not None:
            limit = min(self._limit, len(instances))
            instances = instances[:limit]
        for instance in instances:
            instance = instance.copy()  # Non destructive.
            labels = instance.pop('labels')
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
                labels.sort(key=lambda x: x[1], reverse=True)
                for label, is_positive in labels:
                    sub_instance = instance.copy()
                    sub_instance['label'] = label
                    output = self._templatizer(sub_instance)
                    outputs.append(output)
                yield outputs


class GenerativeDataset(torch.utils.data.IterableDataset):
    """
    Dataset for generative style problems.
    """
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
        if preprocessor_key is None:
            raise ValueError('MultipleChoiceDataset cannot use default preprocessors.')
        preprocessor = PREPROCESSORS[preprocessor_key]
        instances = list(preprocessor(fname, train=train))
        return cls(instances, templatizer, limit, train)

    def __iter__(self):
        instances = self._instances
        if self._train:
            random.shuffle(instances)  # WARNING: Side effects.
        # TODO: This is not right. Will artificially increase dataset size.
        if self._limit is not None:
            limit = min(self._limit, len(instances))
            instances = instances[:limit]
        for instance in instances:
            instance = instance.copy()
            yield self._templatizer(instance, train=self._train)


def load_trigger_dataset(
    fname,
    templatizer,
    limit=None,
    train=False,
    preprocessor_key=None,
):
    if preprocessor_key is None:
        preprocessor = PREPROCESSORS[fname.split('.')[-1]]
    else:
        preprocessor = PREPROCESSORS[preprocessor_key]
    instances = []
    for x in preprocessor(fname, train=train):
        try:
            model_inputs, label_id = templatizer(x, train=train)
        except ValueError as e:
            logger.warning('Encountered error "%s" when processing "%s".  Skipping.', e, x)
            continue
        else:
            instances.append((model_inputs, label_id))
    if limit:
        limit = min(len(instances), limit)
        return random.sample(instances, limit)
    else:
        return instances


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
    Loads a dataset for classification

    Parameters
    ==========
    tokenizer : transformers.PretrainedTokenizer
        Maps text to id tensors.
    sentence1 :
    """
    instances = []
    label_map = label_map or {}
    if preprocessor_key is None:
        preprocessor = PREPROCESSORS[fname.suffix]
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


DATASET_CONSTRUCTORS = {
    'classification': load_trigger_dataset,
    'generative': GenerativeDataset.load,
    'multiple-choice': MultipleChoiceDataset.load,
}


def get_word_embeddings(model):
    for module in model.modules():
        if hasattr(module, 'word_embeddings'):
            return module.word_embeddings


def get_lm_head(model):
    for module in model.modules():
        if hasattr(module, 'cls'):
            return module.cls
        elif hasattr(module, 'lm_head'):
            return module.lm_head
        elif hasattr(module, 'predictions'):
            return module.predictions
        else:
            raise NotImplementedError(
                'Unable to retrieve MLM head for model. You may need to add a special case to '
                '`utils.get_lm_head`.'
            )


def get_clf_head(model):
    for module in model.modules():
        if hasattr(module, 'classifier'):
            return module.classifier
        else:
            raise NotImplementedError(
                'Unable to retrieve classifier head for model. You may need to add a special case to '
                '`utils.get_clf_head`.'
            )


DistributedConfig = collections.namedtuple(
    'DistributedConfig',
    ['device', 'local_rank', 'world_size', 'is_main_process'],
)


def distributed_setup(local_rank):
    world_size = os.getenv('WORLD_SIZE', -1)
    if local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', local_rank)
        if world_size != -1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=local_rank,
                world_size=world_size,
            )
    is_main_process = local_rank in [-1, 0] or world_size == -1
    logger.info('Rank: %s - World Size: %s', local_rank, world_size)
    return DistributedConfig(device, local_rank, world_size, is_main_process)

