import csv
import json
import logging
import random

import torch
from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


def pad_squeeze_sequence(sequence, *args, **kwargs):
    """Squeezes fake batch dimension added by tokenizer before padding sequence."""
    return pad_sequence([x.squeeze(0) for x in sequence], *args, **kwargs)


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
        self._x = None

    def update(self, x):
        if self._x:
            self._x = self._weight * x + (1 - self._weight) * self._x
        else:
            self._x = x

    def reset(self):
        self._x = None

    def get_metric(self):
        return self._x


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
            # NOTE: We need to squeeze to get rid of fake batch dim.
            sequence = [x[key] for x in model_inputs]
            padded = pad_squeeze_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        labels = pad_squeeze_sequence(labels, batch_first=True, padding_value=0)
        return padded_inputs, labels


def encode_label(tokenizer, label):
    """
    Helper function for encoding labels. Deals with the subtleties of handling multiple tokens.
    """
    if isinstance(label, str):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids([label])).unsqueeze(0)
        if encoded.eq(tokenizer.unk_token_id).any():
            # TODO: MAKE THIS SKIPPING UNK LABELS LOGIC BETTER
            # raise ValueError(f'Label "{label}" gets mapped to unk.')
            return None
    elif isinstance(label, list):
        encoded = torch.tensor(tokenizer.convert_tokens_to_ids(label)).unsqueeze(0)
        if encoded.eq(tokenizer.unk_token_id).any():
            raise ValueError(f'Label "{label}" gets mapped to unk.')
    # TODO: This is hacky.
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
                 add_special_tokens=False):
        if not hasattr(tokenizer, 'predict_token') or \
           not hasattr(tokenizer, 'trigger_token'):
            raise ValueError(
                'Tokenizer missing special trigger and predict tokens in vocab.'
                'Use `utils.add_special_tokens` to add them.'
            )
        self._template = template
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._add_special_tokens = add_special_tokens

    @property
    def num_trigger_tokens(self):
        return sum(token == '[T]' for token in self._template.split())

    def __call__(self, format_kwargs):
        # Format the template string
        format_kwargs = format_kwargs.copy()
        label = format_kwargs.pop(self._label_field)
        text = self._template.format(**format_kwargs)
        logger.debug(f'Formatted text: {text}')
        if label is None:
            raise Exception(f'Bad data: {text}')

        # Have the tokenizer encode the text and process the output to:
        # - Create a trigger and predict mask
        # - Replace the predict token with a mask token
        model_inputs = self._tokenizer.encode_plus(
            text,
            add_special_tokens=self._add_special_tokens,
            # add_prefix_space=True,
            return_tensors='pt'
        )
        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        # Encode the label(s)
        if self._label_map is not None:
            label = self._label_map[label]
        label_id = encode_label(self._tokenizer, label)

        return model_inputs, label_id


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')


def load_tsv(fname):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for row in reader:
            yield row


def load_jsonl(fname):
    with open(fname, 'r') as f:
        for line in f:
            yield json.loads(line)


LOADERS = {
    '.tsv': load_tsv,
    '.jsonl': load_jsonl
}


def load_trigger_dataset(fname, templatizer, limit=None):
    loader = LOADERS[fname.suffix]
    # TODO: MAKE THIS SKIPPING UNK LABELS LOGIC BETTER
    # instances = [templatizer(x) for x in loader(fname)]
    instances = []
    for x in loader(fname):
        t = templatizer(x)
        if t[1] != None:
            instances.append(t)

    if limit:
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
    limit=None
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
    loader = LOADERS[fname.suffix]
    for instance in loader(fname):
        logger.debug(instance)
        model_inputs = tokenizer.encode_plus(
            instance[input_field_a],
            instance[input_field_b] if input_field_b else None,
            add_special_tokens=True,
            add_prefix_space=True,
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
        instances = random.sample(instances, limit)
    return instances, label_map


def get_unique_objects(fname):
    """
    Return all the unique objects from a JSONL file of TREx triplets
    """
    # TODO: handle USE_CTX a.k.a. relation extraction
    unique_objs = set()
    samples = load_jsonl(fname)
    for sample in samples:
        unique_objs.add(sample['obj_label'].lower())
    return list(unique_objs)
