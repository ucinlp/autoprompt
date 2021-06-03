"""Used for converting task inputs into MLM prompts."""
# pylint: disable=missing-function-docstring

import logging
import random

import torch


logger = logging.getLogger(__name__)


def _encode_label(tokenizer, label, tokenize=False):
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


def _get_special_ids(tokenizer):
    """Gets the ids of special [T] and [P] tokens."""
    trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    if trigger_token_id == tokenizer.unk_token_id:
        raise ValueError('Tokenizer does not have special [T] token.')
    predict_token_id = tokenizer.convert_tokens_to_ids('[P]')
    if predict_token_id == tokenizer.unk_token_id:
        raise ValueError('Tokenizer does not have special [P] token.')
    return trigger_token_id, predict_token_id


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
        label_id = _encode_label(
            tokenizer=self._tokenizer,
            label=label,
            tokenize=self._tokenize_labels
        )

        return model_inputs, label_id


class FinetuneTemplatizer:
    """
    A wrapper around a transformers tokenizer to facilitate sequence
    classification w/out messing too much with the API.
    """
    def __init__(
        self,
        tokenizer,
        label_map,
        input_field_a,
        input_field_b=None,
        label_field='label',
    ):
        for v in label_map.values():
            assert isinstance(v, int)
            assert 0 <= v <= len(label_map)-1
        self._tokenizer = tokenizer
        self._input_field_a = input_field_a
        self._input_field_b = input_field_b
        self._label_field = label_field
        self._label_map = label_map

    @property
    def pad_token_id(self):
        return self._tokenizer.pad_token_id


    def __call__(self, format_kwargs, train=False, **kwargs):

        # Convert label to id
        label = format_kwargs.pop(self._label_field)
        if label not in self._label_map:
            self._label_map[label] = len(self._label_map)
        label_id = torch.tensor([self._label_map[label]])

        model_inputs = self._tokenizer(
            text=format_kwargs[self._input_field_a],
            text_pair=format_kwargs.get(self._input_field_b, None),
            add_special_tokens=True,
            return_tensors='pt',
            truncation='longest_first',
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
        randomize_mask=False,
    ):
        self._template = template
        self._tokenizer = tokenizer
        self._label_field = label_field
        self._label_map = label_map
        self._add_padding = add_padding
        self._randomize_mask = randomize_mask

        trigger_token_id, predict_token_id = _get_special_ids(tokenizer)
        self._trigger_token_id = trigger_token_id
        self._predict_token_id = predict_token_id

        if label_map is not None:
            logger.debug(
                'Label map (tokenized): %s',
                {k: tokenizer.encode(v, add_special_tokens=False) for k, v in label_map.items()}
            )
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
        if self._randomize_mask:
            input_ids[predict_mask] = random.randrange(1000, len(self._tokenizer))
        else:
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
