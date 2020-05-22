import csv
import json
import logging

from torch.nn.utils.rnn import pad_sequence


logger = logging.getLogger(__name__)


class TriggerCollator:
    """
    An object for collating outputs of TriggerTemplatizer
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
            sequence = [x[key].squeeze(0) for x in model_inputs]
            padded = pad_sequence(sequence, batch_first=True, padding_value=padding_value)
            padded_inputs[key] = padded
        return padded_inputs, labels


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
            return_tensors="pt"
        )
        input_ids = model_inputs['input_ids']
        trigger_mask = input_ids.eq(self._tokenizer.trigger_token_id)
        predict_mask = input_ids.eq(self._tokenizer.predict_token_id)
        input_ids[predict_mask] = self._tokenizer.mask_token_id

        model_inputs['trigger_mask'] = trigger_mask
        model_inputs['predict_mask'] = predict_mask

        return model_inputs, label


def add_task_specific_tokens(tokenizer):
    tokenizer.add_special_tokens({
        'additional_special_tokens': ['[T]', '[P]']
    })
    tokenizer.trigger_token = '[T]'
    tokenizer.trigger_token_id = tokenizer.convert_tokens_to_ids('[T]')
    tokenizer.predict_token = '[P]'
    tokenizer.predict_token_id = tokenizer.convert_tokens_to_ids('[P]')


def templatize_tsv(fname, templatizer):
    with open(fname, 'r') as f:
        reader = csv.DictReader(f, delimiter='\t')
        instances = [templatizer(row) for row in reader]
    return instances


def templatize_jsonl(fname, templatizer):
    with open(fname, 'r') as f:
        instances = [templatizer(json.loads(line)) for line in f]
    return instances


def load_dataset(path, templatizer):
    if path.suffix == '.tsv':
        return templatize_tsv(path, templatizer)
    elif path.suffix == '.jsonl':
        return templatize_jsonl(path, templatizer)
    else:
        raise ValueError(f'File "{path}" not supported. Currently supported formats: .tsv, .jsonl')
