"""
Prints the learned trigger or its nearest neighbor approximation.
"""
import argparse
import json
from pathlib import Path

import torch

import autoprompt.models as models
import autoprompt.utils as utils
import autoprompt.templatizers as templatizers


MODEL_LOOKUP = {
    'continuous': models.ContinuousTriggerMLM,
    'discrete': models.DiscreteTriggerMLM,
}


def main(args):
    assert args.ckpt_dir.is_dir()
    with open(args.ckpt_dir / 'args.json', 'r') as f:
        original_args = json.load(f)
    config, tokenizer, base_model = utils.load_transformers(original_args['model_name'])
    label_map = utils.load_label_map(original_args['label_map'])
    templatizer = templatizers.MultiTokenTemplatizer(
        template=original_args['template'],
        tokenizer=tokenizer,
        label_field=original_args['label_field'],
        label_map=label_map,
        add_padding=original_args['add_padding'],
    )
    initial_trigger_ids = utils.get_initial_trigger_ids(original_args['initial_trigger'], tokenizer)
    model_kwargs = {'initial_trigger_ids': initial_trigger_ids}
    if args.model_type:
        model_kwargs['num_trigger_tokens'] = templatizer.num_trigger_tokens
    model = MODEL_LOOKUP[args.model_type](base_model, **model_kwargs)
    state_dict = torch.load(
        args.ckpt_dir / 'pytorch_model.bin',
        map_location='cpu',
    )
    model.load_state_dict(state_dict)

    if args.model_type == 'continuous':
        trigger_embeddings = model.trigger_embeddings
        word_embeddings = model.word_embeddings.weight
        scores = torch.mm(trigger_embeddings, word_embeddings.transpose(0, 1))
        trigger_ids = torch.argmax(scores, dim=-1)
    elif args.model_type == 'discrete':
        trigger_ids = model.trigger_ids
    print(f'Template: {original_args["template"]}')
    print(f'Tokens: {tokenizer.convert_ids_to_tokens(trigger_ids)}')

    



if  __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('ckpt_dir', type=Path, help='Checkpoint directory.')
    parser.add_argument('-t', '--model_type', type=str, required=True,
                        choices=list(MODEL_LOOKUP.keys()),
                        help=f'The type of the model.')

    args = parser.parse_args()

    main(args)

