"""
This is a hacky little attempt using the tools from the trigger creation script to identify a
good set of label strings. The idea is to train a linear classifier over the predict token and
then look at the most similar tokens.
"""
import argparse
import json
import logging
from pathlib import Path

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import BertForMaskedLM, RobertaForMaskedLM
from tqdm import tqdm

import autoprompt.data as data
import autoprompt.templatizers as templatizers
import autoprompt.utils as utils
import autoprompt.create_trigger as ct
from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


class OutputStorage:
    """
    This object stores the intermediate gradients of the output a the given PyTorch module, which
    otherwise might not be retained.
    """
    def __init__(self, module):
        self._stored_output = None
        module.register_forward_hook(self.hook)

    def hook(self, module, input, output):
        """Forward hook to be registered to the model."""
        # pylint: disable=redefined-builtin,unused-argument
        self._stored_output = output

    def get(self):
        """Gets the stored output."""
        return self._stored_output


def get_final_embeddings(model):
    """Gets final embeddings from model."""
    if isinstance(model, BertForMaskedLM):
        return model.cls.predictions.transform
    if isinstance(model, RobertaForMaskedLM):
        return model.lm_head.layer_norm
    raise NotImplementedError(f'{model} not currently supported')


def get_word_embeddings(model):
    """Gets word embeddings from model."""
    if isinstance(model, BertForMaskedLM):
        return model.cls.predictions.decoder.weight
    if isinstance(model, RobertaForMaskedLM):
        return model.lm_head.decoder.weight
    raise NotImplementedError(f'{model} not currently supported')


def main(args):
    # pylint: disable=C0116,E1121,R0912,R0915
    utils.set_seed(args['seed'])
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    logger.info('Loading model, tokenizer, etc.')
    config, tokenizer, model = utils.load_transformers(args['model_name'])
    model.to(device)
    final_embeddings = get_final_embeddings(model)
    embedding_storage = OutputStorage(final_embeddings)
    word_embeddings = get_word_embeddings(model)

    label_map = json.loads(args['label_map'])
    reverse_label_map = {y: x for x, y in label_map.items()}
    templatizer = templatizers.TriggerTemplatizer(
        args['template'],
        tokenizer,
        label_map=label_map,
        label_field=args['label_field'],
    )

    # The weights of this projection will help identify the best label words.
    projection = torch.nn.Linear(config.hidden_size, len(label_map))
    projection.to(device)

    # Obtain the initial trigger tokens and label mapping
    if args['initial_trigger']:
        trigger_ids = tokenizer.encode(
            args['initial_trigger'],
            add_special_tokens=False,
            add_prefix_space=True
        )
        assert len(trigger_ids) == templatizer.num_trigger_tokens
    else:
        trigger_ids = [tokenizer.mask_token_id] * templatizer.num_trigger_tokens
    trigger_ids = torch.tensor(trigger_ids, device=device).unsqueeze(0)

    logger.info('Loading datasets')
    collator = data.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = data.load_trigger_dataset(
        args['train'],
        templatizer,
        preprocessor_key=args['preprocessor'],
    )
    train_loader = DataLoader(train_dataset, batch_size=args['bsz'], shuffle=True, collate_fn=collator)

    optimizer = torch.optim.Adam(projection.parameters(), lr=args['lr'])

    scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
    scores = F.softmax(scores, dim=0)
    for i, row in enumerate(scores):
        _, top = row.topk(args['k'])
        decoded = tokenizer.convert_ids_to_tokens(top)
        logger.info(f"Top k for class {reverse_label_map[i]}: {', '.join(decoded)}")

    logger.info('Training')
    for i in range(args['iters']):
        pbar = tqdm(train_loader)
        for model_inputs, labels in pbar:
            optimizer.zero_grad()
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            trigger_mask = model_inputs.pop('trigger_mask')
            predict_mask = model_inputs.pop('predict_mask')
            model_inputs = ct.replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
            with torch.no_grad():
                model(**model_inputs)
            embeddings = embedding_storage.get()
            predict_embeddings = embeddings.masked_select(predict_mask.unsqueeze(-1)).view(embeddings.size(0), -1)
            logits = projection(predict_embeddings)
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            pbar.set_description(f'loss: {loss : 0.4f}')

        scores = torch.matmul(projection.weight, word_embeddings.transpose(0, 1))
        scores = F.softmax(scores, dim=0)
        for j, row in enumerate(scores):
            _, top = row.topk(args['k'])
            decoded = tokenizer.convert_ids_to_tokens(top)
            logger.info(f"Top k for class {reverse_label_map[j]}: {', '.join(decoded)}")

    out = {}
    for i, row in enumerate(scores):
        _, top = row.topk(args['k'])
        out[reverse_label_map[i]] = tokenizer.convert_ids_to_tokens(top)
    print(out)



if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', type=Path, required=True, help='Train data path')
    parser.add_argument('--template', type=str, help='Template string')
    parser.add_argument('--label-map', type=str, help='JSON object defining label map')
    parser.add_argument('--initial-trigger', type=str, default=None, help='Manual prompt')
    parser.add_argument('--label-field', type=str, default='label',
                        help='Name of the label field')
    parser.add_argument('--preprocessor', type=str, default=None,
                        choices=PREPROCESSORS.keys())
    parser.add_argument('--lr', type=float, default=3e-4, help='Learning rate')
    parser.add_argument('--k', type=int, default=5, help='Number of label tokens to print')
    parser.add_argument('--bsz', type=int, default=32, help='Batch size')
    parser.add_argument('--iters', type=int, default=500,
                        help='Number of iterations to run label search')
    parser.add_argument('--model-name', type=str, default='bert-base-cased',
                        help='Model name passed to HuggingFace AutoX classes.')
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--debug', action='store_true')
    args = vars(parser.parse_args())

    if args['debug']:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)
