import argparse
import logging
from pathlib import Path
import random

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelWithLMHead
from tqdm import tqdm

import autoprompt.utils as utils


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


def forward_w_triggers(model, model_inputs, labels):
    # Ensure destructive pop operations are only limited to this function.
    model_inputs = model_inputs.copy()
    trigger_mask = model_inputs.pop('trigger_mask')
    predict_mask = model_inputs.pop('predict_mask')
    input_ids = model_inputs.pop('input_ids')

    # Get embeddings of input sequence
    inputs_embeds = model.embeds(input_ids)
    inputs_embeds[trigger_mask] = model.relation_embeds.unsqueeze(0)
    model_inputs['inputs_embeds'] = inputs_embeds

    loss, logits, *_ = model(**model_inputs, labels=labels)

    # Get predictions
    # TODO(rloganiv): Add support for other decoding options. Currently only
    # parallel decoding is supported.
    preds = logits.argmax(dim=-1)

    # Compute correctness. Not sure if this should be moved out of forward or
    # not.

    return loss, preds


def main(args):
    logger.info("Dataset: %s" %str(args.train).split("/")[3])
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name, config=config)
    # TODO(rloganiv): See if transformers has a general API call for getting
    # the word embeddings. If so, then most of the below code can be 
    if args.model_name == "bert-base-cased":
        model.embeds = model.bert.embeddings.word_embeddings
    elif args.model_name == "roberta-base":
        model.embeds = model.roberta.embeddings.word_embeddings
    if not args.finetune:
        for param in model.parameters():
            param.requires_grad = False
    # TODO: Double check parameters get registered. Maybe it's better to make a
    # new Module so the tensor isn't unexpected upon reloading?
    model.relation_embeds = torch.nn.Parameter(torch.rand(args.trigger_length, 
                            model.embeds.weight.shape[1], requires_grad=True))
    model.to(device)

    templatizer = utils.MultiTokenTemplatizer(
        template=args.template,
        tokenizer=tokenizer,
        label_field=args.label_field,
    )
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(args.train, templatizer, train=True)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.load_trigger_dataset(args.dev, templatizer)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    test_dataset = utils.load_trigger_dataset(args.test, templatizer)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        avg_loss = utils.ExponentialMovingAverage()
        pbar = tqdm(train_loader)
        for model_inputs, labels in pbar:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            loss, logits = forward_w_triggers(model, model_inputs, labels)
            loss.backward()
            optimizer.step()
            avg_loss.update(loss.item())
            pbar.set_description(f'loss: {avg_loss.get_metric(): 0.4f}')

        logger.info('Evaluating...')
        model.eval()
        correct = 0
        total = 0
        for model_inputs, labels in dev_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            loss, preds = forward_w_triggers(model, model_inputs, labels)
            correct  
        accuracy = correct / (total + 1e-13)
        logger.info(f'Accuracy: {accuracy : 0.4f}')

        if accuracy > best_accuracy:
            logger.info('Best performance so far.')
            # torch.save(model.state_dict(), args.ckpt_dir / WEIGHTS_NAME)
            # model.config.to_json_file(args.ckpt_dir / CONFIG_NAME)
            # tokenizer.save_pretrained(args.ckpt_dir)
            best_accuracy = accuracy

    logger.info('Testing...')
    model.eval()
    correct = 0
    total = 0
    # TO DO: currently testing on last model, not best validation model
    for model_inputs, labels in test_loader:
        model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        mask_token_idxs = (model_inputs["input_ids"] == eos_idx).nonzero()[:,1] + args.trigger_length
        model_inputs = generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx)
        labels = labels.to(device)[:, 1]
        logits, *_ = model(**model_inputs)
        mask_logits = logits[torch.arange(0, logits.shape[0], dtype=torch.long), mask_token_idxs]
        preds = torch.topk(mask_logits, 1, dim=1).indices[:, 0]
        correct += (preds == labels).sum().item()
        total += labels.size(0)
    accuracy = correct / (total + 1e-13)
    logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str)
    parser.add_argument('--train', type=Path)
    parser.add_argument('--dev', type=Path)
    parser.add_argument('--test', type=Path)
    parser.add_argument('--template', type=int, default=5)
    parser.add_argument('--label-field', type=str, default='label')
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--ckpt-dir', type=Path, default=Path('ckpt/'))
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-f', '--force-overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)

