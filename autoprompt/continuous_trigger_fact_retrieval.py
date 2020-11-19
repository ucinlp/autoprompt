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


def generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx):
    source_ids = model_inputs["input_ids"]
    source_mask = model_inputs["attention_mask"]
    eos_token_idxs = (source_ids == eos_idx).nonzero()[:,1]

    subject_embeds = model.embeds(source_ids)
    relation_embeds = model.relation_embeds.to(source_ids.device)
    
    inputs_embeds = torch.cat([torch.cat([sembedding[:eos_token_idxs[idx], :], relation_embeds,
                        model.embeds.weight[tokenizer.mask_token_id].unsqueeze(0), sembedding[
                        eos_token_idxs[idx]:, :]], dim=0).unsqueeze(0) for idx, sembedding in 
                        enumerate(subject_embeds)], dim=0)
    # TO DO: add full stop character after [MASK]?
    input_attention_mask = torch.cat([torch.ones((len(source_ids), relation_embeds.shape[0]+1), 
                                      dtype=torch.long).to(source_ids.device), source_mask], dim=1)
    return {"inputs_embeds": inputs_embeds, "attention_mask": input_attention_mask}


def main(args):
    logger.info("Dataset: %s" %str(args.train).split("/")[3])
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = AutoModelWithLMHead.from_pretrained(args.model_name, config=config)
    if args.model_name == "bert-base-cased":
        model.embeds = model.bert.embeddings.word_embeddings
        eos_idx = 102
        if not args.finetune:
            for param in model.bert.parameters():
                param.requires_grad = False
    elif args.model_name == "roberta-base":
        model.embeds = model.roberta.embeddings.word_embeddings
        eos_idx = tokenizer.eos_token_id
        if not args.finetune:
            for param in model.roberta.parameters():
                param.requires_grad = False
    if not args.finetune:
        for param in model.parameters():
            param.requires_grad = False
    model.relation_embeds = torch.nn.Parameter(torch.rand(args.trigger_length, 
                            model.embeds.weight.shape[1], requires_grad=True))
    model.to(device)

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_continuous_trigger_dataset(
        args.train,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        limit=args.limit
    )
    
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.load_continuous_trigger_dataset(
        args.dev,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field
    )
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    test_dataset = utils.load_continuous_trigger_dataset(
        args.test,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field
    )
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
            mask_token_idxs = (model_inputs["input_ids"] == eos_idx).nonzero()[:,1] + args.trigger_length
            model_inputs = generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx)
            labels = labels.to(device)[:, 1]
            optimizer.zero_grad()
            logits, *_ = model(**model_inputs)    
            mask_logits = logits[torch.arange(0, logits.shape[0], dtype=torch.long), mask_token_idxs]
            loss = F.cross_entropy(mask_logits, labels)
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
    parser.add_argument('--field-a', type=str)
    parser.add_argument('--field-b', type=str, default=None)
    parser.add_argument('--label-field', type=str, default='label')
    parser.add_argument('--trigger-length', type=int, default=5)
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
