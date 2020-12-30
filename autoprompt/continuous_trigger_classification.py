import argparse
import logging
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForSequenceClassification, PreTrainedModel
from transformers import get_linear_schedule_with_warmup, AdamW
from tqdm import tqdm

import autoprompt.utils as utils


logger = logging.getLogger(__name__)


def generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx):
    source_ids = model_inputs["input_ids"]
    source_mask = model_inputs["attention_mask"]
    eos_token_idxs = (source_ids == eos_idx).nonzero()[:,1]
    if eos_token_idxs.shape[0] == 2*source_ids.shape[0]:
        eos_token_idxs = eos_token_idxs[::2]

    subject_embeds = model.embeds(source_ids)
    relation_embeds = model.relation_embeds.to(source_ids.device)

    inputs_embeds = torch.cat([torch.cat([sembedding[:eos_token_idxs[idx], :], 
                                          relation_embeds,
                                          sembedding[eos_token_idxs[idx]:, :]], dim=0).unsqueeze(0) 
                                          for idx, sembedding in enumerate(subject_embeds)], dim=0)
    input_attention_mask = torch.cat([torch.ones((len(source_ids), relation_embeds.shape[0]), 
                                      dtype=torch.long).to(source_ids.device), source_mask], dim=1)
    return {"inputs_embeds": inputs_embeds, "attention_mask": input_attention_mask}


class ContTriggerTransformer(PreTrainedModel):

    def __init__(self, config, model_name, trigger_length, finetune=False):
        super(ContTriggerTransformer, self).__init__(config)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_name, config=self.config)
        if model_name == "bert-base-cased":
            self.embeds = self.model.bert.embeddings.word_embeddings
            if not finetune:
                for param in self.model.bert.parameters():
                    param.requires_grad = False
        elif model_name == "roberta-base":
            self.embeds = self.model.roberta.embeddings.word_embeddings
            if not finetune:
                for param in self.model.roberta.parameters():
                    param.requires_grad = False
        indices = np.random.randint(0, self.embeds.weight.shape[0], size=trigger_length)
        self.relation_embeds = torch.nn.Parameter(self.embeds.weight.detach()[indices], 
                                                                            requires_grad=True)

    
    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        **kwargs
    ):
        output = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
        )

        return output



def main(args):
    utils.set_seed(args.seed)

    # Handle multi-GPU setup
    world_size = os.getenv('WORLD_SIZE', -1)
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
        if args.world_size != -1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=args.local_rank,
                world_size=world_size,
            )
    is_main_process = args.local_rank in [-1, 0] or world_size == -1

    # Only have main process log (unless debug is enabled)
    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO if is_main_process else logging.WARN)
    logger.warning('Rank: %s - World Size: %s', args.local_rank, world_size)

    config = AutoConfig.from_pretrained(args.model_name, num_labels=args.num_labels)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name)
    model = ContTriggerTransformer(config, args.model_name, args.trigger_length, finetune=args.finetune)
    
    if args.model_name == "bert-base-cased":
        eos_idx = 102
    elif args.model_name == "roberta-base":
        eos_idx = tokenizer.eos_token_id
    
    model.to(device)
    if world_size != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
        )

    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

    train_dataset, label_map = utils.load_classification_dataset(
        args.train,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        limit=args.limit,
    )
    if world_size == -1:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=collator, sampler=train_sampler)

    dev_dataset, _ = utils.load_classification_dataset(
        args.dev,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map,
        limit=args.limit,
    )
    if world_size == -1:
        dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)
    else:
        dev_sampler = torch.utils.data.DistributedSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, collate_fn=collator, sampler=dev_sampler)

    test_dataset, _ = utils.load_classification_dataset(
        args.test,
        tokenizer,
        args.field_a,
        args.field_b,
        args.label_field,
        label_map,
    )
    if world_size == -1:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    else:
        test_sampler = torch.utils.data.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    optimizer = torch.optim.Adam(list(model.model.classifier.parameters()) + [model.relation_embeds],
                                                        lr=args.lr, weight_decay=args.weight_decay)

    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        avg_loss = utils.ExponentialMovingAverage()
        if is_main_process:
            iter_ = tqdm(train_loader)
        else:
            iter_ = train_loader
        for model_inputs, labels in iter_:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            model_inputs = generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx)
            labels = labels.to(device)
            optimizer.zero_grad()
            logits, *_ = model(**model_inputs)    
            loss = F.cross_entropy(logits, labels.squeeze(-1))
            loss.backward()
            optimizer.step()
            # NOTE: This loss will only be on the subset of training data in
            # the main process.
            avg_loss.update(loss.item())
            if is_main_process:
                iter_.set_description(f'loss: {avg_loss.get_metric(): 0.4f}')

        logger.info('Evaluating...')
        model.eval()
        correct = torch.tensor(0.0, device=device)
        total = torch.tensor(0.0, device=device)
        with torch.no_grad():
            for model_inputs, labels in dev_loader:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                model_inputs = generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx)
                labels = labels.to(device)
                logits, *_ = model(**model_inputs)
                _, preds = logits.max(dim=-1)
                correct += (preds == labels.squeeze(-1)).sum().item()
                total += labels.size(0)

        # Gather accuracy across processes
        if world_size != -1:
            torch.distributed.reduce(correct, 0)
            torch.distributed.reduce(total, 0)
        accuracy = correct / (total + 1e-13)
        logger.info(f'Accuracy: {accuracy : 0.4f}')

        if is_main_process:
            if accuracy > best_accuracy:
                logger.info('Best performance so far.')
                best_accuracy = accuracy
                if not args.ckpt_dir.exists():
                    args.ckpt_dir.mkdir(parents=True)
                model.save_pretrained(args.ckpt_dir)
                tokenizer.save_pretrained(args.ckpt_dir)

    logger.info('Testing...')
    checkpoint = torch.load(args.ckpt_dir / "pytorch_model.bin")
    model.load_state_dict(checkpoint)
    if args.tmp:
        logger.info('Temporary mode enabled, deleting checkpoint')
        shutil.rmtree(args.ckpt_dir)
    model.eval()
    correct = torch.tensor(0.0, device=device)
    total = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            model_inputs = generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx)
            labels = labels.to(device)
            logits, *_ = model(**model_inputs)
            _, preds = logits.max(dim=-1)
            correct += (preds == labels.squeeze(-1)).sum().item()
            total += labels.size(0)

    if world_size != -1:
        torch.distributed.reduce(correct, 0)
        torch.distributed.reduce(total, 0)
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
    parser.add_argument('--ckpt-dir', type=Path, default='ckpt')
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--weight-decay', type=float, default=1e-6)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-f', '--force-overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    main(args)

