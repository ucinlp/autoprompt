import argparse
import json
import logging
import os
from pathlib import Path
import random
import shutil

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AdamW, AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from transformers.modeling_bert import BertPreTrainedModel
from transformers.modeling_roberta import RobertaPreTrainedModel
from tqdm import tqdm

import autoprompt.utils as utils
from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


class ExactMatchEvaluator:
    def __init__(
            self,
            model,
            tokenizer,
            decoding_strategy,
            **kwargs
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._decoding_strategy = decoding_strategy
        
    def __call__(self, model_inputs, labels):
        predict_mask = model_inputs['predict_mask'].clone()
        preds = decode(self._model, model_inputs, decoding_strategy=self._decoding_strategy)
        print('Label: ' + self._tokenizer.decode(labels[predict_mask].tolist()))
        print('Predicted: ' + self._tokenizer.decode(preds[predict_mask].tolist()))
        correct = (preds == labels).all().item()
        return correct


class MultipleChoiceEvaluator:
    def __init__(
        self,
        model,
        tokenizer,
        label_map,
        **kwargs
    ):
        self._model = model
        self._tokenizer = tokenizer
        self._label_map = label_map
        label_tokens = self._tokenizer(
            list(label_map.values()),
            add_special_tokens=False,
            return_tensors='pt',
        )['input_ids']
        if label_tokens.size(1) != 1:
            raise ValueError(
                'Multi-token labels not supported for multiple choice evaluation'
            )
        self._label_tokens = label_tokens.view(1, -1)

    def __call__(self, model_inputs, labels):

        # Ensure everything is on the same device
        label_tokens = self._label_tokens.to(labels.device)

        # Get predictions
        predict_mask = model_inputs['predict_mask']
        labels = labels[predict_mask].unsqueeze(-1)
        logits, *_ = forward_w_triggers(self._model, model_inputs)
        predict_logits = torch.gather(
            input=logits[predict_mask],
            dim=-1,
            index=label_tokens.repeat(labels.size(0), 1)
        )
        predictions = predict_logits.argmax(dim=-1, keepdims=True)

        # Convert label tokens to their indices in the label map.
        _, label_inds = torch.where(labels.eq(label_tokens))
        label_inds = label_inds.unsqueeze(-1)

        # Get loss
        predict_logp = F.log_softmax(predict_logits, dim=-1)
        loss = -predict_logp.gather(-1, label_inds).mean()

        # Get evaluation score
        correct = predictions.eq(label_inds).sum()

        return loss, correct


EVALUATORS = {
    'exact-match': ExactMatchEvaluator,
    'multiple-choice': MultipleChoiceEvaluator,
}


def forward_w_triggers(model, model_inputs, labels=None):
    """
    Run model forward w/ preprocessing for continuous triggers.

    Parameters
    ==========
    model : transformers.PretrainedModel
        The model to use for predictions.
    model_inputs : Dict[str, torch.LongTensor]
        The model inputs.
    labels : torch.LongTensor
        (optional) Tensor of labels. Loss will be returned if provided.
    """
    # Ensure destructive pop operations are only limited to this function.
    model_inputs = model_inputs.copy()
    trigger_mask = model_inputs.pop('trigger_mask')
    predict_mask = model_inputs.pop('predict_mask')
    input_ids = model_inputs.pop('input_ids')

    # Get embeddings of input sequence
    batch_size = input_ids.size(0)
    inputs_embeds = model.embeds(input_ids)
    inputs_embeds[trigger_mask] = model.relation_embeds.repeat((batch_size, 1))
    model_inputs['inputs_embeds'] = inputs_embeds
    
    return model(**model_inputs, labels=labels)


def decode(model, model_inputs, decoding_strategy="iterative"):
    """
    Decode from model.

    WARNING: This modifies the model_inputs tensors.

    Parameters
    ==========
    model : transformers.PretrainedModel
        The model to use for predictions.
    model_inputs : Dict[str, torch.LongTensor]
        The model inputs.
    decoding_strategy : str
        The decoding strategy. One of: parallel, monotonic, iterative.
        * parallel: all predictions made at the same time.
        * monotonic: predictions decoded from left to right.
        * iterative: predictions decoded in order of highest probability.
    """
    assert decoding_strategy in ['parallel', 'monotonic', 'iterative']

    # Initialize output to ignore label.
    output = torch.zeros_like(model_inputs['input_ids'])
    output.fill_(-100)

    if decoding_strategy == 'parallel':
        # Simple argmax over arguments.
        predict_mask = model_inputs['predict_mask']
        logits, *_ = forward_w_triggers(model, model_inputs)
        preds = logits.argmax(dim=-1)
        output[predict_mask] = preds[predict_mask]

    elif decoding_strategy == 'monotonic':
        predict_mask = model_inputs['predict_mask']
        input_ids = model_inputs['input_ids']
        iterations = predict_mask.sum().item()
        for i in range(iterations):
            # NOTE: The janky double indexing below should be accessing the
            # first token in the remaining predict mask.
            logits, *_ = forward_w_triggers(model, model_inputs)
            idx = predict_mask.nonzero()[0,1]
            pred = logits[:, idx].argmax(dim=-1)
            input_ids[:, idx] = pred
            output[:, idx] = pred
            predict_mask[:, idx] = False

    elif decoding_strategy == 'iterative':
        predict_mask = model_inputs['predict_mask']
        input_ids = model_inputs['input_ids']
        iterations = predict_mask.sum().item()
        for i in range(iterations):
            # NOTE: We're going to be lazy and make the search for the most
            # likely prediction easier by setting the logits for any tokens
            # other than the candidates to a huge negative number.
            logits, *_ = forward_w_triggers(model, model_inputs)
            logits[~predict_mask] = -1e32
            top_scores, preds = torch.max(logits, dim=-1)
            idx = torch.argmax(top_scores, dim=-1)
            pred = preds[:, idx]
            input_ids[:, idx] = pred
            output[:, idx] = pred
            predict_mask[:, idx] = False

    else:
        raise ValueError(
            'Something is really wrong with the control flow in this function'
        )

    return output


def main(args):
    if args.evaluation_strategy == 'exact-match':
        assert args.decoding_strategy is not None
    elif args.evaluation_strategy == 'multiple-choice':
        assert args.label_map is not None
    logger.info(args)

    utils.set_seed(args.seed)

    # Handle multi-GPU setup
    world_size = os.getenv('WORLD_SIZE', -1)
    if args.local_rank == -1:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    else:
        device = torch.device('cuda', args.local_rank)
        if world_size != -1:
            torch.distributed.init_process_group(
                backend='nccl',
                init_method='env://',
                rank=args.local_rank,
                world_size=world_size,
            )
    is_main_process = args.local_rank in [-1, 0] or world_size == -1

    if args.debug:
        logging.basicConfig(level=logging.DEBUG)
    else:
        logging.basicConfig(level=logging.INFO if is_main_process else logging.WARN)

    logger.warning('Rank: %s - World Size: %s', args.local_rank, world_size)

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(
        args.model_name,
        add_prefix_space=True,
        additional_special_tokens=('[T]', '[P]'),
    )
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config)

    # TODO(rloganiv): See if transformers has a general API call for getting
    # the word embeddings. If so simplify the below code.
    if not args.finetune:
        for param in model.parameters():
            param.requires_grad = False
    if isinstance(model, BertPreTrainedModel):
        model.embeds = model.bert.embeddings.word_embeddings
        # model.bert.cls.predictions.decoder.bias.zero_()
    elif isinstance(model, RobertaPreTrainedModel):
        model.embeds = model.roberta.embeddings.word_embeddings
        # model.lm_head.decoder.bias.zero_()
    else:
        raise ValueError(f'{args.model_name} not currently supported.')

    if args.label_map is not None:
        label_map = json.loads(args.label_map)
    else:
        label_map = None
    logger.info(f'Label map: {label_map}')
    templatizer = utils.MultiTokenTemplatizer(
        template=args.template,
        tokenizer=tokenizer,
        label_field=args.label_field,
        label_map=label_map,
        add_padding=args.add_padding,
    )
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(
        args.train,
        templatizer=templatizer,
        train=True,
        preprocessor_key=args.preprocessor,
        limit=args.limit,
    )
    if world_size == -1:
        train_sampler = torch.utils.data.RandomSampler(train_dataset)
    else:
        train_sampler = torch.utils.data.DistributedSampler(train_dataset, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, collate_fn=collator, sampler=train_sampler)
    dev_dataset = utils.load_trigger_dataset(
        args.dev,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
        limit=args.limit,
    )
    if world_size == -1:
        dev_sampler = torch.utils.data.SequentialSampler(dev_dataset)
    else:
        dev_sampler = torch.utils.data.DistributedSampler(dev_dataset)
    dev_loader = DataLoader(dev_dataset, batch_size=args.bsz, collate_fn=collator, sampler=dev_sampler)
    test_dataset = utils.load_trigger_dataset(
        args.test,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
    )
    if world_size == -1:
        test_sampler = torch.utils.data.SequentialSampler(test_dataset)
    else:
        test_sampler = torch.utils.data.DistributedSampler(test_dataset)
    test_loader = DataLoader(test_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)

    evaluator = EVALUATORS[args.evaluation_strategy](
        model=model,
        tokenizer=tokenizer,
        label_map=label_map,
        decoding_strategy=args.decoding_strategy,
    )

    # TODO: Consider refactoring to make model more portable.
    if args.initial_trigger is not None:
        initial_trigger_ids = torch.tensor(
            tokenizer.convert_tokens_to_ids(args.initial_trigger),
        )
        model.relation_embeds = torch.nn.Parameter(
            model.embeds(initial_trigger_ids).detach().clone()
        )
    else:
        model.relation_embeds = torch.nn.Parameter(
            torch.randn(
                templatizer.num_trigger_tokens,
                model.embeds.weight.size(1),
                requires_grad=True,
            ), 
        )
    model.to(device)
    if world_size != -1:
        model = torch.nn.parallel.DistributedDataParallel(
            model,
            device_ids=[args.local_rank],
        )


    optimizer = AdamW(
        model.parameters(),
        lr=args.lr,
        weight_decay=1e-2,
        betas=(0.9, 0.999),
    )

    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        if is_main_process and not args.quiet:
            iter_ = tqdm(train_loader)
        else:
            iter_ = train_loader
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        for model_inputs, labels in iter_:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            loss, correct = evaluator(model_inputs, labels)
            loss.backward()
            optimizer.step()
            total_loss += loss.detach() * labels.size(0)
            total_correct += correct.detach()
            denom += labels.size(0)
            # NOTE: This loss/accuracy is only on the subset  of training data
            # in the main process.
            if is_main_process and not args.quiet:
                iter_.set_description(
                    f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
                    f'Accuracy: {total_correct / (denom + 1e-13): 0.4f}'
                )

        logger.info('Evaluating...')
        model.eval()
        total_loss = torch.tensor(0.0, device=device)
        total_correct = torch.tensor(0.0, device=device)
        denom = torch.tensor(0.0, device=device)
        if is_main_process and not args.quiet:
            iter_ = tqdm(dev_loader)
        else:
            iter_ = dev_loader
        with torch.no_grad():
            for model_inputs, labels in iter_:
                model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
                labels = labels.to(device)
                loss, correct = evaluator(model_inputs, labels)
                total_loss += loss.detach() * labels.size(0)
                total_correct += correct.detach()
                denom += labels.size(0)
        if world_size != -1:
            torch.distributed.reduce(total_loss, 0)
            torch.distributed.reduce(total_correct, 0)
            torch.distributed.reduce(denom, 0)

        logger.info(
            f'Loss: {total_loss / (denom + 1e-13): 0.4f}, '
            f'Accuracy: {total_correct / (denom + 1e-13): 0.4f}'
        )
        accuracy = total_correct / (denom + 1e-13)

        if is_main_process:
            if accuracy > best_accuracy:
                logger.info('Best performance so far.')
                best_accuracy = accuracy
                if not args.ckpt_dir.exists():
                    args.ckpt_dir.mkdir(parents=True)
                model.save_pretrained(args.ckpt_dir)
                tokenizer.save_pretrained(args.ckpt_dir)

    logger.info('Testing...')
    if args.epochs > 0:
        checkpoint = torch.load(args.ckpt_dir / "pytorch_model.bin")
        model.load_state_dict(checkpoint)
        if args.tmp:
            logger.info('Temporary mode enabled, deleting checkpoint')
            shutil.rmtree(args.ckpt_dir)
    model.eval()
    total_correct = torch.tensor(0.0, device=device)
    denom = torch.tensor(0.0, device=device)
    with torch.no_grad():
        for model_inputs, labels in test_loader:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            _, correct = evaluator(model_inputs, labels)
            total_correct += correct.detach()
            denom += labels.size(0)
    if world_size != -1:
        torch.distributed.reduce(correct, 0)
        torch.distributed.reduce(denom, 0)
    accuracy = total_correct / (denom + 1e-13)
    logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--label-map', type=str, default=None)
    parser.add_argument('--initial-trigger', nargs='+', type=str, default=None)
    parser.add_argument('--label-field', type=str, default='label')
    parser.add_argument('--add-padding', action='store_true')
    parser.add_argument('--preprocessor', type=str, default=None,
                        choices=PREPROCESSORS.keys())
    parser.add_argument('--evaluation-strategy', type=str, required=True,
                        choices=EVALUATORS.keys())
    parser.add_argument('--decoding-strategy', type=str, default=None,
                        choices=['parallel', 'monotonic', 'iterative'])
    parser.add_argument('--finetune', action='store_true')
    parser.add_argument('--ckpt-dir', type=Path, default=Path('ckpt/'))
    parser.add_argument('--tmp', action='store_true')
    parser.add_argument('--num-labels', type=int, default=2)
    parser.add_argument('--bsz', type=int, default=32)
    parser.add_argument('--epochs', type=int, default=10)
    parser.add_argument('--lr', type=float, default=3e-2)
    parser.add_argument('--limit', type=int, default=None)
    parser.add_argument('--seed', type=int, default=1234)
    parser.add_argument('-f', '--force-overwrite', action='store_true')
    parser.add_argument('--debug', action='store_true')
    parser.add_argument('--quiet', action='store_true')
    parser.add_argument('--local_rank', type=int, default=-1)
    args = parser.parse_args()

    if args.debug:
        level = logging.DEBUG
    else:
        level = logging.INFO
    logging.basicConfig(level=level)

    main(args)

