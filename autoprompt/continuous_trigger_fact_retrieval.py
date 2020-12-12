import argparse
import logging
from pathlib import Path
import random
import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer, AutoModelForMaskedLM
from tqdm import tqdm

import autoprompt.utils as utils
from autoprompt.preprocessors import PREPROCESSORS


logger = logging.getLogger(__name__)


def set_seed(seed: int):
    """Sets the relevant random seeds."""
    random.seed(seed)
    np.random.seed(seed)
    torch.random.manual_seed(seed)
    torch.cuda.manual_seed(seed)


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


def decode(model, model_inputs, strategy="iterative"):
    """
    Decode from model.

    WARNING: This modifies the model_inputs tensors.

    Parameters
    ==========
    model : transformers.PretrainedModel
        The model to use for predictions.
    model_inputs : Dict[str, torch.LongTensor]
        The model inputs.
    strategy : str
        The decoding strategy. One of: parallel, monotonic, iterative.
        * parallel: all predictions made at the same time.
        * monotonic: predictions decoded from left to right.
        * iterative: predictions decoded in order of highest probability.
    """
    assert strategy in ['parallel', 'monotonic', 'iterative']

    # Initialize output to ignore label.
    output = torch.zeros_like(model_inputs['input_ids'])
    output.fill_(-100)

    if strategy == 'parallel':
        # Simple argmax over arguments.
        predict_mask = model_inputs['predict_mask']
        logits = forward_w_triggers(model, model_inputs).logits
        preds = logits.argmax(dim=-1)
        output[predict_mask] = preds[predict_mask]

    elif strategy == 'monotonic':
        predict_mask = model_inputs['predict_mask']
        input_ids = model_inputs['input_ids']
        iterations = predict_mask.sum().item()
        for i in range(iterations):
            # NOTE: The janky double indexing below should be accessing the
            # first token in the remaining predict mask.
            logits = forward_w_triggers(model, model_inputs).logits
            idx = predict_mask.nonzero()[0,1]
            pred = logits[:, idx].argmax(dim=-1)
            input_ids[:, idx] = pred
            output[:, idx] = pred
            predict_mask[:, idx] = False

    elif strategy == 'iterative':
        predict_mask = model_inputs['predict_mask']
        input_ids = model_inputs['input_ids']
        iterations = predict_mask.sum().item()
        for i in range(iterations):
            # NOTE: We're going to be lazy and make the search for the most
            # likely prediction easier by setting the logits for any tokens
            # other than the candidates to a huge negative number.
            logits = forward_w_triggers(model, model_inputs).logits
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
    logger.info("Dataset: %s" % str(args.train).split("/")[3])
    set_seed(args.seed)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    config = AutoConfig.from_pretrained(args.model_name)
    tokenizer = AutoTokenizer.from_pretrained(args.model_name, add_prefix_space=True)
    utils.add_task_specific_tokens(tokenizer)
    model = AutoModelForMaskedLM.from_pretrained(args.model_name, config=config)
    # TODO(rloganiv): See if transformers has a general API call for getting
    # the word embeddings. If so simplify the below code.
    if args.model_name == "bert-base-cased":
        model.embeds = model.bert.embeddings.word_embeddings
    elif args.model_name == "roberta-base":
        model.embeds = model.roberta.embeddings.word_embeddings
    if not args.finetune:
        for param in model.parameters():
            param.requires_grad = False

    templatizer = utils.MultiTokenTemplatizer(
        template=args.template,
        tokenizer=tokenizer,
        label_field=args.label_field,
    )
    collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)
    train_dataset = utils.load_trigger_dataset(
        args.train,
        templatizer=templatizer,
        train=True,
        preprocessor_key=args.preprocessor,
    )
    train_loader = DataLoader(train_dataset, batch_size=args.bsz, shuffle=True, collate_fn=collator)
    dev_dataset = utils.load_trigger_dataset(
        args.dev,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
    )
    dev_loader = DataLoader(dev_dataset, batch_size=1, collate_fn=collator)
    test_dataset = utils.load_trigger_dataset(
        args.test,
        templatizer=templatizer,
        preprocessor_key=args.preprocessor,
    )
    test_loader = DataLoader(test_dataset, batch_size=1, collate_fn=collator)

    # TODO: Double check parameters get registered. Maybe it's better to make a
    # new Module so the tensor isn't unexpected upon reloading?
    model.relation_embeds = torch.nn.Parameter(
        torch.rand(
            templatizer.num_trigger_tokens,
            model.embeds.weight.shape[1],
            requires_grad=True,
        ), 
    )
    model.to(device)

    optimizer = torch.optim.Adam(model.parameters(), lr=args.lr, weight_decay=1e-6)

    best_accuracy = 0
    for epoch in range(args.epochs):
        logger.info('Training...')
        model.train()
        avg_loss = utils.ExponentialMovingAverage()
        pbar = tqdm(train_loader)
        for model_inputs, labels in pbar:
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            outputs = forward_w_triggers(model, model_inputs, labels)
            outputs.loss.backward()
            optimizer.step()
            avg_loss.update(outputs.loss.item())  #  TODO: This is slow
            pbar.set_description(f'loss: {avg_loss.get_metric(): 0.4f}')

        logger.info('Evaluating...')
        model.eval()
        correct = 0
        total = 0
        for model_inputs, labels in tqdm(dev_loader):
            model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
            labels = labels.to(device)
            pmask = model_inputs['predict_mask'].clone()
            preds = decode(model, model_inputs, strategy=args.strategy)
            print('Label: ' + tokenizer.decode(labels[pmask].tolist()))
            print('Predicted: ' + tokenizer.decode(preds[pmask].tolist()))
            correct += (preds == labels).all().item()
            total += 1

        accuracy = correct / (total + 1e-13)
        logger.info(f'Accuracy: {accuracy : 0.4f}')

        if accuracy > best_accuracy:
            logger.info('Best performance so far.')
            # torch.save(model.state_dict(), args.ckpt_dir / WEIGHTS_NAME)
            # model.config.to_json_file(args.ckpt_dir / CONFIG_NAME)
            # tokenizer.save_pretrained(args.ckpt_dir)
            best_accuracy = accuracy

    # logger.info('Testing...')
    # model.eval()
    # correct = 0
    # total = 0
    # # TO DO: currently testing on last model, not best validation model
    # for model_inputs, labels in test_loader:
        # model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
        # mask_token_idxs = (model_inputs["input_ids"] == eos_idx).nonzero()[:,1] + args.trigger_length
        # model_inputs = generate_inputs_embeds(model_inputs, model, tokenizer, eos_idx)
        # labels = labels.to(device)[:, 1]
        # logits, *_ = model(**model_inputs)
        # mask_logits = logits[torch.arange(0, logits.shape[0], dtype=torch.long), mask_token_idxs]
        # preds = torch.topk(mask_logits, 1, dim=1).indices[:, 0]
        # correct += (preds == labels).sum().item()
        # total += labels.size(0)
    # accuracy = correct / (total + 1e-13)
    # logger.info(f'Accuracy: {accuracy : 0.4f}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--model-name', type=str, required=True)
    parser.add_argument('--train', type=Path, required=True)
    parser.add_argument('--dev', type=Path, required=True)
    parser.add_argument('--test', type=Path, required=True)
    parser.add_argument('--template', type=str, required=True)
    parser.add_argument('--label-field', type=str, default='label')
    parser.add_argument('--preprocessor', type=str, default=None,
                        choices=PREPROCESSORS.keys())
    parser.add_argument('--strategy', type=str, default='iterative',
                        choices=['parallel', 'monotonic', 'iterative'])
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

