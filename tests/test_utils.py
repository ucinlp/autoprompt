from unittest import TestCase

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import utils


class TestTriggerTemplatizer(TestCase):
    def test_bert(self):
        template = '[T] [T] {arbitrary} [T] {fields} [P]'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        utils.add_task_specific_tokens(tokenizer)
        templatizer = utils.TriggerTemplatizer(
            template,
            tokenizer,
            add_special_tokens=False
        )

        format_kwargs = {
            'arbitrary': 'does this',
            'fields': 'work',
            'label': 'this is a generic label, and can be really long'
        }
        model_inputs, label = templatizer(format_kwargs)

        # Label is not expected to change
        assert label == format_kwargs['label']

        # For BERT ouput is expected to have the following keys
        assert 'input_ids' in model_inputs
        assert 'token_type_ids' in model_inputs
        assert 'attention_mask' in model_inputs

        print(model_inputs)

        # Test that the custom masks match our expectations
        expected_trigger_mask = torch.tensor(
            [[True, True, False, False, True, False, False]]
        )
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor(
            [[False, False, False, False, False, False, True]]
        )
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])

        # Lastly, ensure [P] is replaced by a [MASK] token
        input_ids = model_inputs['input_ids']
        predict_mask = model_inputs['predict_mask']
        predict_token_id = input_ids[predict_mask].squeeze().item()
        assert predict_token_id == tokenizer.mask_token_id

    def test_roberta(self):
        template = ' [T] [T] {arbitrary} [T] {fields} [P]'
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        utils.add_task_specific_tokens(tokenizer)
        templatizer = utils.TriggerTemplatizer(
            template,
            tokenizer,
            add_special_tokens=False
        )

        format_kwargs = {
            'arbitrary': 'does this',
            'fields': 'work',
            'label': 'this is a generic label, and can be really long'
        }
        model_inputs, label = templatizer(format_kwargs)

        # Label is not expected to change
        assert label == format_kwargs['label']

        # For BERT ouput is expected to have the following keys
        print(model_inputs)
        assert 'input_ids' in model_inputs
        assert 'attention_mask' in model_inputs

        # Test that the custom masks match our expectations
        expected_trigger_mask = torch.tensor(
            [[True, True, False, False, True, False, False]]
        )
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor(
            [[False, False, False, False, False, False, True]]
        )
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])

        # Lastly, ensure [P] is replaced by a [MASK] token
        input_ids = model_inputs['input_ids']
        predict_mask = model_inputs['predict_mask']
        predict_token_id = input_ids[predict_mask].squeeze().item()
        assert predict_token_id == tokenizer.mask_token_id


class TestCollator(TestCase):

    def test_bert(self):
        template = '[T] [T] {arbitrary} [T] {fields} [P]'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        utils.add_task_specific_tokens(tokenizer)
        templatizer = utils.TriggerTemplatizer(
            template,
            tokenizer,
            add_special_tokens=False
        )
        collator = utils.TriggerCollator(pad_token_id=tokenizer.pad_token_id)

        instances = [
            {'arbitrary': 'a', 'fields': 'the', 'label': 'hot'},
            {'arbitrary': 'a a', 'fields': 'the the', 'label': 'cold'}
        ]
        templatized_instances = [templatizer(x) for x in instances]
        print(templatized_instances)
        loader = DataLoader(
            templatized_instances,
            batch_size=2,
            shuffle=False,
            collate_fn=collator
        )
        model_inputs, labels = next(iter(loader))

        # Check results match our expectations
        assert labels == ('hot', 'cold')

        expected_trigger_mask = torch.tensor([
            [True, True, False, True, False, False, False, False],
            [True, True, False, False, True, False, False, False],
        ])
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor([
            [False, False, False, False, False, True, False, False],
            [False, False, False, False, False, False, False, True],
        ])
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])

    def test_roberta(self):
        template = ' [T] [T] {arbitrary} [T] {fields} [P]'
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        utils.add_task_specific_tokens(tokenizer)
        templatizer = utils.TriggerTemplatizer(
            template,
            tokenizer,
            add_special_tokens=False
        )
        collator = utils.TriggerCollator(pad_token_id=tokenizer.pad_token_id)

        instances = [
            {'arbitrary': 'a', 'fields': 'the', 'label': 'hot'},
            {'arbitrary': 'a a', 'fields': 'the the', 'label': 'cold'}
        ]
        templatized_instances = [templatizer(x) for x in instances]
        print(templatized_instances)
        loader = DataLoader(
            templatized_instances,
            batch_size=2,
            shuffle=False,
            collate_fn=collator
        )
        model_inputs, labels = next(iter(loader))

        # Check results match our expectations
        assert labels == ('hot', 'cold')

        expected_trigger_mask = torch.tensor([
            [True, True, False, True, False, False, False, False],
            [True, True, False, False, True, False, False, False],
        ])
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor([
            [False, False, False, False, False, True, False, False],
            [False, False, False, False, False, False, False, True],
        ])
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])
