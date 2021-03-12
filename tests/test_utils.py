from unittest import TestCase

import torch
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer

import autoprompt.utils as utils


class TestEncodeLabel(TestCase):
    def setUp(self):
        self._tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def test_single_token(self):
        output = utils.encode_label(self._tokenizer, 'the')
        expected_output = torch.tensor([self._tokenizer.convert_tokens_to_ids(['the'])])
        assert torch.equal(output, expected_output)

    def test_multiple_tokens(self):
        output = utils.encode_label(self._tokenizer, ['a', 'the'])
        expected_output = torch.tensor([
            self._tokenizer.convert_tokens_to_ids(['a', 'the'])
        ])
        assert torch.equal(output, expected_output)


class TestTriggerTemplatizer(TestCase):
    def setUp(self):
        self.default_template = '[T] [T] {arbitrary} [T] {fields} [P]'
        self.default_config = AutoConfig.from_pretrained('bert-base-cased')
        self.default_tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        utils.add_task_specific_tokens(self.default_tokenizer)
        self.default_instance = {
            'arbitrary': 'does this',
            'fields': 'work',
            'label': 'and'
        }

    def test_bert(self):
        templatizer = utils.TriggerTemplatizer(
            self.default_template,
            self.default_config,
            self.default_tokenizer,
            add_special_tokens=False
        )
        model_inputs, label = templatizer(self.default_instance)

        # Label should be mapped to its token id
        expected_label = torch.tensor([self.default_tokenizer.convert_tokens_to_ids([self.default_instance['label']])])
        assert torch.equal(expected_label, label)

        # For BERT ouput is expected to have the following keys
        assert 'input_ids' in model_inputs
        assert 'token_type_ids' in model_inputs
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
        assert predict_token_id == self.default_tokenizer.mask_token_id

    def test_roberta(self):
        config = AutoConfig.from_pretrained('roberta-base')
        tokenizer = AutoTokenizer.from_pretrained('roberta-base')
        utils.add_task_specific_tokens(tokenizer)
        templatizer = utils.TriggerTemplatizer(
            self.default_template,
            config,
            tokenizer,
            add_special_tokens=False
        )

        model_inputs, label = templatizer(self.default_instance)

        # Label should be mapped to its token id
        expected_label = torch.tensor([tokenizer.convert_tokens_to_ids([self.default_instance['label']])])
        assert torch.equal(expected_label, label)

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

    def test_collator(self):
        template = '[T] [T] {arbitrary} [T] {fields} [P]'
        tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
        config = AutoConfig.from_pretrained('bert-base-cased')
        utils.add_task_specific_tokens(tokenizer)
        templatizer = utils.TriggerTemplatizer(
            template,
            config,
            tokenizer,
            add_special_tokens=False
        )
        collator = utils.Collator(pad_token_id=tokenizer.pad_token_id)

        instances = [
            {'arbitrary': 'a', 'fields': 'the', 'label': 'hot'},
            {'arbitrary': 'a a', 'fields': 'the the', 'label': 'cold'}
        ]
        templatized_instances = [templatizer(x) for x in instances]
        loader = DataLoader(
            templatized_instances,
            batch_size=2,
            shuffle=False,
            collate_fn=collator
        )
        model_inputs, labels = next(iter(loader))

        # Check results match our expectations
        expected_labels = torch.tensor([
            tokenizer.encode('hot', add_special_tokens=False, add_prefix_space=True),
            tokenizer.encode('cold', add_special_tokens=False, add_prefix_space=True),
        ])
        assert torch.equal(expected_labels, labels)

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
