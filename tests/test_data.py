from unittest import TestCase

import torch
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

import autoprompt.data as data
import autoprompt.templatizers as templatizers


class TestCollator(TestCase):

    def test_collator(self):
        template = '[T] [T] {arbitrary} [T] {fields} [P]'
        tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-cased',
            add_prefix_space=True,
            additional_special_tokens=('[T]', '[P]'),
        )
        templatizer = templatizers.TriggerTemplatizer(
            template,
            tokenizer
        )
        collator = data.Collator(pad_token_id=tokenizer.pad_token_id)

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
            tokenizer.encode('hot', add_special_tokens=False),
            tokenizer.encode('cold', add_special_tokens=False),
        ])
        assert torch.equal(expected_labels, labels)

        expected_trigger_mask = torch.tensor([
            [False, True, True, False, True, False, False, False, False, False],
            [False, True, True, False, False, True, False, False, False, False],
        ])
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor([
            [False, False, False, False, False, False, True, False, False, False],
            [False, False, False, False, False, False, False, False, True, False],
        ])
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])

