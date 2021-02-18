from unittest import TestCase

import torch
from transformers import AutoTokenizer

import autoprompt.templatizers as templatizers


class TestEncodeLabel(TestCase):
    def setUp(self):
        self._tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')

    def test_single_token(self):
        output = templatizers._encode_label(self._tokenizer, 'the')
        expected_output = torch.tensor([self._tokenizer.convert_tokens_to_ids(['the'])])
        assert torch.equal(output, expected_output)

    def test_multiple_tokens(self):
        output = templatizers._encode_label(self._tokenizer, ['a', 'the'])
        expected_output = torch.tensor([
            self._tokenizer.convert_tokens_to_ids(['a', 'the'])
        ])
        assert torch.equal(output, expected_output)

    # TODO(rloganiv): Test no longer fails as Error was downgraded to a Warning
    # message in the log. With introduction of separate templatizer for
    # multi-token labels perhaps we should go back to raising an error?

    # def test_fails_on_multi_word_piece_labels(self):
    #     with self.assertRaises(ValueError):
    #         utils.encode_label(
    #             self._tokenizer,
    #             'Supercalifragilisticexpialidocious',
    #             tokenize=True,
    #         )
    #     with self.assertRaises(ValueError):
    #         utils.encode_label(
    #             self._tokenizer,
    #             ['Supercalifragilisticexpialidocious', 'chimneysweep'],
    #             tokenize=True,
    #         )


class TestTriggerTemplatizer(TestCase):
    def setUp(self):
        self.default_template = '[T] [T] {arbitrary} [T] {fields} [P]'
        self.default_tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-cased',
            add_prefix_space=True,
            additional_special_tokens=('[T]', '[P]'),
        )
        self.default_instance = {
            'arbitrary': 'does this',
            'fields': 'work',
            'label': 'and'
        }

    def test_bert(self):
        templatizer = templatizers.TriggerTemplatizer(
            self.default_template,
            self.default_tokenizer,
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
            [[False, True, True, False, False, True, False, False, False]]
        )
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor(
            [[False, False, False, False, False, False, False, True, False]]
        )
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])

        # Lastly, ensure [P] is replaced by a [MASK] token
        input_ids = model_inputs['input_ids']
        predict_mask = model_inputs['predict_mask']
        predict_token_id = input_ids[predict_mask].squeeze().item()
        assert predict_token_id == self.default_tokenizer.mask_token_id

    def test_roberta(self):
        tokenizer = AutoTokenizer.from_pretrained(
            'roberta-base',
            add_prefix_space=True,
            additional_special_tokens=('[T]', '[P]'),
        )
        templatizer = templatizers.TriggerTemplatizer(
            self.default_template,
            tokenizer,
        )

        model_inputs, label = templatizer(self.default_instance)

        # Label should be mapped to its token id
        expected_label = torch.tensor([tokenizer.convert_tokens_to_ids([self.default_instance['label']])])
        assert torch.equal(expected_label, label)

        # For BERT ouput is expected to have the following keys
        assert 'input_ids' in model_inputs
        assert 'attention_mask' in model_inputs

        # Test that the custom masks match our expectations
        expected_trigger_mask = torch.tensor(
            [[False, True, True, False, False, True, False, False, False]]
        )
        assert torch.equal(expected_trigger_mask, model_inputs['trigger_mask'])

        expected_predict_mask = torch.tensor(
            [[False, False, False, False, False, False, False, True, False]]
        )
        assert torch.equal(expected_predict_mask, model_inputs['predict_mask'])

        # Lastly, ensure [P] is replaced by a [MASK] token
        input_ids = model_inputs['input_ids']
        predict_mask = model_inputs['predict_mask']
        predict_token_id = input_ids[predict_mask].squeeze().item()
        assert predict_token_id == tokenizer.mask_token_id


class TestMultiTokenTemplatizer(TestCase):
    def setUp(self):
        self.default_template = '[T] [T] {arbitrary} [T] {fields} [P]'
        self.default_tokenizer = AutoTokenizer.from_pretrained(
            'bert-base-cased',
            add_prefix_space=True,
            additional_special_tokens=('[T]', '[P]'),
        )

    def test_label(self):
        templatizer = templatizers.MultiTokenTemplatizer(
            self.default_template,
            self.default_tokenizer,
            add_padding=True,
        )
        format_kwargs = {
            'arbitrary': 'ehh',
            'fields': 'whats up doc',
            'label': 'bugs bunny'
        }
        model_inputs, labels = templatizer(format_kwargs)
        input_ids = model_inputs.pop('input_ids')

        # Check that all shapes are the same
        for tensor in model_inputs.values():
            self.assertEqual(input_ids.shape, tensor.shape)
        self.assertEqual(input_ids.shape, labels.shape)

        # Check that detokenized inputs replaced [T] and [P] with the correct
        # number of masks. The expected number of predict masks is 5,
        # corresponding to:
        #   ['bugs', 'b', '##un', '##ny', '<pad>']
        # and the expected number of trigger masks is 3.
        self.assertEqual(model_inputs['trigger_mask'].sum().item(), 3)
        self.assertEqual(model_inputs['predict_mask'].sum().item(), 5)
        mask_token_id = self.default_tokenizer.mask_token_id
        num_masks = input_ids.eq(mask_token_id).sum().item()
        self.assertEqual(num_masks, 8)
