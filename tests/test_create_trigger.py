from unittest import TestCase

import torch
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer

import autoprompt.create_trigger as ct


def _load(model_name):
    config = AutoConfig.from_pretrained('bert-base-cased')
    model = AutoModelWithLMHead.from_pretrained('bert-base-cased')
    tokenizer = AutoTokenizer.from_pretrained('bert-base-cased')
    return config, model, tokenizer


class TestGetEmbeddings(TestCase):
    def test_bert(self):
        model_name = 'bert-base-cased'
        config, model, tokenizer = _load(model_name)
        embeddings = ct.get_embeddings(model, config)
        self.assertEqual(embeddings.weight.shape[0], config.vocab_size)

    def test_roberta(self):
        model_name = 'roberta-base'
        config, model, tokenizer = _load(model_name)
        embeddings = ct.get_embeddings(model, config)
        self.assertEqual(embeddings.weight.shape[0], config.vocab_size)


class TestGradientStorage(TestCase):
    def test_gradient_storage(self):
        num_embeddings = 3
        embedding_dim = 4
        embeddings = torch.nn.Embedding(num_embeddings, embedding_dim)
        embedding_storage = ct.GradientStorage(embeddings)

        inputs = torch.tensor([0, 1, 2, 1])
        outputs = embeddings(inputs)
        outputs.retain_grad()
        loss = outputs.sum()
        loss.backward()

        assert torch.equal(outputs.grad, embedding_storage.get())


def test_replace_trigger_tokens():
    model_inputs = {
        'input_ids': torch.tensor([
            [1, 2, 3, 4],
            [1, 1, 1, 0]
        ])
    }
    trigger_ids = torch.tensor([[5, 6]])
    trigger_mask = torch.tensor([
            [True, True, False, False],
            [False, True, False, True]
    ])
    replaced = ct.replace_trigger_tokens(model_inputs, trigger_ids, trigger_mask)
    expected = torch.tensor([
        [5, 6, 3, 4],
        [1, 5, 1, 6]
    ])
    assert torch.equal(expected, replaced['input_ids'])
