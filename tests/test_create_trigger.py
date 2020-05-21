from unittest import TestCase

import torch
from transformers import AutoConfig, AutoModelWithLMHead, AutoTokenizer

import create_trigger as ct


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
