from unittest import TestCase

import transformers

import autoprompt.continuous_trigger as ct


class TestContinuousTriggerModel(TestCase):
    def setUp(self):
        pass

    def test_initialize_from_other_model(self):
        config = transformers.AutoConfig.from_pretrained('bert-base-cased')
        model = ct.ContinuousTriggerModel(config)

    def test_load_pretrained_model(self):
        pass

