import io
from unittest import TestCase
import autoprompt.preprocessors


class TestMultiRC(TestCase):
    def test_preprocess_multirc(self):
        fname = 'tests/fixtures/multirc_example.jsonl'
        output = list(autoprompt.preprocessors.preprocess_multirc(fname))
        self.assertEqual(len(output), 2)
        expected_0 = {
            'passage': 'This is a test',
            'question': 'Is this a test?',
            'answer': 'yes',
            'label': '1',
        }
        self.assertDictEqual(output[0], expected_0)
        expected_1 = {
            'passage': 'This is a test',
            'question': 'Is this a test?',
            'answer': 'no',
            'label': '0',
        }
        self.assertDictEqual(output[1], expected_1)


class TestWSC(TestCase):
    def test_preprocess_wsc(self):
        fname = 'tests/fixtures/wsc_example.jsonl'
        output = list(autoprompt.preprocessors.preprocess_wsc(fname))
        self.assertEqual(len(output), 1)
        expected_0 = {
            'sentence': 'The actress used to be named Terpsichore , but she changed it to Tina a few years ago, because she figured *it* was easier to pronounce.',
            'pronoun': 'it',
            'label': 'Tina',
        }
        self.assertDictEqual(output[0], expected_0)
