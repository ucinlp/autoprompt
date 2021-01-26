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


class TestCOPA(TestCase):
    def test_preprocess_copa_train(self):
        fname = 'tests/fixtures/copa_example.jsonl'
        output = list(autoprompt.preprocessors.preprocess_copa(fname, train=True))
        self.assertEqual(len(output), 2)
        expected_0 = {
            'premise': 'the chandelier shattered on the floor',
            'choice1': 'the chandelier\'s lights flickered on and off',
            'choice2': 'the chandelier dropped from the ceiling',
            'conjunction': 'because',
            'labels': [
                ('the chandelier dropped from the ceiling', True),
                ('the chandelier\'s lights flickered on and off', False),
            ]
        }
        self.assertDictEqual(expected_0, output[0])

        expected_1 = {
            'premise': 'the chandelier shattered on the floor',
            'choice1': 'the chandelier dropped from the ceiling',
            'choice2': 'the chandelier\'s lights flickered on and off',
            'conjunction': 'because',
            'labels': [
                ('the chandelier dropped from the ceiling', True),
                ('the chandelier\'s lights flickered on and off', False),
            ]
        }
        self.assertDictEqual(expected_1, output[1])

    def test_preprocess_copa_eval(self):
        fname = 'tests/fixtures/copa_example.jsonl'
        output = list(autoprompt.preprocessors.preprocess_copa(fname, train=False))
        self.assertEqual(len(output), 1)
        expected_0 = {
            'premise': 'the chandelier shattered on the floor',
            'choice1': 'the chandelier\'s lights flickered on and off',
            'choice2': 'the chandelier dropped from the ceiling',
            'conjunction': 'because',
            'labels': [
                ('the chandelier dropped from the ceiling', True),
                ('the chandelier\'s lights flickered on and off', False),
            ]
        }
        self.assertDictEqual(expected_0, output[0])


class TestReCoRD(TestCase):
    def test_preprocess_record(self):
        fname = 'tests/fixtures/record_example.jsonl'
        output = list(autoprompt.preprocessors.preprocess_record(fname))
        self.assertEqual(len(output), 1)

        expected_passage = (
            'By Hamish Mackay Goals from Diego Costa and Kurt Zouma ensured Chelsea came back to '
            'beat Olimpija Ljubljana 2-1 after a first half scare. Jose Mourinho\'s men found '
            'themselves behind going in to the break after Nik Kapun put the Slovenian side 1-0 up. '
            'But second half goals from Costa and Zouma put the Blues back in charge and Chelsea '
            'were comfortable from then on. Branislav Ivanovic thought he had added a third but, '
            'after initially awarding it, the referee appeared to change his mind and the goal was '
            'chalked off. VIDEO Scroll down to watch Diego Costa\'s first goal and a Fernando Torres '
            'horror miss\n- Chelsea come from behind to beat Olimpija Ljubljana 2-1 in Slovenia\n- '
            'Didier Drogba joined the squad and watched from the bench\n- Chelsea named strong '
            'lineup including Diego Costa and Cesc Fabregas\n- Nik Kapun put the hosts ahead in '
            'first half\n- Fabregas set up Costa for the equaliser in the second half\n- Young '
            'defender Kurt Zouma grabbed the winner for Mourinho\'s side\n- Branislav Ivanovic had a '
            'goal controversially disallowed\n- Fernando Torres missed a clear cut chance to make it '
            '3-1'
        )
        self.assertEqual(output[0]['passage'], expected_passage)

        expected_question = (
            'Speaking after the game, Mourinho said: \'The important thing is to give competition to the players, '
            'the best thing was that @placeholder made it difficult.'
        )
        self.assertEqual(output[0]['question'], expected_question)

        # TODO: Might want a more comprehensive test of the labels, but I am too lazy to infer the
        # unique entities. For now we'll just ensure that the correct label is in there.
        for label_text, label in output[0]['labels']:
            if label:
                self.assertEqual(label_text, 'Olimpija Ljubljana')

