import unittest
import torch
from lang import Lang


class LangTest(unittest.TestCase):

    @classmethod
    def setUp(self) -> None:
        self.lang = Lang()
        self.word2idx = {
            "<s>": 0,
            "</s>": 1,
            ".": 2,
            "?": 3,
            "you": 4,
            "the": 5,
            "a": 6,
            "it": 7,
            "i": 8,
            "and": 9,
            "to": 10,
            "that": 11,
            "what": 12,
            "do": 13,
            "is": 14,
            "your": 15,
            "this": 16,
            "like": 17,
            "we": 18,
            "in": 19,
            "oh": 20,
            "can": 21,
            "he": 22,
            "don't": 23,
            "on": 24,
            "that's": 25,
            "it's": 26,
            "have": 27,
            "!": 28,
            "of": 29,
            "are": 30,
            "go": 31
        }
        self.index2word = {
            0: '<s>',
            1: '</s>',
            2: '.',
            3: '?',
            4: 'you',
            5: 'the',
            6: 'a',
            7: 'it',
            8: 'i',
            9: 'and',
            10: 'to',
            11: 'that',
            12: 'what',
            13: 'do',
            14: 'is',
            15: 'your',
            16: 'this',
            17: 'like',
            18: 'we',
            19: 'in',
            20: 'oh',
            21: 'can',
            22: 'he',
            23: "don't",
            24: 'on',
            25: "that's",
            26: "it's",
            27: 'have',
            28: '!',
            29: 'of',
            30: 'are',
            31: 'go'
        }

    @classmethod
    def tearDown(self) -> None:
       self.lang = None

    def test_load_vocab1(self):
        self.lang.load_vocab("HW3/UnitTestData/unittest-eng_idx_map.json")
        expected = 32
        self.assertEqual(expected, self.lang.vocab_size)

    def test_load_vocab2(self):
        self.lang.load_vocab("HW3/UnitTestData/unittest-eng_idx_map.json")
        expected = self.word2idx
        self.assertDictEqual(expected, self.lang.word2index)

    def test_load_vocab3(self):
        self.lang.load_vocab("HW3/UnitTestData/unittest-eng_idx_map.json")
        expected = self.index2word
        self.assertDictEqual(expected, self.lang.index2word)

    def test_load_vocab4(self):
        self.lang.load_vocab("HW3/UnitTestData/unittest-phone_idx_map.json")
        expected = 56
        self.assertEqual(expected, self.lang.vocab_size)

    def test_words_to_tensor1(self):
        self.lang.word2index = self.word2idx
        word_list = ['are', 'we', 'in', '?', '</s>']
        expected = torch.tensor([[30], [18], [19], [3], [1]], dtype=torch.long)
        actual = self.lang.words_to_tensor(word_list)
        self.assertTrue(torch.all(expected.eq(actual)))

    def test_words_to_tensor2(self):
        self.lang.word2index = self.word2idx
        word_list = ['are', 'we', 'in', '?', '</s>']
        expected = torch.long
        actual = self.lang.words_to_tensor(word_list).dtype
        self.assertEqual(expected, actual)

    def test_tensor_to_words1(self):
        self.lang.index2word = self.index2word
        word_tensor = torch.tensor([[30], [18], [19], [3], [1]], dtype=torch.long)
        expected = ['are', 'we', 'in', '?', '</s>']
        actual = self.lang.tensor_to_words(word_tensor)
        self.assertListEqual(expected, actual)

if __name__ == '__main__':
    unittest.main()
