import unittest
from constants import BOS_STR, EOS_STR
from utils import load_tsv_data


class UtilsTest(unittest.TestCase):

    def test_load_tsv_data(self):
        expected = [([BOS_STR, "aɪ", "d", "uː", "ɪ", "t", EOS_STR],	["i", "do", "it", ".", EOS_STR]),
                    ([BOS_STR, "h", "æ", "v", "eɪ", EOS_STR],	["have", "a", "?", EOS_STR]),
                    ([BOS_STR, "d", "uː", "w", "iː", "h", "æ", "v", EOS_STR],	["do", "we", "have", ".", EOS_STR]),
                    ([BOS_STR, "w", "ʌ", "t", "j", "uː", "d", "uː", EOS_STR],	["what", "you", "do", "?", EOS_STR])]
        actual = load_tsv_data("HW3/UnitTestData/unittest-dev.tsv")
        self.assertListEqual(expected, actual)


if __name__ == '__main__':
    unittest.main()
