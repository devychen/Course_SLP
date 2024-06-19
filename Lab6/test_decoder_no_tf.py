import random
import unittest
import torch
from Decoder import Decoder


class DecoderTest(unittest.TestCase):

    def test_forward_step1(self):
        # outputs
        torch.manual_seed(42)
        output_vocab_size = 4
        hidden_size = 6
        decoder = Decoder(hidden_size, output_vocab_size)
        x = torch.tensor([0])
        hidden = torch.rand((1, hidden_size))
        expected = torch.tensor([[-1.2475, -1.1073, -1.6232, -1.6870]])
        actual,_ = decoder.forward_step(x, hidden)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward_step2(self):
        # hidden
        torch.manual_seed(42)
        output_vocab_size = 4
        hidden_size = 6
        decoder = Decoder(hidden_size, output_vocab_size)
        x = torch.tensor([0])
        hidden = torch.rand((1, hidden_size))
        expected = torch.tensor([[-0.0562,  0.1859,  0.6642,  0.1611, -0.0069, -0.2811]])
        _, actual = decoder.forward_step(x, hidden)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward1(self):
        # outputs, no teacher forcing, no target tensor
        torch.manual_seed(42)
        random.seed(42)
        hidden_size = 6
        output_vocab_size = 4
        decoder = Decoder(hidden_size, output_vocab_size)
        encoder_hidden = torch.rand((1, hidden_size))
        expected = torch.tensor([[-1.2475, -1.1073, -1.6232, -1.6870]])
        actual,_ = decoder.forward(encoder_hidden)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward2(self):
        # hidden, no teacher forcing, no target tensor
        torch.manual_seed(42)
        random.seed(42)
        hidden_size = 6
        output_vocab_size = 4
        decoder = Decoder(hidden_size, output_vocab_size)
        encoder_hidden = torch.rand((1, hidden_size))
        expected = torch.tensor([[-0.0562,  0.1859,  0.6642,  0.1611, -0.0069, -0.2811]])
        _, actual = decoder.forward(encoder_hidden)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")


if __name__ == '__main__':
    unittest.main()
