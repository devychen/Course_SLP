import unittest
import torch
from Encoder import Encoder


class EncoderTest(unittest.TestCase):

    def test_init_hidden_1dir1(self):
        # shape
        torch.manual_seed(42)
        input_vocab_size = 6
        hidden_size = 4
        encoder = Encoder(input_vocab_size, hidden_size)
        expected = torch.zeros(1, 4)
        actual = encoder.initHidden()
        self.assertTupleEqual(expected.shape, actual.shape)

    def test_init_hidden_1dir2(self):
        # contents
        torch.manual_seed(42)
        input_vocab_size = 6
        hidden_size = 4
        encoder = Encoder(input_vocab_size, hidden_size)
        expected = torch.zeros(1, 4)
        actual = encoder.initHidden()
        self.assertTrue(torch.all(expected.isclose(actual)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward_step1(self):
        # output
        torch.manual_seed(42)
        input_vocab_size = 10
        hidden_size = 4
        encoder = Encoder(input_vocab_size, hidden_size)
        x = torch.tensor([0], dtype=torch.long)
        hidden = torch.zeros(1, hidden_size)
        expected = torch.tensor([[ 0.1681,  0.6698,  0.5264, -0.0350]])
        actual,_ = encoder.forward_step(x, hidden)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward_step2(self):
        # hidden
        torch.manual_seed(42)
        input_vocab_size = 10
        hidden_size = 4
        encoder = Encoder(input_vocab_size, hidden_size)
        x = torch.tensor([5], dtype=torch.long)
        hidden = torch.zeros(1, hidden_size)
        expected = torch.tensor([[ 0.5425, -0.1099, -0.5448, -0.5202]])
        _, actual = encoder.forward_step(x, hidden)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward1(self):
        # hidden, single input
        torch.manual_seed(42)
        input_vocab_size = 10
        hidden_size = 4
        encoder = Encoder(input_vocab_size, hidden_size)
        x = torch.tensor([[5]], dtype=torch.long)
        expected = torch.tensor([[ 0.5425, -0.1099, -0.5448, -0.5202]])
        _, actual = encoder.forward(x)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")

    def test_forward2(self):
        # hidden, multiple inputs
        torch.manual_seed(42)
        input_vocab_size = 10
        hidden_size = 4
        encoder = Encoder(input_vocab_size, hidden_size)
        x = torch.tensor([[0], [9], [1]], dtype=torch.long)
        expected = torch.tensor([[ 0.1747,  0.3622,  0.7718, -0.2004]])
        _, actual = encoder.forward(x)
        self.assertTrue(torch.all(expected.isclose(actual, atol=.0001)),
                        msg=f"\nexpected: {expected}\nactual: {actual}")


if __name__ == '__main__':
    unittest.main()
