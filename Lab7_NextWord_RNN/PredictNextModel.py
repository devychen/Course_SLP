import torch
import torch.nn as nn
from constants import *


class PredictNextModel(nn.Module):
    """
    A model to predict the next word in a sequence (sentence).
    """
    def __init__(self, vocab_size, hidden_size, rnn_type=USE_RNN):
        super(PredictNextModel, self).__init__()

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.rnn_type = rnn_type

        self.embedding = nn.Embedding(vocab_size, hidden_size)

        if rnn_type == USE_RNN:
            self.rnn = nn.RNN(hidden_size, hidden_size)
        elif rnn_type == USE_GRU:
            self.rnn = nn.GRU(hidden_size, hidden_size)
        else:
            self.rnn = nn.LSTM(hidden_size, hidden_size)

        self.lin = nn.Linear(hidden_size, vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward(self, x, hidden):
        """
        Forward pass, accepts one input (x), and the hidden state.

        Returns the output (a log probability distribution
        over the vocabulary), and the new hidden state

        :param x: token tensor of shape (1,)
        :param hidden: hidden tensor
        :returns: output, hidden_state
        """
        embedded = self.embedding(x)
        output, hidden = self.rnn(embedded, hidden)
        output = self.lin(output)
        output = self.log_softmax(output)
        return output, hidden

    def init_hidden(self):
        """
        Return initial hidden tensor. The shape depends on the rnn type.
        For RNN and GRU the hidden state shape is (1, hidden_size).

        :return: hidden state tensor
        """
        if self.rnn_type == USE_LSTM:
            return torch.randn(2, 1, self.hidden_size)
        else:
            return torch.randn(1, self.hidden_size)

