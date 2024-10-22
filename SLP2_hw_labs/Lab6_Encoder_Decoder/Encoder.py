"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    (Enter the assignment number - e.g. A1)
Author(s):     (Enter the full names of author(s) here)

Honor Code:    I/We pledge that this program represents my/our own work,
               and that I/we have not given or received unauthorized help
               with this assignment.
"""

import torch
import torch.nn as nn


class Encoder(nn.Module):
    """
    The encoder of a seq2seq network is an RNN that takes each token in a
    sequence and the final hidden tensor is an encoding of the entire input.

    The final hidden tensor is used as the initial hidden tensor of the decoder.
    """
    def __init__(self, input_vocab_size, hidden_size, bidirectional=False):
        super(Encoder, self).__init__()
        self.hidden_size = hidden_size
        self.bidirectional = bidirectional

        self.embedding = nn.Embedding(input_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True, bidirectional=self.bidirectional)

    def forward_step(self, x, hidden):
        """
        One step of a forward pass.
        Returns:
        1: the output (1, hidden_size * directions)
        2: the new hidden tensor of shape (directions, hidden_size)

        where directions = 2 if bidirectional, else 1

        :param x: token tensor of shape (1,)
        :param hidden: hidden tensor of shape (directions, hidden_size)
        :return: output of shape (1, hidden_size * directions), hidden tensor of shape (directions, hidden_size)
        """
        pass

    def forward(self, x):
        """
        Full forward pass, using the output and hidden state
        returned by forward_step() for the next iteration.

        Use initHidden() to create the initial hidden tensor.

        Iterate each element of the input x (of shape (1,)),
            calling forward_step() with the current input and hidden tensor.
            Collect encoder outputs at each step in a list.

        If bidirectional, sum the hidden state along dimension 1 to get a tensor of shape (1, hidden_size)

        Return encoder outputs, list of tensors of shape (1, hidden_size * directions),
        and final hidden tensor (1, hidden_size)

        :param x: tensor of shape (num_input_tokens,)
        :return: encoder outputs (len(x), hidden_size * directions), and final hidden tensor (1, hidden_size)
        """

        # initialize hidden state
        pass

        # iterate token tensors in x
        pass

        # if bidirectional, sum hidden states of both directions.
        # hidden shape (2, hidden_size) --> (1, hidden_size)
        pass

        # return values
        pass

    def initHidden(self):
        """
        Return initial hidden state, tensor of zeros.
        If bidirectional, shape (2, hidden_size),
        otherwise shape (1, hidden_size)

        :return: initial hidden state tensor of zeros
        """
        pass
