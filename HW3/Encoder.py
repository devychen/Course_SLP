"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    A3
Author(s):     Yifei Chen

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
        embedded = self.embedding(x).view(0) # shape: (1, 1, hidden_size)
        output, hidden = self.gru(embedded, hidden)
        # if self.bidirectional:
        #     output = output[:, :, :self.hidden_size] + output[:, :, self.hidden_size:]
        return output, hidden

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
        hidden = self.initHidden()
        
        # iterate token tensors in x
        outputs = []
        for token in x:
            output, hidden = self.forward_step(token.unsqueeze(0), hidden)
            outputs.append(output)
        
        outputs = torch.cat(outputs, dim=0)

        # if bidirectional, sum hidden states of both directions.
        # hidden shape (2, hidden_size) --> (1, hidden_size)
        if self.bidirectional:
            #? hidden = hidden.view(2, 1, self.hidden_size).sum(dim=0, keepdim=True).view(1, self.hidden_size)
            hidden = torch.sum(hidden, dim=0, keepdim=True)  # sum the bidirectional hidden states
        # return values
        return outputs, hidden

    def initHidden(self):
        """
        Return initial hidden state, tensor of zeros.
        If bidirectional, shape (2, hidden_size),
        otherwise shape (1, hidden_size)

        :return: initial hidden state tensor of zeros
        """
        directions = 2 if self.bidirectional else 1
        return torch.zeros(directions, self.hidden_size)
