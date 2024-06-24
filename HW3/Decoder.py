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
import random
from constants import MAX_LENGTH, BOS_IDX, EOS_IDX


class Decoder(nn.Module):
    """
    The decoder is an RNN that takes the last encoder hidden tensor and
    outputs a sequence of words to create the translation.
    """
    def __init__(self, hidden_size, output_vocab_size, teacher_forcing_ratio=0):
        super(Decoder, self).__init__()

        self.teacher_forcing_ratio = teacher_forcing_ratio
        self.hidden_size = hidden_size

        self.embedding = nn.Embedding(output_vocab_size, hidden_size)
        self.gru = nn.GRU(hidden_size, hidden_size, batch_first=True)
        self.lin = nn.Linear(hidden_size, output_vocab_size)
        self.log_softmax = nn.LogSoftmax(dim=1)

    def forward_step(self, x, hidden):
        """
        One step of a forward pass.
        Returns 1: the output (1, out_vocab_size), which is a log probability distribution
        over the output vocabulary, and 2: the new hidden state

        :param x: token tensor of shape (1,)
        :param hidden: hidden tensor of shape (1, hidden_size)
        :returns: output of shape (1, output vocab_size), hidden_state of shape (1, hidden_size)
        """
        embedded = self.embedding(x)
        output, hidden = self.gru(embedded, hidden)
        output = self.lin(output) # linear layer before logsoftmax
        output = self.log_softmax(output) # applied logsoftmax after linear layer
        output = output.view(1,-1)
        return output, hidden


    def forward(self, encoder_hidden, target_tensor=None):
        """
        Full forward pass, using the output and hidden state
        returned by forward_step() for the next iteration.

        Create a tensor containing BOS_IDX as the initial input.
        Collect decoder outputs at each step in a list.

        If teacher_forcing_ratio > 0 and target_tensor is provided:
        Use the gold tokens in target_tensor as inputs at each step
        (instead of the decoder outputs)
        for ~ teacher_forcing_ratio % of the forward passes.

        Otherwise:
        Use the decoder's own predictions as inputs at each step.
        The predicted token index is the index of the highest value in
        the log probability distribution. Hint: argmax along dim 1
        Stop when the decoder output is the EOS token, or MAX_LENGTH is reached.

        Before returning the collected decoder outputs, concatenate
        decoder outputs along dim 0 to remove batch_size dimension
        (we are not using this dimension, since our batch_size=1).

        Return decoder outputs (num_outputs, output vocab_size), and decoder_hidden (1, hidden_size)

        :param encoder_hidden: tensor of shape (1, hidden_size)
        :param target_tensor: optional for training with teacher forcing -
            gold output of shape (num_gold_tokens, 1) - used only if teacher_forcing_ratio > 0
        :return: decoder outputs (num_outputs, output vocab_size), and decoder_hidden (1, hidden_size)
        """

        # initialize decoder input and hidden tensors
        decoder_input = torch.tensor([BOS_IDX]) #, device=encoder_hidden.device)
        decoder_hidden = encoder_hidden

        decoder_outputs = []

        # if self.teacher_forcing_ratio > 0 and target_tensor is provided,
        # use teacher forcing for ~ teacher_forcing_ratio % of the inputs
        use_teacher_forcing = False
        if (target_tensor is not None) and (random.random() < self.teacher_forcing_ratio):
            use_teacher_forcing = True

        if use_teacher_forcing:
            for target_token in target_tensor:
                output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs.append(output)
                decoder_input = target_token  # Next input is current target

        else:
            for _ in range(MAX_LENGTH):
                output, decoder_hidden = self.forward_step(decoder_input, decoder_hidden)
                decoder_outputs.append(output)
                top1 = output.argmax(1)
                decoder_input = top1  # Next input is current prediction

                if decoder_input.item() == EOS_IDX:
                    break

        # Combine decoder outputs (list[tensor of shape (1, output vocab size)])
        # into one tensor of shape (len(decoder_outputs), output_vocab_size)
        # Each "row" represents log probabilities over the output vocabulary
        # for one output token
        decoder_outputs = torch.cat(decoder_outputs, dim=0)

        # return values
        return decoder_outputs, decoder_hidden
