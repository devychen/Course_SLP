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
from torch import optim
import time
import copy
from constants import *
from Encoder import Encoder
from Decoder import Decoder
from utils import load_tsv_data
from lang import Lang


class Trainer():
    def __init__(self, input_lang, output_lang,
                 encoder, decoder,
                 encoder_optimizer, decoder_optimizer,
                 criterion):

        self.input_lang = input_lang
        self.output_lang = output_lang
        self.encoder = encoder
        self.decoder = decoder
        self.encoder_optimizer = encoder_optimizer
        self.decoder_optimizer = decoder_optimizer
        self.criterion = criterion

        self.train_io_token_pairs = None  # lst[(lst[str], lst[str])]
        self.train_io_tensor_pairs = None  # lst[(tensor, tensor)]

    def _generate_io_tensor_pairs(self, io_token_pairs):
        """
        For each tuple element in io_token_pairs lst[(lst[str], lst[str])],
        generate the corresponding input and output tensors.
        Helper for load_data().

        Hint: use the input and output Lang objects to create the tensors.

        :param io_token_pairs: lst[(lst[str], lst[str])]
        :return: lst[(tensor, tensor)]
        """
        io_tensor_pairs = []
        for input_tokens, output_tokens in io_token_pairs:
            input_tensor = self.input_lang.words_to_tensor(input_tokens)
            output_tensor = self.output_lang.words_to_tensor(output_tokens)
            io_tensor_pairs.append((input_tensor, output_tensor))
        return io_tensor_pairs

    def load_data(self, train_data_file):
        """
        Load training data, see utils.load_tsv_data().
        Create io_token_pairs and io_tensor_pairs for the training data.
        Store the io poirs in the appropriate class variables.

        :param train_data_file: tab separated input/output phrases for training
        """
        self.train_io_token_pairs = load_tsv_data(train_data_file)
        self.train_io_tensor_pairs = self._generate_io_tensor_pairs(self.train_io_token_pairs)

    def _train_one(self, input_tensor, target_tensor):
        """
        Train using one input/output pair,
        given as input and target tensors.
        The encoder and decoder, as well as their optimizers,
        and criterion (i.e. loss function) were passed to the constructor.

        :param input_tensor: input sentence as a tensor
        :param target_tensor: target sentence as a tensor
        :return: decoder_output and scalar loss
        """

        # zero optimizer gradients
        self.encoder_optimizer.zero_grad()
        self.decoder_optimizer.zero_grad()

        # forward pass on encoder and decoder
        # on encoder
        encoder_hidden = self.encoder.initHidden()
        encoder_outputs, encoder_hidden = self.encoder(input_tensor)
        # on decoder
        decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, target_tensor)

        # Create a view of target_tensor that removes the last dimension.
        # This is ok because the last dimension is always 1 (1 token index).
        # Example: target_tensor shape (5, 1) --> (5,)
        target_tensor = target_tensor.view(-1)

        # The number of decoder outputs must be the same as the number of tokens in target
        # in order to compute the loss.
        # Either chop off extra predictions, or pad with 0's
        length_diff = target_tensor.size(0) - decoder_outputs.size(0)
        if length_diff > 0:
            # decoder output is shorter than target - pad first dimension with 0s
            decoder_outputs = nn.functional.pad(decoder_outputs, pad=(0, 0, 0, length_diff),
                                                mode='constant', value=0)
        elif length_diff < 0:
            # decoder output is longer than target - remove outputs after target length
            indices = torch.tensor([i for i in range(target_tensor.size(0))])
            decoder_outputs = torch.index_select(decoder_outputs, 0, indices)

        # Compute loss for this sample.
        # NLLLoss expects two tensors:
        #   predictions: tensor of shape (target_tensor.size(0), output vocab_size) containing
        #       a log probability distribution for each output
        #   gold values: target tensor of shape (num_tokens,)
        loss = self.criterion(decoder_outputs, target_tensor)

        # backpropagation
        loss.backward()

        # update weights
        self.encoder_optimizer.step()
        self.decoder_optimizer.step()

        # return decoder output and scalar loss
        return decoder_outputs, loss.item()

    def train(self, n_epochs):
        """
        Train the encoder and decoder for n_epochs.

        Keep track of the best model, and return a dictionary
        with the following keys:
            LOSS_KEY
            ENCODER_STATE_KEY
            DECODER_STATE_KEY

        For this assignment, the best model is considered the one with
        the lowest loss, averaged over one epoch.

        :param n_epochs: number of epochs to train
        :return: dictionary containing the loss and states of the best model
        """
        best_model = {
            LOSS_KEY: float('inf'),
            ENCODER_STATE_KEY: None,
            DECODER_STATE_KEY: None,
        }

        for epoch in range(n_epochs):
            epoch_loss = 0
            for input_tensor, target_tensor in self.train_io_tensor_pairs:
                _, loss = self._train_one(input_tensor, target_tensor)
                epoch_loss += loss

            epoch_loss /= len(self.train_io_tensor_pairs)

            if epoch_loss < best_model[LOSS_KEY]:
                best_model[LOSS_KEY] = epoch_loss
                best_model[ENCODER_STATE_KEY] = copy.deepcopy(self.encoder.state_dict())
                best_model[DECODER_STATE_KEY] = copy.deepcopy(self.decoder.state_dict())

            print(f'Epoch {epoch+1}/{n_epochs}, Loss: {epoch_loss:.4f}')

        return best_model


def main():
    """
    Train and save your models here.
    Your baseline model should get a bleu score near the
    score for the given baseline model.


    - Create Lang objects for input and output languages.
    - Choose hyperparameter values. For the baseline model, these are:
        n_epochs = 20
        hidden_size = 8
        learning_rate = .01
        teacher_forcing_ratio = 0  # not used in baseline
        bidirectional = False

        It is recommended to use AdamW optimizers (the optimizer updates the weights and bias).

    - Create an encoder and a decoder, and an optimizer for each
    - Use nn.NLLLoss for the loss function

    - Train the model
    - Save the model:

        The model file should contain at least the following keys,
        but you can save additional keys if you'd like:

        ENCODER_STATE_KEY
        DECODER_STATE_KEY
        LOSS_KEY
        INPUT_LANG_STATE_KEY
        OUTPUT_LANG_STATE_KEY
        HIDDEN_SIZE_KEY
        BIDIRECTIONAL_KEY
        N_EPOCHS_KEY
        LEARNING_RATE_KEY
        TFR_KEY


    Later you can load and evaluate your model with Evaluator.
    """
    input_lang = Lang()
    output_lang = Lang()

    input_lang.load_vocab('HW3/data/input_vocab.json')
    output_lang.load_vocab('HW3/data/output_vocab.json')

    hidden_size = 8
    learning_rate = 0.01
    teacher_forcing_ratio = 0
    bidirectional = False
    n_epochs = 20

    encoder = Encoder(input_lang.vocab_size, hidden_size, bidirectional)
    decoder = Decoder(hidden_size, output_lang.vocab_size, teacher_forcing_ratio)

    encoder_optimizer = optim.AdamW(encoder.parameters(), lr=learning_rate)
    decoder_optimizer = optim.AdamW(decoder.parameters(), lr=learning_rate)
    criterion = nn.NLLLoss()

    trainer = Trainer(input_lang, output_lang, encoder, decoder, encoder_optimizer, decoder_optimizer, criterion)
    trainer.load_data('HW3/data/train.tsv')
    best_model = trainer.train(n_epochs)

    model_data = {
        ENCODER_STATE_KEY: best_model[ENCODER_STATE_KEY],
        DECODER_STATE_KEY: best_model[DECODER_STATE_KEY],
        LOSS_KEY: best_model[LOSS_KEY],
        INPUT_LANG_STATE_KEY: input_lang.get_state_dict(),
        OUTPUT_LANG_STATE_KEY: output_lang.get_state_dict(),
        HIDDEN_SIZE_KEY: hidden_size,
        BIDIRECTIONAL_KEY: bidirectional,
        N_EPOCHS_KEY: n_epochs,
        LEARNING_RATE_KEY: learning_rate,
        TFR_KEY: teacher_forcing_ratio,
    }

    torch.save(model_data, 'seq2seq_model.pth')


if __name__ == '__main__':
    main()
