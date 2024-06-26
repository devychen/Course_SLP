"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    A3
Author(s):     Yifei Chen

Honor Code:    I/We pledge that this program represents my/our own work,
               and that I/we have not given or received unauthorized help
               with this assignment.
"""

import torch
from torcheval.metrics.functional.text.bleu import bleu_score
from constants import *
from Encoder import Encoder
from Decoder import Decoder
from lang import Lang
from utils import load_tsv_data


class Evaluator:
    """
    Class for loading an encoder-decoder model from files,
    and evaluating on dev data.

    Public methods include:
        - load_model()
        - load_data()
        - evaluate()
    """

    def __init__(self):
        # class variables
        self.input_lang = None
        self.output_lang = None
        self.encoder = None
        self.decoder = None
        self.io_token_pairs = None

        # You may want to add a few more class variables
        # for model metadata, so you can print them during evaluation.

    def load_model(self, model_file):
        """
        Loads an encoder/decoder model from model_file, as saved in trainer.
        All states required to create the encoder, decoder, and Lang objects
        should be contained in model_file:

        INPUT_LANG_STATE_KEY
        OUTPUT_LANG_STATE_KEY
        ENCODER_STATE_KEY
        DECODER_STATE_KEY
        HIDDEN_SIZE_KEY
        BIDIRECTIONAL_KEY # True/False

        The following keys should also be present, but are not
        necessary to instantiate the model for evaluation:
        TFR_KEY   # teacher_forcing_ratio
        LOSS_KEY
        N_EPOCHS_KEY
        LEARNING_RATE_KEY

        Instantiate an encoder, a decoder, as well as
        an input Lang and an output Lang.
        Set the appropriate class variables.

        :param model_file: model as saved after training
        """
        model_data = torch.load(model_file, map_location=torch.device('cpu'))

        # Load input and output languages
        self.input_lang = Lang()
        self.input_lang.load_state_dict(model_data[INPUT_LANG_STATE_KEY])

        self.output_lang = Lang()
        self.output_lang.load_state_dict(model_data[OUTPUT_LANG_STATE_KEY])

        # Load model architecture and weights
        self.hidden_size = model_data[HIDDEN_SIZE_KEY]
        self.bidirectional = model_data[BIDIRECTIONAL_KEY]

        self.encoder = Encoder(self.input_lang.vocab_size, self.hidden_size, self.bidirectional)
        self.encoder.load_state_dict(model_data[ENCODER_STATE_KEY])

        self.decoder = Decoder(self.hidden_size, self.output_lang.vocab_size)
        self.decoder.load_state_dict(model_data[DECODER_STATE_KEY])

    def load_data(self, data_file):
        """
        Reads data_file containing tab separated input \t output columns.
        See utils.load_tsv_data().
        Store token pairs in self.io_token_pairs

        :param data_file: two-column, tab-separated file
        """
        self.io_token_pairs = load_tsv_data(data_file)


    def _evaluate_sentence(self, sentence, target):
        """
        Evaluate and return the model prediction of one sentence
        and the bleu score (use the imported bleu_score function) calculated
        against target, with n-gram=1.
        Helper for evaluate().

        Note that bleu_score() requires inputs as strings, so you
        need to join the tokens.

        :param sentence: sentence as a list[str] of tokens
        :param target: target translation as a list[str] of tokens
        :return: model prediction as a list[str], and the bleu score
        """

        with torch.no_grad():
            input_tensor = self.input_lang.words_to_tensor(sentence).unsqueeze(0)
            target_tensor = self.output_lang.words_to_tensor(target).unsqueeze(0)

            encoder_hidden = self.encoder.initHidden()
            encoder_outputs, encoder_hidden = self.encoder(input_tensor)

            decoder_outputs, decoder_hidden = self.decoder(encoder_hidden, target_tensor)
            _, topi = decoder_outputs.topk(1)
            decoded_words = [self.output_lang.index2word[idx.item()] for idx in topi.squeeze()]

            # Calculate BLEU score
            bleu = bleu_score(' '.join(decoded_words), ' '.join(target))

        return decoded_words, bleu

    def evaluate(self, verbose=False):
        """
        Calculate the bleu score for each pair in self.io_token_pairs.
        Return the average bleu score over all pairs.

        :param verbose: if True, print input, target, translation, score for each sentence
        :return: average bleu score over all pairs in self.io_token_pairs
        """
        total_bleu = 0.0
        num_pairs = len(self.io_token_pairs)

        for input_tokens, target_tokens in self.io_token_pairs:
            translation, bleu = self._evaluate_sentence(input_tokens, target_tokens)
            total_bleu += bleu

            if verbose:
                print(f'Input:    {" ".join(input_tokens)}')
                print(f'Target:   {" ".join(target_tokens)}')
                print(f'Prediction: {" ".join(translation)}')
                print(f'BLEU Score: {bleu:.4f}\n')

        average_bleu = total_bleu / num_pairs if num_pairs > 0 else 0.0
        return average_bleu
        


def main():
    """Load and evaluate your models here"""
    evaluator = Evaluator()
    evaluator.load_model('HW3/Models/seq2seq_model.pth')
    evaluator.load_data('HW3/Data/dev.tsv')
    average_bleu = evaluator.evaluate(verbose=True)

    print(f'Avg. BLEU Score: {average_bleu:.4f}')
    


if __name__ == '__main__':
    main()


# You should get a bleu score ≈ .4 for both.