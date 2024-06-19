"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    (Enter the assignment number - e.g. A1)
Author(s):     (Enter the full names of author(s) here)

Honor Code:    I/We pledge that this program represents my/our own work,
               and that I/we have not given or received unauthorized help
               with this assignment.
"""
import torch
import json


class Lang:
    """
    Utility class to work with language vocabulary.
    Stores {word:idx} and {idx:word} maps, and has
    methods for converting words to tensors,
    and for converting tensors to words.
    Two Lang objects are needed - one for the input language (phones),
    and one for the output language (english).
    """
    def __init__(self):
        self.word2index = None  # {word:idx} map
        self.index2word = None  # {idx:word} map
        self.vocab_size = None

    def load_vocab(self, vocab_file):
        """
        Load the {word:idx} mapping in vocab_file into self.word2index.
        Create the reverse mapping {idx:word} and store in self.index2word.
        Set self.vocab_size.

        :param vocab_file: json file containing {word:idx} mapping
        """
        pass

    def words_to_tensor(self, word_list):
        """
        Create a tensor of type torch.long of shape (len(word_list), 1)
        containing the indices of the words in word_list.

        Examples:
        word_list: ['<s>', 'ɑː', 'ɹ', 'w', 'iː', 'ɡ', 'ʊ', 'd', '</s>']
        returns: tensor([[0], [28], [5], [17], [19], [27], [30], [7], [1]])

        word_list: ['are', 'we', 'good', '?', '</s>']
        tensor([[30], [18], [79], [3], [1]])

        :param word_list: list[str] of words to convert
        :return: the generated tensor
        """
        pass

    def tensor_to_words(self, word_tensor):
        """
        Converts a tensor of shape (num_words,), containing indices,
        to a list of tokens of length num_words.

        Example:
        word_tensor: tensor([30, 18, 79, 3, 1])
        returns: ['are', 'we', 'good', '?', '</s>']

        :param word_tensor: tensor of shape (num_words,)
        :return: list[str] decoded words as list
        """
        pass

    def get_state_dict(self):
        """
        Return the state dictionary of this Lang object, with the following keys:
        "word2index"
        "index2word"
        "vocab_size"

        :return: dictionary containing the state of this Lang
        """
        pass

    def load_state_dict(self, state_dict):
        """
        Set all class variables to the values in state_dict, as returned by get_state_dict()

        :param state_dict: dictionary containing the state of this Lang
        """
        pass
