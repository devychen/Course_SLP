"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    (Enter the assignment number - e.g. A1)
Author(s):     (Enter the full names of author(s) here)

Honor Code:    I/We pledge that this program represents my/our own work,
               and that I/we have not given or received unauthorized help
               with this assignment.
"""
import spacy
import compress_fasttext
import torch
import numpy as np
import json
from constants import *


class Preprocessor:
    def __init__(self, spacy_model, embeddings):
        """
        Class for doing A2 text preprocessing and data preparation tasks, including:

            - loading a train, dev, or test file (csv)
            - preprocessing texts with spacy model
            - generating X and y tensors using embeddings
            - saving the generated tensors to files, to avoid the long preprocessing
                step on each training run

        Public methods include:
            - load_data()
            - generate_tensors()
            - save_tensors()

        :param spacy_model: loaded spaCy model (de_core_news_sm)
        :param embeddings: loaded word embeddings (fasttext-de-mini)
        """

        # set class variables passed in
        self.nlp = spacy_model
        self.embeddings = embeddings

        # These class variables will be set by the preprocessing methods
        self.X_texts = None  # list[str] containing X (input) textual data
        self.y_texts = None  # list[str] containing y (label) textual data
        self.X_tensor = None  # tensor of shape (n_samples, embedding_dim)
        self.y_tensor = None  # tensor of shape (n_samples,)
        self.label_map = None  # dictionary containing {label:class_code} mapping

    def _load_csv_data(self, data_file):
        """
        Private method to load csv data from data_file.
        Set class variable self.X_texts to a list[str] of texts,
        and self.y_texts to a list[str] of the corresponding
        topics that are read from data_file.
        Helper function for load_data().

        Each line in the input file is the form: topic;text

        Note: Input files are not well-formed csv, even though they use the .csv extension.
        It is recommended to read the file line-by-line. Remove leading and trailing
        whitespace, including newlines.

        :param data_file: file with lines of the form topic;text
        """
        pass

    def _load_label_map(self, label_map_file):
        """
        Private method to load the json in label_map_file into the
        class dictionary self.label_map {label:class_code}.
        Helper function for load_data().

        :param label_map_file: json file containing label mapping
        """
        pass

    def load_data(self, data_file, label_map_file):
        """
        Public function to read and store the textual data (in data_file),
        as well as the label map (in label_map_file).

        :param data_file: csv data_file containing lines of form topic;text
        :param label_map_file: json file containing mapping from labels to class codes
        """
        pass

    def _preprocess_text(self, text):
        """
        Private function to preprocess one text.
        Uses the spaCy model in self.nlp to remove stopwords, punctuation,
        whitespace, and tokens that represent numbers (e.g. “10.9”, “10”, “ten”).
        Helper function for _calc_mean_embedding().

        :param text: str : one text
        :return: list[str] : list of preprocessed token strings
        """
        pass

    def _calc_mean_embedding(self, text):
        """
        Private function to calculate the mean embedding, as a tensor of
        dtype torch.float32, for the given text.
        Helper function for _generate_X_tensor()

        Returns the mean of the word vectors for the words in text_tokens. Word
        vectors are retrieved from the embeddings in self.embeddings.
        Words that are not contained in the embeddings are ignored.

        :param text: str containing one text
        :return: mean vector, as a tensor of type torch.float32, of the text tokens contained in the embeddings
        """
        pass

    def _generate_X_tensor(self):
        """
        Private function to create a tensor of shape (len(self.X_texts), embedding_dim),
        from the texts in class variable self.X_texts (previously set in load_data()).
        Generate the tensor and store in self.X_tensors.
        Each row in self.X_tensors represents the mean vector embedding for
        one text in self.X_texts.
        Helper function for generate_tensors().
        """
        pass

    def _generate_y_tensor(self):
        """
        Private method to create a tensor for the gold data class codes.
        Using the label mapping in the class variable self.label_map,
        generate a tensor for the gold data stored in self.y_texts.
        Both self.label_map and self.y_texts were previously set in load_data().
        Store the generated tensor of shape (len(self.y_texts)) in
        class variable self.y_tensor.
        Helper function for generate_tensors().

        Note that the model can't predict string labels such as 'Sport' or 'Kultur', so the
        labels must be encoded as integers for training. The model predictions are the integer
        label codes, which are converted back to their string label values for humans.
        """
        pass

    def generate_tensors(self):
        """
        Public function to generate tensors for use in a neural network
        from data previously loaded by load_data().

        Generate X and y tensors from data in self.X_texts and self.y_texts.
        """
        pass

    def save_tensors(self, tensor_file):
        """
        Public function to save generated tensors X and y,
        along with the label_map, to tensor_file.

        Create dictionary with keys X_KEY, Y_KEY, MAP_KEY (defined in constants.py),
        with values self.X_tensor, self.y_tensor, and self.label_map, respectively.

        Note: You can use torch.save() to save the dictionary.
        By convention, PyTorch files end with file extension .pt.

        A saved tensor file can later be quickly loaded for use in training or evaluation.
        Loading the tensors from file is much faster than preprocessing texts and
        converting them to tensors for each training or evaluation.

        :param tensor_file: name of the file to save
        """
        pass


if __name__ == '__main__':
    """
    Create tensor files for train, dev, test data here.
    Save in Data directory.
    """
    pass
