"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    A3
Author(s):     Yifei Chen

Honor Code:    I/We pledge that this program represents my/our own work,
               and that I/we have not given or received unauthorized help
               with this assignment.
"""

# implement load_tsv_data() to read training data in data/train.tsv

import csv
from constants import BOS_STR, EOS_STR


def load_tsv_data(data_file):
    """
    Reads data_file containing tab separated input \t output columns.
    Skip header row.
    For all remaining lines, split() input phones and output text.
    Return lst[(lst[str], lst[str])], a list of tuples, where each tuple
    contains two lists of strings, one for the input and one for the output.

    The first list (the input language phonemes), should have both BOS_STR and EOS_STR markers.
    The second list (the output language English), should have only the EOS_STR marker.

    Helper for Trainer and Evaluator.

    :param data_file: two-column, tab-separated file
    :return: lst[(lst[str], lst[str])] containing input/output string lists with sentence markers as described.
    """
    data = []

    with open(data_file, newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader) # skip header row

        for row in reader:
            input_phones = row[0].split()
            output_text = row[1].split()

            input_with_markers = [BOS_STR] + input_phones + [EOS_STR]
            output_with_marker = output_text + [EOS_STR]
            
            data.append((input_with_markers, output_with_marker))
    
    return data 
