"""
 Course:        Statistical Language processing - Summer 2024
 Assignment:    A1
 Author(s):     Yifei Chen

 Honor Code:    I pledge that this program represents my own work,
                and that I have not given or received unauthorized help
                on this assignment.
"""

from gensim.models import KeyedVectors
# Successfully installed gensim-4.3.2, using pip install gensim
import spacy
import numpy as np
from numpy import dot
from numpy.linalg import norm
import pandas as pd
import csv # not given in starter code

# download and load the spacy english small model,
# disable parser and ner pipes
print('Loading spaCy...')
nlp = spacy.load("en_core_web_sm", disable = ["parser", "ner"])

# load word2vec embeddings,
# which are located in the same directory as this script.
# Limit the vocabulary to 100,000 words
# - should be enough for this application
# - loading all takes a long time
print('Loading word2vec embeddings...')
word2vec_file_path = 'HW1/GoogleNews-vectors-negative300.bin.gz'
word2vec_model = KeyedVectors.load_word2vec_format(word2vec_file_path, binary=True, limit=100000) # Load Word2Vec model

def load_data(filename):
    """
    Load the Ted Talk data from filename and extract the
    "description" and "url" columns. Return a dictionary of dictionaries,
    where the keys of the outer dictionary are unique integer values which
    will be used as IDs.
    Each inner dictionary represents a row in the input file and contains
    the keys 'description', and 'url'.

    :param filename: input filename in csv format
    :return: dict of dicts, where the inner dicts represent rows in the input
    file, and the keys in the outer dict serve as IDs.
    """
    data = {}
    with open(filename, 'r', encoding='utf-8') as file:
        csv_reader = csv.DictReader(file)
        for index, row in enumerate(csv_reader): 
            data[index] = {'description': row['description'], 'url': row['url']}
    return data
    # csv_reader is a variable representing an instance of the csv.DictReader class.,
    # an iterator that we use to read rows from the CSV file one by one, 
    # with each row being represented as a dictionary.


def preprocess_text(text):
    """
    Preprocess one text. Helper function for preprocess_texts().

    Preprocessing includes lowercasing and removal of stopwords,
    punctuation, whitespace, and urls.

    The returned list of tokens could be an empty list.

    :param text: one text string
    :return: list of preprocessed token strings. May be an empty list if all tokens are eliminated.
    """
    text = text.lower()
    tokens = nlp(text) # tokenisation
    preprocessed_tokens = []
    for token in tokens:
        if not token.is_stop and not token.is_punct and not token.is_space and not token.like_url:
            preprocessed_tokens.append(token.text)
    
    return preprocessed_tokens


def preprocess_texts(data_dict):
    """
    Preprocess the description in each inner dict of data_dict by
    lowercasing and removing stopwords, punctuation, whitespace, and urls.
    The list of token strings for an individual text is not a set,
    and therefore may contain duplicates. Add a new key 'pp_text'
    to each inner dict, where the value is a list[str] representing
    the preprocessed tokens the text.

    :param data_dict: a nested dictionary with a structure as returned by load_data()
    :return: the input data_dict, with key 'pp_text' of preprocessed token strings for
    each description
    """
    for key, value in data_dict.items():
        description = value['description']
        preprocessed_description = preprocess_text(description)
        value['pp_text'] = preprocessed_description
    return data_dict


def get_vector(tokens):
    """
    Calculate a single vector for the preprocessed word strings in tokens.
    The vector is calculated as the mean of the word2vec vectors for the
    words in tokens. Words that are not contained in the word2vec pretrained
    embeddings are ignored. If none of the tokens are contained in word2vec,
    return None.

    :param tokens: list of strings containing the preprocessed tokens
    :return: mean of the word2vec vectors of the words in tokens, or None
    if none of the tokens are in word2vec.
    """
    vectors = []
    for token in tokens:
        if token in word2vec_model:
            vectors.append(word2vec_model[token])
    
    if vectors:
        return np.mean(vectors, axis=0)
    else:
        return None


def get_vectors(data_dict):
    """
    Calculate the vector of the preprocessed text 'pp_text' in each
    inner dict of data_dict. Add a new key 'vector'
    to each inner dict, where the value is the mean of individual word vectors
    as returned by get_vector().

    If 'pp_text' is an empty list, or none of the words in 'pp_text' are
    in word2vec, the value of 'vector' is None.

    :param data_dict: a nested dictionary where inner dicts have key 'pp_text'
    :return: the input data_dict, with key 'vector' for each inner dict
    """
    for key, value in data_dict.items():
        tokens = value['pp_text']
        vector = get_vector(tokens)
        value['vector'] = vector
    return data_dict


def cosine_similarity(v1, v2):
    """
    Calculate the cosine similarity of v1 and v2.

    :param v1: vector 1
    :param v2: vector 2
    :return: cosine similarity
    """
    if v1 is None or v2 is None:
        return None
    
    dot_product = 0
    norm_v1 = 0
    norm_v2 = 0
    
    for i in range(len(v1)):
        dot_product += v1[i] * v2[i]
        norm_v1 += v1[i] ** 2
        norm_v2 += v2[i] ** 2
    
    if norm_v1 == 0 or norm_v2 == 0:
        return None
    
    return dot_product / (np.sqrt(norm_v1) * np.sqrt(norm_v2))


def k_most_similar(query, data_dict, k=5):
    """
    Find the k most similar entries in data_dict to the query.

    The query is first preprocessed, then a mean word vector is calculated for it.

    Return a list of tuples of length k where each tuple contains
    the id of the data_dict entry and the cosine similarity score between the
    data_dict entry and the user query.

    In some cases, the similarity cannot be calculated. For example:
    - If none of the preprocessed token strings are in word2vec for an entry in data_dict.
    If you built data_dict according to the instructions, the value of 'vector'
    is None in these cases, and those entries should simply not be considered.
    - If a vector for the query can't be calculated, return an empty list.

    :param query: a query string as typed by the user
    :param data_dict: a nested dictionary where inner dicts have key 'vector'
    :param k: number of top results to return
    :return: a list of tuples of length k, each containing an id and a similarity score,
    or an empty list if the query can't be processed
    """
    query_tokens = preprocess_text(query)
    
    if not query_tokens:
        return []
    
    query_vector = get_vector(query_tokens)
    
    if query_vector is None:
        return []
    
    similarities = []
    for key, value in data_dict.items():
        if value['vector'] is not None:
            sim = cosine_similarity(query_vector, value['vector'])
            if sim is not None:
                similarities.append((key, sim))
    
    similarities.sort(key=lambda x: x[1], reverse=True)
    
    return similarities[:k]


def recommender_app(data_dict):
    """
    Implement your recommendation system here.

    - Repeatedly prompt the user to type a query
        - Print the description and the url of the top 5 most similar,
        or "No Results Found" if appropriate
        - Return when the query is "quit" (without quotes)

    :param data_dict: nested dictionaries containing
    description,url,tokens,and vectors for each description
    in the input data
    """
    while True:
        query = input("Please enter a query (type 'quit' to exit): ")
        if query.lower() == 'quit':
            break
        else:
            similarities = k_most_similar(query, data_dict)
            if not similarities:
                print("No Results Found")
            else:
                print("Top 5 Most Similar:")
                for i, (id_, sim) in enumerate(similarities, start=1):
                    print(f"{i}. Description: {data_dict[id_]['description']}")
                    print(f"   URL: {data_dict[id_]['url']}")
                    print(f"   Similarity Score: {sim}")
                    print()


def main():
    """
    Bring it all together here.
    """
    pass


if __name__ == '__main__':
    main()