from gensim.models import KeyedVectors
import compress_fasttext
import matplotlib.pyplot as plt
import numpy as np
from sklearn.manifold import TSNE
from enum import Enum
plt.rcParams["figure.figsize"] = (8, 5.5)


class Embedding(Enum):
    GOOGLE = 1
    FASTTEXT_DE_MINI = 2


def load_embeddings(embedding_type):
    emb = None
    match embedding_type:
        case Embedding.GOOGLE:
            # Load 200000 GoogleNews Vectors
            emb = KeyedVectors.load_word2vec_format(
            'GoogleNews-vectors-negative300.bin.gz',
            binary=True,
            limit=200000)

        case Embedding.FASTTEXT_DE_MINI:
            # Load FastText German Compressed
            emb = compress_fasttext.models.CompressedFastTextKeyedVectors.load("fasttext-de-mini")
    return emb


def plot_most_similar(emb, word, size):
    """
    Generate a 2D plot of words similar to 'word' using T-SNE for dimension reduction.

    :param emb: word embeddings
    :param word: word string
    :param size: number of dimensions in the model
    """
    arr = np.empty((0, size), dtype='f')
    word_labels = [word]
    close_words = emb.most_similar(word, topn=12)
    arr = np.append(arr, np.array([emb[word]]), axis=0)
    for wrd_score in close_words:
        wrd_vector = emb[wrd_score[0]]
        word_labels.append(wrd_score[0])
        arr = np.append(arr, np.array([wrd_vector]), axis=0)

    tsne = TSNE(n_components=2, random_state=0, perplexity=5.0)
    np.set_printoptions(suppress=True)
    Y = tsne.fit_transform(arr)
    x_coords = Y[:, 0]
    y_coords = Y[:, 1]
    plt.scatter(x_coords, y_coords)
    for label, x, y in zip(word_labels, x_coords, y_coords):
        if label == word:
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=20, color='red')
        else:
            plt.annotate(label, xy=(x, y), xytext=(0, 0), textcoords='offset points', fontsize=20)
    plt.xlim(x_coords.min() - 10, x_coords.max() + 10)
    plt.ylim(y_coords.min() - 10, y_coords.max() + 10)
    plt.title("T-SNE: Most similar words to: '{}'\n".format(word), fontsize=18)
    plt.show()


def prompt_vectors(emb):
    """
    Repeatedly get words from the user (using the input() function) and
    print the vector for the word, or "Word not found" if the input word
    is not contained in the embeddings.
    Stop when the user enters "quit".

    :param emb: the embeddings
    :return: None
    """

    while True:
        prompt = f"\nEnter a word to see its embedding: "
        word = input(prompt)
        if word == "quit":
            break

        # ToDo: Add your code here
        if word in emb:
            print(f"Embedding for '{word}': {emb[word]}")
        else:
            print("Word not found")


def prompt_most_similar(emb, num_results):
    """
    Repeatedly get words from the user (using the input() function) and
    print the top 'num_results' most similar words contained in the
    embeddings.
    In addition to printing similar words, also plot the vectors of
    the similar words using the plot_most_similar() function above.
    Stop when the user enters "quit".

    :param emb: the embeddings
    :param num_results: number of results to print
    :return: None
    """
    ## Add your code here
    while True:
        prompt = f"\nEnter a word to find {num_results} most similar words: "
        word = input(prompt)
        if word == "quit":
            break

        if word in emb:
            similar_words = emb.most_similar(word, topn=num_results)
            print(f"Most similar words to '{word}':")
            for similar_word, _ in similar_words:
                print(similar_word)
            plot_most_similar(emb, word, emb.vector_size)
        else:
            print("Word not found")




def prompt_similarity(emb):
    """
    Repeatedly get 2 words from the user and
    print their cosine similarity.
    Stop when the user enters "quit" as the first word.

    :param emb: the embeddings
    :return: None
    """
    # Add your code here
    while True:
        word1 = input("\nEnter the first word: ")
        if word1 == "quit":
            break

        word2 = input("Enter the second word: ")

        if word1 in emb and word2 in emb:
            similarity_score = emb.similarity(word1, word2)
            print(f"Cosine similarity between '{word1}' and '{word2}': {similarity_score}")
        else:
            print("Word not found")


def main():
    emb = load_embeddings(Embedding.GOOGLE)
    #prompt_vectors(emb)
    #prompt_most_similar(emb, 12)
    #prompt_similarity(emb)


if __name__ == '__main__':
    main()
