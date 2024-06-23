import torch
import torch.nn as nn
from torch import optim
import random
from PredictNextModel import PredictNextModel
from constants import *


def prepare_data(training_sequences, word2idx):
    """
    Prepare train and target tensors for each sequence (sentence) in training_sequences.

    The model is trained to predict the next word in a sequence,
    so in the training tensor we use BOS (to start the process), but not EOS
    and in the target tensor we don't have BOS (the model will not predict BOS,
    but we do have EOS (the model should predict EOS to end a sequence).

    In this way, the target matches what the model should predict.

    Example:
        sentence: "This is an example ."
        train: [[BOS_idx] [This_idx] [is_idx] [an_idx] [example_idx] [._idx]]
        target: [[This_idx] [is_idx] [an_idx] [example_idx] [._idx] [EOS_idx]]

    :param seq: list[list[str]] containing training sentences
    :return: list of tuples [(train_tensor, target_tensor)]
    """

    training_tensors = []
    for seq in training_sequences:
        # get the indices of the words in the vocab
        idxs = [word2idx[w] for w in seq]

        # add BOS for training
        train_idxs = [word2idx[BOS]] + idxs

        # add EOS for target
        target_idxs = idxs + [word2idx[EOS]]

        # convert to tensors
        train_tensor = torch.tensor(train_idxs, dtype=torch.long).view(-1, 1)
        target_tensor = torch.tensor(target_idxs, dtype=torch.long)

        training_tensors.append((train_tensor, target_tensor))

    return training_tensors


def get_training_data(filename):
    """
    Generate training sentences (one per line in file),
    and word2idx, idx2word maps.

    Assumes that tokens are separated by spaces in file.

    :param filename: filename containing one sentence per line.
    :return: list[(train_tensor, target_tensor)] of training data, word2idx, idx2word
    """

    with open(filename, "r", encoding="utf-8") as corpus:
        sentences = corpus.readlines()

    word2idx = {BOS: 0, EOS: 1}

    training_sents = []
    for sent in sentences:
        split_sent = sent.split()
        training_sents.append(split_sent)

        # assign each word a unique index
        for word in split_sent:
            if word not in word2idx:
                word2idx[word] = len(word2idx)

    idx2word = {idx: word for word, idx in word2idx.items()}

    print(f"vocab_size: {len(word2idx)}\ntraining_size: {len(training_sents)}")

    training_tensors = prepare_data(training_sents, word2idx)

    return training_tensors, word2idx, idx2word


def train(model, optimizer, criterion, n_epochs, training_tensors, print_every):
    """
    Train a PredictNextModel.

    :param model:
    :param optimizer:
    :param criterion:
    :param n_epochs:
    :param training_tensors:
    :param print_every:
    """

    # accumulate losses and print average every print_every epochs
    loss_sum = 0

    # epochs
    for epoch in range(1, n_epochs + 1):

        # train one sentence
        for train_tensor, target_tensor in training_tensors:

            # reset gradients
            optimizer.zero_grad()

            # initialize hidden tensor
            hidden = model.init_hidden()

            # iterate training words
            outputs = torch.tensor(())
            for i in range(train_tensor.size(0)):
                output, hidden = model(train_tensor[i], hidden)
                outputs = torch.cat((outputs, output), dim=0)

            # compute loss
            loss = criterion(outputs, target_tensor)

            # accumulate losses until next print
            # where the average loss is printed
            loss_sum += loss.item()

            # backpropagation
            loss.backward()

            # update weights
            optimizer.step()

        if epoch % print_every == 0:
            print(f'loss: {loss_sum / (len(training_tensors) * print_every)}')
            loss_sum = 0


def instantiate_and_train(rnn_type, hidden_size, lr, n_epochs, training_data, word2idx, print_every):
    """
    Train a PredictNextModel implemented with rnn_type (one of USE_RNN, USE_GRU, USE_LSTM).

    :param rnn_type: one of USE_RNN, USE_GRU, USE_LSTM
    :param hidden_size: hidden size for the PredictNextModel
    :param lr: learning rate of the optimizer
    :param n_epochs: number of training epochs
    :param training_data: list[list[str]] list of tokenized sentences
    :param word2idx: word to vocab index map
    :param print_every: print loss every print_every epochs
    """

    # instantiate model
    vocab_size = len(word2idx)
    predict_next_model = PredictNextModel(vocab_size, hidden_size, rnn_type)

    # instantiate optimizer using model params and lr
    optimizer = optim.AdamW(predict_next_model.parameters(), lr=lr)

    # NLL loss fn
    criterion = nn.NLLLoss()

    # train
    train(predict_next_model, optimizer, criterion, n_epochs, training_data, print_every)

    return predict_next_model


def generate_text(model, idx2word, number_to_generate, choose_random=False):
    """
    Use model to generate text.
    Start with the BOS token, then use the model's prediction as the next input.
    Continue until EOS is generated or MAX_LENGTH is reached.

    If choose_random == True:
        Instead of using the model's highest-probability token as input at the next
        time step, choose randomly from the model's top 3 predictions.

    Hint: use the tensor function topk() to get the highest predictions.

    :param model: model to use for generation
    :param idx2word: vocab map
    :param number_to_generate: number of sentences to generate
    :param choose_random: use top prediction if False, otherwise choose from top 3
    """

    # generate sentences with the trained model
    with torch.no_grad():
        pass


if __name__ == '__main__':

    # generate train and target tensors from the training sentences
    training_tensors, word2idx, idx2word = get_training_data("dr_seuss_geah.txt")

    # hyperparameters
    hidden_size = 16
    learning_rate = .005
    n_epochs = 300
    print_every = 50

    #torch.manual_seed(42)
    num_sentences = 15
    print("\nTrain and generate with RNN:")
    rnn_model = instantiate_and_train(USE_RNN, hidden_size, learning_rate, n_epochs, training_tensors, word2idx, print_every)
    generate_text(rnn_model, idx2word, num_sentences, choose_random=False)

    print("\nTrain and generate with GRU:")
    gru_model = instantiate_and_train(USE_GRU, hidden_size, learning_rate, n_epochs, training_tensors, word2idx, print_every)
    generate_text(gru_model, idx2word, num_sentences, choose_random=False)

    print("\nTrain and generate with LSTM:")
    lstm_model = instantiate_and_train(USE_LSTM, hidden_size, learning_rate, n_epochs, training_tensors, word2idx, print_every)
    generate_text(lstm_model, idx2word, num_sentences, choose_random=False)