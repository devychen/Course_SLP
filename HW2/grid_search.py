"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    (Enter the assignment number - e.g. A1)
Author(s):     (Enter the full names of author(s) here)

Honor Code:    I/We pledge that this program represents my/our own work,
               and that I/we have not given or received unauthorized help
               with this assignment.
"""
from trainer import Trainer
from constants import *


def main():
    """
    Find the best combination of hyperparameters using a simple grid search.
    A few values for each hyperparameter should be enough.

    You do NOT need to save all the models trained during the grid search.
    Rather, keep track of the hyperparameters used for the current best model,
    similar to what you did in the training loop.

    Note that the macro F1 score was calculated during training, and stored
    as part of the dictionary returned by Trainer.train().
    You can get the model score from this dictionary, rather than
    recalculating the score.

    When the grid search is finished, train a model using the best hyperparameters,
    and save that one to Models/best-model (generating files Models/best-model.pt and
    Models/best-model-info.json).

    You will also want to print the scores and hyperparameters, to use in your discussion.
    """

    # simple grid search
    epochs = []
    learning_rates = []
    hidden_sizes = []

    for h_size in hidden_sizes:
        for n_epochs in epochs:
            for lr in learning_rates:
                # train using current hyperparameters
                # record the hyperparameters and the model's f1-score
                # keep track of best hyperparameters
                pass

    # Re-train using the best hyperparameters and save to Models/best-model.
    pass


if __name__ == '__main__':
    main()
