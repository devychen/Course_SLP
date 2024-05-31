"""
Course:        Statistical Language Processing - Summer 2024
Assignment:    A2
Author(s):     Yifei Chen

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
    epochs = [200, 250, 300]
    learning_rates = [0.01, 0.02, 0.03]
    hidden_sizes = [8, 12, 16]

    best_f1_score = 0

    for h_size in hidden_sizes:
        for n_epochs in epochs:
            for lr in learning_rates:
                # train using current hyperparameters
                trainer = Trainer()
                trainer.load_data('HW2/Data/train-tensors.pt', 'HW2/Data/dev_tensors.pt')
                # record the hyperparameters and the model's f1-score
                model_info = trainer.train(hidden_size=h_size, n_epochs=n_epochs, learning_rate=lr)
                f1_score = model_info[F1_MACRO_KEY]
                # keep track of best hyperparameters
                if f1_score > best_f1_score:
                    best_f1_score = f1_score
                    best_hyperparas = {
                        HIDDEN_SIZE_KEY: h_size,
                        N_EPOCHS_KEY: n_epochs,
                        LEARNING_RATE_KEY: lr
                    }
    print("Best hyperparameters:", best_hyperparas)
    print("Best F1 Score:", best_f1_score)      

    # Re-train using the best hyperparameters and save to Models/best-model.
    best_trainer = Trainer()
    best_trainer.load_data('HW2/Data/train-tensors.pt', 'HW2/Data/dev_tensors.pt')
    best_model_info = best_trainer.train(
        best_hyperparas[HIDDEN_SIZE_KEY],
        best_hyperparas[N_EPOCHS_KEY],
        best_hyperparas[LEARNING_RATE_KEY]
    )

    # save the best model
    best_trainer.save_best_model("HW2/Models/best=model")


if __name__ == '__main__':
    main()
