import numpy as np
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, mean_squared_error
import matplotlib.pyplot as plt

from plotting_functions import plot, fancy_plot


def generate_data(n_samples=2000):
    X, y = make_blobs(
        n_samples=n_samples,
        centers=4,
        n_features=2,
        random_state=0
    )
    y = np.mod(y, 2).reshape(-1, 1)
    return X, y


def cost_CE(y, y_pred):
    """
    Return the mean cross entropy loss.

    Cost of 1 sample = (-y*log(y_pred) - (1-y)*log(1-y_pred))
    Mean Cost = sum((-y*log(y_pred) - (1-y)*log(1-y_pred)) / len(y)
    """

    # Predictions of 0 or 1 can result in a call to log(0), which is undefined.
    # change 0's to small probability > 0, and 1's to large probability < 1
    epsilon = np.finfo(np.float64).eps
    y_clipped = np.clip(y_pred, epsilon, (1 - epsilon))

    # Calculate the cost of each sample
    cost_all = -y * np.log(y_clipped) - (1 - y) * np.log((1 - y_clipped))

    # Calculate the average cost
    cost_mean = cost_all.sum() / len(y_clipped)

    return cost_mean


class LogisticRegression:

    def __init__(self, input_size):
        self.input_size = input_size

        self.weights = None
        self.bias = None
        self.dw = None
        self.db = None

        self.init_weights()

    # initialize weights
    def init_weights(self):
        self.weights = np.zeros((self.input_size, 1))
        self.bias = 0

    # forward pass
    def forward(self, X):
        linear_model = np.dot(X, self.weights) + self.bias
        y_pred = self.sigmoid(linear_model)
        return y_pred
    
    # zero previous gradients
    def zero_grad(self):
        self.dw = np.zeros(self.weights.shape)
        self.db = 0

    # compute gradients
    def grad(self, X, y, y_pred):
        n_samples = X.shape[0]
        self.dw = (1 / n_samples) * np.dot(X.T, (y_pred - y))
        self.db = (1 / n_samples) * np.sum(y_pred - y)

    # training loop
    def train(self, X_train, y_train, lr=0.1, n_epochs=1000, plot_path=None):
        
        n_samples, _ = X_train.shape

        if plot_path is not None: loss_items = []

        # gradient descent
        for _ in range(n_epochs):

            # zero previous gradients
            self.zero_grad()

            # get predictions
            y_pred = self.forward(X_train)

            # compute gradients
            self.grad(X_train, y_train, y_pred)

            # update parameters
            self.weights -= lr * self.dw
            self.bias -= lr * self.db

            # record loss
            if plot_path is not None:
                loss = cost_CE(y_train, y_pred)
                loss_items.append(loss)

        # plot
        if plot_path is not None:
            xs = [x for x in range(n_epochs)]
            plt.scatter(xs, loss_items)
            plt.title('Logistic Regression Loss')
            plt.xlabel('Epochs')
            plt.ylabel('CE Loss')
            plt.savefig(plot_path)
            plt.close()

    def predict(self, X_test):
        y_pred = self.forward(X_test)
        pred_labels = y_pred.round().astype(int)
        return pred_labels

    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))
    

class FeedForwardNetwork:
  
    def __init__(self, input_size, hidden_size=4):
        self.input_size = input_size
        self.hidden_size = hidden_size

        self.weights_in = None
        self.bias_in = None
        self.weights_out = None
        self.bias_out = None

        self.dw_in = None
        self.db_in = None
        self.dw_out = None
        self.db_out = None

        self.init_weights()

    # initialize weights
    def init_weights(self):
        """
        Initialize weights and biases of the model. You must have two matrices of weights:
        from input layer to hidden layer and from hidden layer to output layer.
        The model must have one bias for each weight matrix.
        """
        # ToDo
        self.weights_in = np.random.rand(self.input_size, self.hidden_size)
        self.weights_out = np.random.rand(self.hidden_size, 1)
        self.bias_in = np.random.rand(self.hidden_size)
        self.bias_out = np.random.rand(1)
  
    # forward pass
    def forward(self, X):
        """
         Implement forward pass following the instructions in the comments.
         Be sure to return both outputs of hidden and output layers, they are needed
         when computing gradients.

         :param X: training data
         :return: hidden_output: output of hidden layer
         :return: output: output of output layer
         """
        # ToDo: compute output of hidden layer
        hidden_output = np.dot(X, self.weights_in) + self.bias_in
        # ToDo: apply sigmoid
        hidden_output = self.sigmoid(hidden_output)

        # ToDo: compute output of output layer
        output = np.dot(hidden_output, self.weights_out) + self.bias_out
        # ToDo: apply sigmoid
        output = self.sigmoid(output)
        return hidden_output, output
    
    # zero previous gradients
    def zero_grad(self):
        self.dw_in = np.zeros(self.weights_in.shape)
        self.db_in = np.zeros(self.hidden_size)
        self.dw_out = np.zeros(self.weights_out.shape)
        self.db_out = np.zeros(1)
  
    # compute gradients
    def grad(self, X, y, hidden_output, y_pred):

        n_samples = X.shape[0]

        dz2 = y_pred - y
        self.dw_out = (1 / n_samples) * np.dot(hidden_output.T, dz2)
        self.db_out = (1 / n_samples) * np.sum(dz2, axis=0)

        dz1 = np.dot(dz2, self.weights_out.T) * hidden_output * (1 - hidden_output)
        self.dw_in = (1 / n_samples) * np.dot(X.T, dz1)
        self.db_in = (1 / n_samples) * np.sum(dz1, axis=0)
        
    # training loop
    def train(self, X_train, y_train, lr=0.5, n_epochs=1000, plot_path=None):
        """
        Implement training loop. Refer to the training loop of LogisticRegression class
        and to the comments below.
        :param X_train: training data
        :param y_train: true labels
        :param lr: learning rate
        :param n_epochs: number of training epochs
        :param plot_path: if not None, will save a loss plot to this path
        :return: None
        """

        n_samples, _ = X_train.shape

        if plot_path is not None: loss_items = []

        # gradient descent
        for _ in range(n_epochs):

            # ToDo: zero previous gradients
            self.zero_grad()

            # ToDo: forward pass
            hidden_output, y_pred = self.forward(X_train)

            # ToDo: compute gradients
            self.grad(X_train, y_train, hidden_output, y_pred)

            # ToDo: update parameters
            self.weights_in -= lr * self.dw_in
            self.bias_in -= lr * self.db_in
            self.weights_out -= lr * self.dw_out
            self.bias_out -= lr * self.db_out

            # record loss
            if plot_path is not None:
                cost = cost_CE(y_train, y_pred)
                loss_items.append(cost)

        # plot
        if plot_path is not None:
            plt.plot(loss_items)
            plt.title('FFN Loss')
            plt.xlabel('Epochs')
            plt.ylabel('CE Loss')
            plt.savefig(plot_path)
            plt.close()

    def predict(self, X_test):
        """
        Implement prediction function.

        :param X_test: test data
        :return: y_pred: array of predicted labels: 0 or 1
        """
        # ToDo
        _, y_pred = self.forward(X_test)
        pred_labels = y_pred.round().astype(int)
        return pred_labels
    
    def sigmoid(self, X):
        return 1 / (1 + np.exp(-X))


def main():

    np.random.seed(27)
    # generate and plot data
    X, y = generate_data()

    # split and plot data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.33)
    _, n_features = X_train.shape

    # data plots
    plot(X, y, plot_title="All Data", plot_path='./plots/data/data_all.png')
    plot(X_train, y_train, plot_title="Train Data", plot_path='./plots/data/data_train.png')
    plot(X_test, y_test, plot_title="Test Data", plot_path='./plots/data/data_test.png')

    # predict with logistic regression and visualize predictions
    np.random.seed(42)
    logistic_reg = LogisticRegression(input_size=n_features)
    logistic_reg.train(X_train, y_train, lr=0.05, n_epochs=2000, plot_path='./plots/loss/logistic.png')
    y_pred_logistic = logistic_reg.predict(X_test)

    # incorrect dots are bigger
    fancy_plot(X_test, y_test, y_pred_logistic, plot_title="Logistic Regression Predictions\nLarge Dot = Wrong", plot_path='./plots/data/test_pred_logistic.png')
    f1_logistic = f1_score(y_test, y_pred_logistic)
    print(f'F1 with logistic regression: {round(f1_logistic, 3)}')

    # ToDo: predict with FFN and visualize predictions
    np.random.seed(42)
    ffn = FeedForwardNetwork(input_size=n_features)
    ffn.train(X_train, y_train, lr=0.5, n_epochs=2000, plot_path='./plots/loss/ffn.png')
    y_pred_ffn = ffn.predict(X_test)

    # incorrect dots are bigger
    fancy_plot(X_test, y_test, y_pred_ffn, plot_title="FFN Predictions\nLarge Dot = Wrong", plot_path='./plots/data/test_pred_ffn.png')
    f1_ffn = f1_score(y_test, y_pred_ffn)
    print(f'F1 with FFN: {round(f1_ffn, 3)}')


if __name__ == '__main__':
    main()