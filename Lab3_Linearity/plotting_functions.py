import matplotlib.pyplot as plt


def plot(X, y, s=15, plot_title=None, plot_path=None):
    plt.scatter(X[:, 0], X[:, 1], c=y, s=15, cmap='RdYlBu')
    plt.title(plot_title)
    if plot_path is not None: plt.savefig(plot_path)
    plt.show()
    plt.close()


# incorrect dots are bigger
def fancy_plot(X, y_true, y_pred, plot_title=None, plot_path=None):
    plt.scatter(X[:, 0], X[:, 1], c=y_pred, s=14*(y_pred != y_true).astype(int) + 1, cmap='RdYlBu')
    plt.title(plot_title)
    if plot_path is not None: plt.savefig(plot_path)
    plt.show()
    plt.close()