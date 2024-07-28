#
# PyTorch Basics
#
# Resources:
# - PyTorch documentation: (https://pytorch.org/docs/stable/index.html)
# - Patrick Loeber PyTorch Tutorial (lessons 2,3,4) (https://www.python-engineer.com/courses/pytorchbeginner/)
#
import numpy as np
import torch


def check_result():
    """
    Verify the paper-and-pencil result by using one forward and one backward pass
    """

    # track gradient of w, but not x or y
    x = torch.tensor(4.0)
    y = torch.tensor(1.0)
    w = torch.tensor(0.5, requires_grad=True)

    # forward pass
    y_hat = w * x

    # compute loss
    loss = (y_hat - y)**2
    print(f"loss = {loss}")  # should match your hand-computed loss

    # backward pass
    loss.backward()
    print(f"w.grad = {w.grad}")  # should match your hand-computed gradient


def train():
    """
    Extend the code in check_result() to implement a training loop.
    Update the weights at each iteration.

    Gradient calculation on the weight tensor must be disabled while
    updating the weight

    Gradient in the weight tensor must be reset to zero at each iteration.

    Print the loss, the updated weight, and the predicted value at each iteration.

    After 10 iterations, with  x=4, y=1, w=0.5 initially,
    and with a learning rate of .001, the final loss should be: ~ 0.5569
    """
    pass


def check_result2():
    """
    Verify your paper-and-pencil result of the FNN network
    by using one forward and one backward pass
    """
    pass


if __name__ == '__main__':
    check_result()
    #train()
    #check_result2()
