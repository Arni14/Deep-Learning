import torch
from torch import nn
import numpy as np
from d2l import torch as d2l


def corr2d(X, K):
    """ Compute the 2D cross-correlation """
    Y = torch.zeros((X.shape[0] - K.shape[0] + 1, X.shape[1] - K.shape[1] + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            current_window = X[i: i + K.shape[0], j: j + K.shape[1]]
            Y[i, j] = (current_window * K).sum()
    return Y


# Implementing the 2D convolution layer by hand!
class Conv2D(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        # Tensor of 2 dimensions
        self.weight = nn.Parameter(torch.rand(kernel_size))
        # Tensor of one element, broadcast across all the outputs.
        self.bias = nn.Parameter(torch.zeros(1))

    def forward(self, X):
        X = corr2d(X, self.weight)
        return X + self.bias


# -------------------------------------------------- #
# All testing utilities are below the line shown above #
def test_corr_2d():
    X = torch.randint(0, 3, (6, 6), dtype=torch.float32)
    K = torch.randint(0, 3, (3, 3), dtype=torch.float32)
    convolution = corr2d(X, K).numpy()
    print("X:")
    print(str(X.numpy()) + "\n")

    print("K:")
    print(str(K.numpy()) + "\n")

    print("Convolution:")
    print(str(convolution) + "\n")


def convolutions_edge_detection():
    X = torch.ones((6, 8))
    X[:, 2: 6] = 0
    print("X:")
    print(str(X) + "\n")

    # Set up a kernel that we can convolve with
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)
    print("Convolution / edges:")
    print(str(Y) + "\n")

    # Do the same convolution but with the transpose of X to get a vertical edge
    # detector.
    Y_t = corr2d(X.t(), K)
    print("Vertical edge detection:")
    print(str(Y_t))


def learn_same_kernel():
    # Set up the network layer
    conv2d = nn.Conv2d(1, 1, kernel_size=(1, 2), bias=False)

    # Set up the desired input-output pairs
    X = torch.ones((6, 8))
    X[:, 2: 6] = 0
    K = torch.tensor([[1.0, -1.0]])
    Y = corr2d(X, K)

    # Reshape the images
    X = X.reshape((1, 1, 6, 8))
    Y = Y.reshape((1, 1, 6, 7))
    lr = 3e-2

    # Train the conv2d to learn the weights to see if it learns the right
    # kernel.
    num_iter = 16
    for i in range(num_iter):
        Y_hat = conv2d(X)
        loss = (Y_hat - Y) ** 2
        conv2d.zero_grad()
        loss.sum().backward()
        # Update the kernel
        conv2d.weight.data[:] -= lr * conv2d.weight.grad
        if not (i + 1) % 2:
            print(f'epoch {i + 1}, loss {loss.sum():.3f}')
    print()
    print(conv2d.weight.data.detach().numpy().reshape((1, 2)))


def main():
    # test_corr_2d()
    # convolutions_edge_detection()
    learn_same_kernel()


if __name__ == "__main__":
    main()
