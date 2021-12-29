import torch
from torch import nn


def pool2d(X, pool_size, mode='max'):
    p_h, p_w = pool_size
    Y = torch.zeros((X.shape[0] - p_h + 1, X.shape[1] - p_w + 1))
    for i in range(Y.shape[0]):
        for j in range(Y.shape[1]):
            X_curr = X[i : i + p_h, j : j + p_w]
            if mode == 'max':
                Y[i, j] = X_curr.max()
            else:
                Y[i, j] = X_curr.mean()
    return Y


def init_tensors():
    return torch.tensor([[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]])


def custom_pool():
    X = init_tensors()
    print("Output of custom max pooling:")
    print(pool2d(X, (2, 2)))
    print("\n")
    print("Output of custom avg pooling:")
    print(pool2d(X, (2, 2), 'avg'))
    print("\n")


def use_pytorch_pooling():
    # Shape is (1, 1, 4, 4)
    X = torch.arange(16, dtype=torch.float32).reshape((1, 1, 4, 4))

    # First pool, 3x3 window, no stride and padding
    first_pool = nn.MaxPool2d(3)
    print("Using 3x3 Max Pool, No Stride/Padding:")
    print(first_pool(X))
    print("\n")

    # Now with same window, but padding 1 and stride 2
    second_pool = nn.MaxPool2d(3, padding=1, stride=2)
    print("Using 3x3 Max Pool, Stride=2, Padding=1:")
    print(second_pool(X))
    print("\n")


def main():
    custom_pool()
    use_pytorch_pooling()


if __name__ == "__main__":
    main()