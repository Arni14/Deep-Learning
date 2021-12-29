import torch
from torch import nn
from d2l import torch as d2l


def corr2d_multi_in(X, K):
    return sum(d2l.corr2d(x, k) for x, k in zip(X, K))


def corr2d_multi_in_out(X, K):
    return torch.stack([corr2d_multi_in(X, k) for k in K], 0)


# All testing utilities
# -------------------------------------------------------------------------
def init_tensors():
    X = torch.tensor([[[0.0, 1.0, 2.0], [3.0, 4.0, 5.0], [6.0, 7.0, 8.0]],
                      [[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]])
    K = torch.tensor([[[0.0, 1.0], [2.0, 3.0]], [[1.0, 2.0], [3.0, 4.0]]])
    return X, K


def multi_in_test():
    X, K = init_tensors()
    print(corr2d_multi_in(X, K))


def multi_out_and_in_test():
    X, K = init_tensors()
    K = torch.stack((K, K + 1, K + 2), 0)
    print(str(K.shape) + "\n") # Ensure we got the stacking right, should be (3, K.shape)
    print(corr2d_multi_in_out(X, K))


def main():
    multi_in_test()
    multi_out_and_in_test()


if __name__ == "__main__":
    main()