from torch import nn
import torch
from torch.nn import functional as F
from d2l import torch as d2l


class MyRNN(nn.Module):
    def __init__(self, token_dim, hidden_dim, num_outputs):
        super().__init__()
        # Initialize trainable params
        self.W_xh = nn.Parameter(torch.randn((token_dim, hidden_dim), dtype=torch.float32) * 0.01, requires_grad=True)
        self.W_hh = nn.Parameter(torch.randn((hidden_dim, hidden_dim), dtype=torch.float32) * 0.01, requires_grad=True)
        self.b_h = nn.Parameter(torch.zeros(hidden_dim, dtype=torch.float32), requires_grad=True)
        self.b_q = nn.Parameter(torch.zeros(num_outputs, dtype=torch.float32), requires_grad=True)
        self.W_hq = nn.Parameter(torch.randn((hidden_dim, num_outputs), dtype=torch.float32) * 0.01, requires_grad=True)

        # Initialize important dimensions
        self.d = token_dim
        self.h = hidden_dim
        self.q = num_outputs

    def forward(self, x):
        """
        x is a tensor of shape (input_length, batch_size, token_dim) and we store
        both the hidden state and the output at every token into the output.
        We return a (input_length x batch_size, hidden_dim) tensor that contains
        the logits for every iteration and batch.
        """
        n, input_length = x.shape[1], x.shape[0]
        h = self.get_initial_hidden(n)
        out = []
        for i in range(input_length):
            h = F.relu(x[i] @ self.W_xh + h @ self.W_hh + self.b_h)
            out.append(h @ self.W_hq + self.b_q)
        return torch.cat(out, dim=0), h

    def get_initial_hidden(self, n):
        return torch.randn((n, self.h)) * 0.01


def one_hot():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    hidden_dim = 512
    iterator = iter(train_iter)
    X, Y = next(iterator)
    print(X)
    print(X.shape)
    one_hot = F.one_hot(X, len(vocab))
    print(one_hot)
    print(one_hot.shape)
    one_hot_transpose = F.one_hot(X.T, len(vocab))
    print(one_hot_transpose)
    print(one_hot_transpose.shape)


def run_rnn():
    # Set up the required values and data
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    hidden_dim = 512
    iterator = iter(train_iter)
    x, y = next(iterator)
    x = F.one_hot(x.T, len(vocab))
    x = x.to(torch.float32)

    # Run the network
    net = MyRNN(len(vocab), hidden_dim, len(vocab))
    y_final, new_state = net(x)
    print(x.shape)
    print(y_final.shape)


def main():
    run_rnn()


if __name__ == "__main__":
    main()
