import torch
from torch import nn
from torch.nn import functional as F
from d2l import torch as d2l

if __name__ == "__main__":
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # We use a tensor to initialize the hidden state, whose shape is (number of
    # hidden layers, batch size, number of hidden units).
    num_hidden = 256
    rnn_layer = nn.RNN(len(vocab), num_hidden)
    # For our purpose:
    state = torch.zeros((1, batch_size, num_hidden))
    print(state.shape)
    # Let's run a test input through the rnn layer
    X = torch.rand(size=(num_steps, batch_size, len(vocab)))
    y, state_new = rnn_layer(X, state)
    print(y.shape, state_new.shape)
