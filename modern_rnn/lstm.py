import torch
from torch import nn
from d2l import torch as d2l


def init_state(batch_size, num_hidden, device):
    return torch.zeros((batch_size, num_hidden), device=device), \
           torch.zeros((batch_size, num_hidden), device=device)


def get_params(vocab_size, num_hidden, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device) * 0.01

    def three():
        return (normal((num_inputs, num_hidden)),
                normal((num_hidden, num_hidden)),
                torch.zeros(num_hidden, device=device))
    # Parameter weights for Input, Forget, Output, and Memory
    W_xi, W_hi, b_i = three()
    W_xf, W_hf, b_f = three()
    W_xo, W_ho, b_o = three()
    W_xc, W_hc, b_c = three()
    # Weights for the output layer
    W_hq = normal((num_hidden, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc,
              W_hc, b_c, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def lstm(inputs, state, params):
    W_xi, W_hi, b_i, W_xf, W_hf, b_f, W_xo, W_ho, b_o, W_xc, W_hc, b_c, W_hq, b_q = params
    h, c = state
    outputs = []
    for X in inputs:
        i = torch.sigmoid(X @ W_xi + h @ W_hi + b_i)
        f = torch.sigmoid(X @ W_xf + h @ W_hf + b_f)
        o = torch.sigmoid(X @ W_xo + h @ W_ho + b_o)
        c_candidate = torch.tanh(X @ W_xc + h @ W_hc + b_c)
        c = f * c + i * c_candidate
        h = o * torch.tanh(c)
        outputs.append(h @ W_hq + b_q)
    return torch.cat(outputs, dim=0), (h, c)


def testing_d2l_lstm(vocab, num_hidden, device, num_epochs, lr, train_iter):
    model = d2l.RNNModelScratch(len(vocab), num_hidden, device, get_params,
                                init_state, lstm)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


def testing_pytorch_lstm(vocab, num_hidden, device, num_epochs, lr, train_iter):
    model = d2l.RNNModel(nn.LSTM(len(vocab), num_hidden), len(vocab))
    model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    testing_d2l_lstm(vocab, 256, d2l.try_gpu(), 250, 1, train_iter)
    testing_pytorch_lstm(vocab, 256, d2l.try_gpu(), 500, 1, train_iter)


if __name__ == "__main__":
    main()