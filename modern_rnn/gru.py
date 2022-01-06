from torch import nn
import torch
from d2l import torch as d2l


def init_gru_state(batch_size, num_hidden, device):
    return torch.zeros((batch_size, num_hidden), device=device)


# Just wrote my own GRU implementation.
class MyGRU(nn.Module):
    def __init__(self, vocab_size, num_hidden, device):
        super().__init__()
        num_inputs = num_outputs = vocab_size

        def three():
            return nn.Parameter(torch.randn((num_inputs, num_hidden)) * 0.01), \
                   nn.Parameter(torch.randn((num_hidden, num_hidden)) * 0.01), \
                   nn.Parameter(torch.zeros(num_hidden, device=device))

        self.W_xz, self.W_hz, self.b_z = three() # Update gate parameters
        self.W_xr, self.W_hr, self.b_r = three() # Reset gate parameters
        self.W_xh, self.W_hh, self.b_h = three() # Hidden gate parameters
        self.W_hq, self.b_q = nn.Parameter(torch.randn((num_hidden, num_outputs))) * 0.01, \
                              nn.Parameter(torch.zeros(num_hidden, device=device)) # Output gate parameters

    def forward(self, inputs, initial_state):
        h = initial_state
        outputs = []
        for X in inputs:
            z = torch.sigmoid(X @ self.W_xz + h @ self.W_hz + self.b_z)
            r = torch.sigmoid(X @ self.W_xr + h @ self.W_hr + self.b_r)
            h_candidate = torch.tanh(X @ self.W_xh + (r * h) @ self.W_hh + self.b_h)
            h = z * h + (1 - z) * h_candidate
            outputs.append(X @ self.W_hq + self.b_q)
        return torch.cat(outputs, dim=0), h,


# Using the d2l gru implementation and running it locally.
def get_params(vocab_size, num_hiddens, device):
    num_inputs = num_outputs = vocab_size

    def normal(shape):
        return torch.randn(size=shape, device=device)*0.01

    def three():
        return (normal((num_inputs, num_hiddens)),
                normal((num_hiddens, num_hiddens)),
                torch.zeros(num_hiddens, device=device))

    W_xz, W_hz, b_z = three()  # Update gate parameters
    W_xr, W_hr, b_r = three()  # Reset gate parameters
    W_xh, W_hh, b_h = three()  # Candidate hidden state parameters
    # Output layer parameters
    W_hq = normal((num_hiddens, num_outputs))
    b_q = torch.zeros(num_outputs, device=device)
    # Attach gradients
    params = [W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q]
    for param in params:
        param.requires_grad_(True)
    return params


def gru(inputs, state, params):
    W_xz, W_hz, b_z, W_xr, W_hr, b_r, W_xh, W_hh, b_h, W_hq, b_q = params
    print(state)
    H, = state
    outputs = []
    for X in inputs:
        Z = torch.sigmoid((X @ W_xz) + (H @ W_hz) + b_z)
        R = torch.sigmoid((X @ W_xr) + (H @ W_hr) + b_r)
        H_tilda = torch.tanh((X @ W_xh) + ((R * H) @ W_hh) + b_h)
        H = Z * H + (1 - Z) * H_tilda
        Y = H @ W_hq + b_q
        outputs.append(Y)
    return torch.cat(outputs, dim=0), (H,)


def testing_my_gru(vocab, num_hidden, device):
    curr_gru = MyGRU(len(vocab), num_hidden, device)
    print(list(curr_gru.parameters()))


def testing_d2l_gru(vocab, num_hidden, device, num_epochs, lr, train_iter):
    num_inputs = len(vocab)
    gru_layer = nn.GRU(num_inputs, num_hidden)
    model = d2l.RNNModel(gru_layer, len(vocab))
    model = model.to(device)
    d2l.train_ch8(model, train_iter, vocab, lr, num_epochs, device)


def main():
    batch_size, num_steps = 32, 35
    train_iter, vocab = d2l.load_data_time_machine(batch_size, num_steps)
    # testing_my_gru(vocab, 256, d2l.try_gpu())
    testing_d2l_gru(vocab, 256, d2l.try_gpu(), 500, 1, train_iter)


if __name__ == "__main__":
    main()
