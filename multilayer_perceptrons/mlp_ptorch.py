import torch
from torch import nn
from d2l import torch as d2l
import matplotlib.pyplot as plt

my_mlp = nn.Sequential(
    nn.Flatten(),
    nn.Linear(784, 256),
    nn.ReLU(),
    nn.Linear(256, 10)
)

def init_weights(m, bias_val):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)
        nn.init.constant_(m.bias.data, bias_val)

loss = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(my_mlp.parameters(), lr=0.01)
train_iter, test_iter = d2l.load_data_fashion_mnist(256)
num_epochs = 10

for bias in torch.linspace(0, 0.5, 10):
    my_mlp.apply(lambda m : init_weights(m, bias))
    d2l.train_ch3(my_mlp, train_iter, test_iter, loss, num_epochs, optimizer)
    plt.show()