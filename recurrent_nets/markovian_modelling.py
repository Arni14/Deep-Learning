import torch
from torch import nn
from d2l import torch as d2l

# Using d2l code, produce the dummy dataset.
T = 1000  # Generate a total of 1000 points
time = torch.arange(1, T + 1, dtype=torch.float32)
x = torch.sin(0.01 * time) + torch.normal(0, 0.2, (T,))

# The task is to use this dataset to produce training data. Our construction is to use
# pairs ([x_{t - 1}, ... x_{t - tau}], x_t). However, this won't work for the first tau
# elements of the dataset. SO, we start at x_{tau + 1}
tau = 4
features = torch.zeros((T - tau, tau)) # T - tau examples, all of length tau
labels = x[tau:].reshape((-1, 1))
for i in range(tau):
    features[:, i] = x[i : T - tau + i]

batch_size, n_train = 16, 600
train_iter = d2l.load_array((features[:n_train], labels[:n_train]),
                            batch_size, is_train=True)

# Now, we'll set up a network that learns the mapping from features to labels. This is basically
# one-step mapping, since we use our MLP to map tau previous steps to one step ahead.
hidden_size = 100
net = nn.Sequential(
    nn.Linear(4, hidden_size),
    nn.ReLU(),
    nn.Linear(hidden_size, 1)
)


def init_weights(m):
    if isinstance(m, nn.Linear):
        nn.init.xavier_uniform_(m.weight)


net.apply(init_weights)


def train_one_step(num_epochs, lr, hidden_size):
    curr_net = nn.Sequential(
        nn.Linear(4, hidden_size),
        nn.ReLU(),
        nn.Linear(hidden_size, 1)
    )
    curr_net.apply(init_weights)
    curr_loss = nn.MSELoss(reduction='none')
    curr_optimizer = torch.optim.Adam(net.parameters(), lr=lr)
    for epoch in range(num_epochs):
        for X, y in train_iter:
            curr_optimizer.zero_grad()
            # Run the prediction for X
            y_pred = curr_net(X)
            l = curr_loss(y_pred, y)
            l.sum().backward()
            curr_optimizer.step()
        print(f"Epoch {epoch + 1}, Loss {d2l.evaluate_loss(curr_net, train_iter, curr_loss)}")
    return curr_net


if __name__ == "__main__":
    # Now, we're going to run a training loop on our training set to learn this mapping.
    for i in range(1, 10):
        for lr in [0.001, 0.01, 0.1, 0.3]:
            for h_s in [5, 10, 15, 20, 35]:
                print(f"{i} epochs, {lr} learning rate, {h_s} hidden neurons")
                train_one_step(i, lr, h_s)
                print("\n")
