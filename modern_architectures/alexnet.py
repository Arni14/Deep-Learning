import torch
from torch import nn
from torch import optim
from d2l import torch as d2l

# Please note that a lot of the code below is exactly from the d2l book with my own
# changes -- I've used my own implementation and then tried to emulate parts from
# d2l. 

alex_net = nn.Sequential(
    nn.Conv2d(1, 96, kernel_size=(11, 11), stride=4, padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    nn.Conv2d(96, 256, kernel_size=(5, 5), padding=2), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    nn.Conv2d(256, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(384, 384, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.Conv2d(384, 256, kernel_size=(3, 3), padding=1), nn.ReLU(),
    nn.MaxPool2d(kernel_size=(3, 3), stride=2),
    nn.Flatten(),
    nn.Linear(6400, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 4096), nn.ReLU(),
    nn.Dropout(p=0.5),
    nn.Linear(4096, 10)
)


def evaluate_accuracy_gpu(network, data_iter, device=None):
    if isinstance(network, nn.Module):
        # Set the network to eval mode
        network.eval()
        if not device:
            # Move the network to the right device
            device = next(iter(network.parameters())).device
        metric = d2l.Accumulator(2)

    with torch.no_grad():
        for X, y in data_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            metric.add(d2l.accuracy(network(X), y), d2l.size(y))
    return metric[0] / metric[1]


def training_loop(network, train_iter, test_iter, num_epochs, device,
                  optimizer, loss_fn):
    # Initialize the weights based on the xavier initialization
    def init_weights(m):
        if isinstance(m, nn.Linear) or isinstance(m, nn.Conv2d):
            nn.init.xavier_uniform_(m.weight)
    network.apply(init_weights)
    print('training on', device)

    # Move the network (all the parameters to the device)
    network.to(device)
    animator = d2l.Animator(xlabel='epoch', xlim=[1, num_epochs],
                            legend=['train loss', 'train acc', 'test acc'])
    timer, num_batches = d2l.Timer(), len(train_iter)
    for epoch in range(num_epochs):
        metric = d2l.Accumulator(3)
        # Set the network to training mode -- remember, this actually activates
        # train-mode-only layers like Dropout or Batchnorm.
        network.train()
        for i, (X, y) in enumerate(train_iter):
            timer.start()
            optimizer.zero_grad()
            # Move the tensors to the GPU
            X, y = X.to(device), y.to(device)
            y_pred = network(X)
            loss = loss_fn(y_pred, y)
            loss.backward()
            optimizer.step()
            with torch.no_grad():
                metric.add(loss * X.shape[0], d2l.accuracy(y_pred, y), X.shape[0])
            timer.stop()
            train_l = metric[0] / metric[2]
            train_acc = metric[1] / metric[2]
            if (i + 1) % (num_batches // 5) == 0 or i == num_batches - 1:
                animator.add(epoch + (i + 1) / num_batches,
                             (train_l, train_acc, None))
        test_acc = evaluate_accuracy_gpu(network, test_iter)
        animator.add(epoch + 1, (None, None, test_acc))
    print(f'loss {train_l:.3f}, train acc {train_acc:.3f}, '
          f'test acc {test_acc:.3f}')
    print(f'{metric[2] * num_epochs / timer.sum():.1f} examples/sec '
          f'on {str(device)}')


def shape_evolution():
    random_tensor = torch.randn((1, 1, 224, 224))
    for layer in alex_net:
        random_tensor = layer(random_tensor)
        print(layer.__class__.__name__, 'output shape:\t', random_tensor.shape)


def mnist_alex_net():
    batch_size, lr, num_epochs = 128, 0.01, 10
    train_iter, test_iter = d2l.load_data_fashion_mnist(batch_size, resize=224)
    optimizer = optim.Adam(alex_net.parameters(), lr=lr)
    loss_fn = nn.CrossEntropyLoss()
    training_loop(alex_net, train_iter, test_iter, num_epochs, d2l.try_gpu(),
                  optimizer, loss_fn)


def main():
    mnist_alex_net()


if __name__ == "__main__":
    # pdb.set_trace()
    main()






