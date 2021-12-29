import torch
from torch import nn


def vgg_block(num_convs, in_channels, out_channels):
    layers = []
    for _ in range(num_convs):
        layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
        layers.append(nn.ReLU())
        in_channels = out_channels
    layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
    return nn.Sequential(*layers)


# Using the function above, we define out architectures simply using
# a list of tuples, containing the number of conv layers and number of output channels.
conv_architecture = [(1, 64), (1, 128), (2, 256), (2, 512), (2, 512)]


# Let's write a function that takes in a list of tuples as such and returns
# the full VGG network that amalgamates the blocks.
def vgg(architecture):
    conv_components = []
    in_channels = 1
    for num_layers, out_channels in architecture:
        conv_components.append(vgg_block(num_layers, in_channels, out_channels))
        in_channels = out_channels
    vgg_network = nn.Sequential(
        *conv_components, nn.Flatten(),
        nn.Linear(out_channels * 7 * 7, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 4096), nn.ReLU(), nn.Dropout(p=0.5),
        nn.Linear(4096, 10)
    )
    return vgg_network


def test_output_shapes():
    net = vgg(conv_architecture)
    X = torch.randn((1, 1, 224, 224))
    for layer in net:
        X = layer(X)
        print(layer.__class__.__name__, 'output_shape:\t', X.shape)


def main():
    test_output_shapes()


if __name__ == "__main__":
    main()

