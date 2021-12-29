# Recall that the inputs and outputs of convolutional layers consist of four-dimensional
# tensors with axes corresponding to the example, channel, height, and width.
# Also recall that the inputs and outputs of fully-connected layers are typically
# two-dimensional tensors corresponding to the example and feature. The idea
# behind NiN is to apply a fully-connected layer at each pixel location (for
# each height and width).
import torch
from torch import nn


# Let's define the NiN block
def nin_block(in_channels, out_channels, kernel_size, strides, padding):
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size,
                  stride=strides, padding=padding), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU(),
        nn.Conv2d(out_channels, out_channels, kernel_size=1), nn.ReLU()
    )


nin_model = nn.Sequential(
    nin_block(1, 96, kernel_size=11, strides=4, padding=0),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(96, 256, kernel_size=5, strides=1, padding=2),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nin_block(256, 384, kernel_size=3, strides=1, padding=1),
    nn.MaxPool2d(kernel_size=3, stride=2),
    nn.Dropout(p=0.5),
    nin_block(384, 10, kernel_size=3, strides=1, padding=1),
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten()
)


def test_shapes():
    X = torch.randn((1, 1, 224, 224))
    for layer in nin_model:
        X = layer(X)
        print(layer.__class__.__name__,'output shape:\t', X.shape)


def main():
    test_shapes()


if __name__ == "__main__":
    main()
