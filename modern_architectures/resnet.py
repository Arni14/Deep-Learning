import torch
from torch import nn
from torch.nn import functional as F


class Residual(nn.Module):
    def __init__(self, input_channels, num_channels, use_1x1conv=False, padding=1, strides=1):
        super().__init__()
        self.conv1 = nn.Conv2d(input_channels, num_channels, kernel_size=3, stride=strides, padding=1)
        self.conv2 = nn.Conv2d(num_channels, num_channels, kernel_size=3, padding=1)
        if use_1x1conv:
            self.conv3 = nn.Conv2d(input_channels, num_channels, kernel_size=1, stride=strides)
        else:
            self.conv3 = None
        self.bn1 = nn.BatchNorm2d(num_channels)
        self.bn2 = nn.BatchNorm2d(num_channels)

    def forward(self, x):
        y = F.relu(self.bn1(self.conv1(x)))
        y = self.bn2(self.conv2(y))
        if self.conv3:
            x = self.conv3(x)
        # This is the skip connection!
        y += x
        return F.relu(y)


def resnet_block(input_channels, num_channels, num_residuals, first_block=False):
    block = []
    for i in range(num_residuals):
        if not i and not first_block:
            block.append(Residual(input_channels, num_channels, use_1x1conv=True, strides=2))
        else:
            block.append(Residual(num_channels, num_channels))
    return block


b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.BatchNorm2d(64), nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)


b2 = nn.Sequential(
    nn.Sequential(*resnet_block(64, 64, 2, first_block=True)),
    nn.Sequential(*resnet_block(64, 128, 2)),
    nn.Sequential(*resnet_block(128, 256, 2)),
    nn.Sequential(*resnet_block(256, 512, 2))
)

resnet = nn.Sequential(
    b1, b2,
    nn.AdaptiveAvgPool2d((1, 1)),
    nn.Flatten(), nn.Linear(512, 10)
)


def shape_evolution():
    x = torch.randn((1, 1, 512, 512))
    for layer in resnet:
        x = layer(x)
        print(layer.__class__.__name__, 'output shape:\t', x.shape)


def main():
    shape_evolution()


if __name__ == "__main__":
    main()




