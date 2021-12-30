import torch
from torch import nn
from torch.nn import functional as F
# This architecture is about parallel concatenations using inception
# blocks. Let's start with implementing an inception block.


class InceptionBlock(nn.Module):
    def __init__(self, in_channels, c1, c2, c3, c4, **kwargs):
        super(InceptionBlock, self).__init__(**kwargs)
        # Path 1: Just a 1x1 Convolution
        self.conv1 = nn.Conv2d(in_channels, c1, kernel_size=1)
        # Path 2: A 1x1 Convolution and a 3x3 Convolution
        self.conv2a = nn.Conv2d(in_channels, c2[0], kernel_size=1)
        self.conv2b = nn.Conv2d(c2[0], c2[1], kernel_size=3, padding=1)
        # Path 3: A 1x1 Convolution and a 5x5 Convolution
        self.conv3a = nn.Conv2d(in_channels, c3[0], kernel_size=1)
        self.conv3b = nn.Conv2d(c3[0], c3[1], kernel_size=5, padding=2)
        # Path 4: A 3x3 MaxPool and a 1x1 Convolution
        self.maxpool4a = nn.MaxPool2d(kernel_size=3, stride=1, padding=1)
        self.conv4b = nn.Conv2d(in_channels, c4, kernel_size=1)

    def forward(self, x):
        out1 = F.relu(self.conv1(x))
        out2 = F.relu(self.conv2b(F.relu(self.conv2a(x))))
        out3 = F.relu(self.conv3b(F.relu(self.conv3a(x))))
        out4 = F.relu(self.maxpool4a(F.relu(self.conv4b(x))))
        return torch.cat((out1, out2, out3, out4), dim=1)


# Now let's put a few inception blocks together to build a full GoogLeNet
b1 = nn.Sequential(
    nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b2 = nn.Sequential(
    nn.Conv2d(64, 64, kernel_size=1),
    nn.ReLU(),
    nn.Conv2d(64, 192, kernel_size=3, padding=1),
    nn.ReLU(),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b3 = nn.Sequential(
    InceptionBlock(192, 64, (96, 128), (16, 32), 32),
    InceptionBlock(256, 128, (128, 192), (32, 96), 64),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b4 = nn.Sequential(
    InceptionBlock(480, 192, (96, 208), (16, 48), 64),
    InceptionBlock(512, 160, (112, 224), (24, 64), 64),
    InceptionBlock(512, 128, (128, 256), (24, 64), 64),
    InceptionBlock(512, 112, (144, 288), (32, 64), 64),
    InceptionBlock(528, 256, (160, 320), (32, 128), 128),
    nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
)

b5 = nn.Sequential(
    InceptionBlock(832, 256, (160, 320), (32, 128), 128),
    InceptionBlock(832, 384, (192, 384), (48, 128), 128),
    nn.AdaptiveAvgPool2d((1,1)),
    nn.Flatten()
)

google_net = nn.Sequential(b1, b2, b3, b4, b5, nn.Linear(1024, 10))

def print_shapes():
    X = torch.rand((1, 1, 96, 96))
    for layer in google_net:
        X = layer(X)
        print(layer.__class__.__name__, 'output shape:\t', X.shape)


def main():
    print_shapes()


if __name__ == "__main__":
    main()