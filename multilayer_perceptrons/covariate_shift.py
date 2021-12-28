import torch
from torch import nn
import numpy as np

# Create a random dataset
ones = [-1, 1]
X_train, y_train = torch.randn((256, 20), dtype=torch.float32), torch.from_numpy(np.random.choice(ones, 256))
X_test = torch.randn((50, 20), dtype=torch.float32)
y_fake_test = torch.from_numpy(np.random.choice(ones))

# Train a binary classifier using LR to get h
X_temp = torch.cat(X_train, X_test)
y_temp = torch.cat(y_train, y_fake_test)
binary_classifier = nn.Sequential(nn.Linear(256, 1), nn.Sigmoid())
loss = torch.optim.SGD(binary_classifier.parameters(), lr=0.01)