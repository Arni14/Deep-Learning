import torch
from torch import nn
from torch import functional as F

# Implement a block that takes two blocks as an argument, say net1 and net2 and returns the 
# concatenated output of both networks in the forward propagation. This is also called a parallel block.
class ParallelBlock(nn.Module):
	def __init__(self, block1, block2):
		super().__init__()
		self.block1 = block1
		self.block2 = block2

	def forward(self, X):
		out1 = self.block1(X)
		out2 = self.block2(X)
		return torch.cat((out1, out2), 0)

# Assume that you want to concatenate multiple instances of the same network. Implement a factory 
# function that generates multiple instances of the same block and build a larger network from it.
class Factory(nn.Module):
	def __init__(self, num_rep, block):
		super().__init__()
		self.modules = [block for _ in range(num_rep)]
		self.net = nn.Sequential(*self.modules)

	def forward(self, X):
		return self.net(X)

class MLP(nn.Module):
    # Declare a layer with model parameters. Here, we declare two fully
    # connected layers
    def __init__(self):
        # Call the constructor of the `MLP` parent class `Module` to perform
        # the necessary initialization. In this way, other function arguments
        # can also be specified during class instantiation, such as the model
        # parameters, `params` (to be described later)
        super().__init__()
        self.hidden = nn.Linear(20, 256)  # Hidden layer
        self.out = nn.Linear(256, 10)  # Output layer

    # Define the forward propagation of the model, that is, how to return the
    # required model output based on the input `X`
    def forward(self, X):
        # Note here we use the funtional version of ReLU defined in the
        # nn.functional module.
        return self.out(F.relu(self.hidden(X)))

# We can also create custom layers if need be
class MyLayer(nn.Module):
	def __init__(self):
		super().__init__()

	def forward(self, X):
		return (X - X.mean()) / X.std()

# Custom linear layer in PyTorch!
class MyLinear(nn.Module):
	def __init__(self, in_units, out_units):
		super().__init__()
		self.weight = nn.Parameter(torch.randn(in_units, out_units))
		self.bias = nn.Parameter(torch.randn(out_units,))

	def forward(self, X):
		return torch.matmul(X, self.weight.data) + self.bias.data

# Design a layer that takes an input and computes a tensor reduction.
class TensorReduction(nn.Module):
	def __init__(self, i, j, k):
		super().__init__()
		self.weight = nn.Parameter(torch.randn(i, j, k))

	def forward(self, X):
		tensor_transform = torch.matmul(self.weight, X).t()
		return torch.matmul(X.t(), tensor_transform)

if __name__ == "__main__":
	tr = TensorReduction(2, 2, 2)
	X = torch.tensor([1, 2], dtype=torch.float32)
	print(tr.weight.data)
	print(X)
	print(tr(X))


