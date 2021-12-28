import torch
from torch import nn
from torch import functional as F
import numpy as np
import matplotlib.pyplot as plt

# Pi from numpy
pi = torch.from_numpy(np.array(np.pi))
x = torch.linspace(-pi, pi, 200, dtype=torch.float64, requires_grad=True)
# Compute the sine of x
y = torch.sin(x)
# Compute the cosine of x (gradient of y)
y.sum().backward()
z = x.grad

# Now, do some plotting
x, y, z = x.detach().numpy(), y.detach().numpy(), z.detach().numpy()
plt.plot(x, y)
plt.plot(x, z)
plt.show()