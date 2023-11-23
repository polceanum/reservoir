import torch
import numpy as np
from matplotlib.pyplot import *

# Load the data
data = np.loadtxt('../data/MackeyGlass_t17.txt')
data = torch.tensor(data, dtype=torch.float64)

# Set manual seed for reproducibility
torch.manual_seed(42)

# Parameters
trainLen = 2000
testLen = 2000
initLen = 100
inSize = outSize = 1
resSize = 1000
a = 0.3  # leaking rate
reg = 1e-8  # regularization coefficient

# Initialize input weights
Win = (torch.rand(resSize, 1 + inSize, dtype=torch.float64) - 0.5) * 1

# Initialize reservoir weights
W_dense = torch.rand(resSize, resSize, dtype=torch.float64) - 0.5
# PyTorch does not have built-in sparse matrices like JAX. We'll use dense matrix here.
rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
W = W_dense * (1.25 / rhoW)

# Allocate memory for the design matrix
X = torch.zeros((1 + inSize + resSize, trainLen - initLen), dtype=torch.float64)
Yt = data[None, initLen + 1:trainLen + 1].clone().detach().to(dtype=torch.float64).T

# Run the reservoir
x = torch.zeros((resSize, 1), dtype=torch.float64)
for t in range(trainLen):
    u = data[t]
    x = (1 - a) * x + a * torch.tanh(Win @ torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1))) + W @ x)
    if t >= initLen:
        X[:, t - initLen] = torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1), x))[:, 0]

# Train the output
identity_matrix = torch.eye(1 + inSize + resSize, dtype=torch.float64)
X_T = X.T
Wout = torch.linalg.solve(X @ X_T + reg * identity_matrix, X @ Yt)

# Run in generative mode
Y = torch.zeros((outSize, testLen), dtype=torch.float64)
u = data[trainLen]
for t in range(testLen):
    x = (1 - a) * x + a * torch.tanh(Win @ torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1))) + W @ x)
    y = Wout.T @ torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1), x))
    Y[:, t] = y.view(-1)
    u = y.view(-1)

# Compute MSE
errorLen = 500
mse = torch.mean((data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]) ** 2)
print('MSE = ' + str(mse.item()))

# Plotting
figure(1).clear()
plot(data[trainLen + 1:trainLen + testLen + 1].numpy(), 'g')
plot(Y.T.numpy(), 'b')
title('Target and generated signals $y(n)$ starting at $n=0$')
legend(['Target signal', 'Free-running predicted signal'])

figure(2).clear()
plot(X[0:20, 0:200].T.numpy())
title('Some reservoir activations $\\mathbf{x}(n)$')

figure(3).clear()
bar(range(1 + inSize + resSize), Wout.T.squeeze().numpy())
title('Output weights $\\mathbf{W}^{out}$')

show()
