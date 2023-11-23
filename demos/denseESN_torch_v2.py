import torch
import numpy as np
from matplotlib.pyplot import *

def load_data(file_path):
    data = np.loadtxt(file_path)
    return torch.tensor(data, dtype=torch.float64)

def initialize_weights(resSize, inSize):
    Win = (torch.rand(resSize, 1 + inSize, dtype=torch.float64) - 0.5) * 1
    W_dense = torch.rand(resSize, resSize, dtype=torch.float64) - 0.5
    rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
    W = W_dense * (1.25 / rhoW)
    return Win, W

def run_reservoir(data, Win, W, trainLen, initLen, resSize, a):
    X = torch.zeros((1 + inSize + resSize, trainLen - initLen), dtype=torch.float64)
    x = torch.zeros((resSize, 1), dtype=torch.float64)
    for t in range(trainLen):
        u = data[t]
        x = (1 - a) * x + a * torch.tanh(Win @ torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1))) + W @ x)
        if t >= initLen:
            X[:, t - initLen] = torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1), x))[:, 0]
    return X, x

def train_output(X, Yt, reg):
    identity_matrix = torch.eye(1 + inSize + resSize, dtype=torch.float64)
    X_T = X.T
    Wout = torch.linalg.solve(X @ X_T + reg * identity_matrix, X @ Yt)
    return Wout

def run_generative_mode(data, Wout, Win, W, testLen, trainLen, a, initial_x):
    Y = torch.zeros((outSize, testLen), dtype=torch.float64)
    u = data[trainLen]
    x = initial_x  # Use the passed initial state of x
    for t in range(testLen):
        x = (1 - a) * x + a * torch.tanh(Win @ torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1))) + W @ x)
        y = Wout.T @ torch.vstack((torch.tensor([1.0], dtype=torch.float64), u.view(1, 1), x))
        Y[:, t] = y.view(-1)
        u = y.view(-1)
    return Y

def compute_mse(data, Y, trainLen, errorLen):
    return torch.mean((data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]) ** 2).item()

def plot_results(data, Y, X, Wout, trainLen, testLen):
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

def main():
    torch.manual_seed(42)  # Set seed for reproducibility

    # Parameters
    global inSize, outSize, resSize  # Define as global for use in functions
    inSize = outSize = 1
    resSize = 1000
    a = 0.3
    reg = 1e-8
    trainLen = 2000
    testLen = 2000
    initLen = 100
    errorLen = 500

    data = load_data('../data/MackeyGlass_t17.txt')
    Win, W = initialize_weights(resSize, inSize)
    X, final_x_state = run_reservoir(data, Win, W, trainLen, initLen, resSize, a)
    Yt = data[None, initLen + 1:trainLen + 1].clone().detach().to(dtype=torch.float64).T
    Wout = train_output(X, Yt, reg)
    Y = run_generative_mode(data, Wout, Win, W, testLen, trainLen, a, final_x_state)
    mse = compute_mse(data, Y, trainLen, errorLen)

    print('MSE =', mse)
    plot_results(data, Y, X, Wout, trainLen, testLen)

if __name__ == "__main__":
    main()
