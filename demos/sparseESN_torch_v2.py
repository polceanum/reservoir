import torch
import numpy as np
from adabelief_pytorch import AdaBelief
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression

def plot_power_law(x, y, a, b):
    # Generate a range of x values for plotting the power law curve
    x_range = np.linspace(min(x), max(x), 100)
    y_range = a * x_range ** b

    # Plot the original data points
    plt.scatter(x, y, color='blue', label='Original Data Points + extrapolation')

    # Plot the power law curve
    plt.plot(x_range, y_range, color='red', label=f'Estimated Power Law (y = {a:.2f}x^{b:.2f})')

    # Additional plot settings
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Power Law Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

# Neural Network Definition
class SimpleNN(torch.nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = torch.nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.linear(x)

# Function Definitions
def load_data(file_path):
    data = np.loadtxt(file_path)
    return torch.tensor(data, dtype=torch.float32)

def estimate_power_law(x, y):
    """
    Estimates the parameters a and b for the power law y = ax^b.

    Args:
    x (list or numpy.array): The x values of the data points.
    y (list or numpy.array): The y values of the data points.

    Returns:
    tuple: Estimated parameters a and b.
    """
    # Taking logarithms
    log_x = np.log(x)
    log_y = np.log(y)

    # Reshape for sklearn
    log_x = log_x.reshape(-1, 1)

    # Perform linear regression
    model = LinearRegression().fit(log_x, log_y)

    # The coefficient (slope) is b
    b = model.coef_[0]

    # To find a, we use the intercept, which is log(a)
    a = np.exp(model.intercept_)

    return a, b

def initialize_reservoir(resSize, density):
    indices = torch.randint(0, resSize, (2, int(resSize**2 * density)), dtype=torch.long)
    values = (torch.rand(int(resSize**2 * density), dtype=torch.float32) - 0.5) * 2
    W_sparse = torch.sparse_coo_tensor(indices, values, (resSize, resSize), dtype=torch.float32)
    return W_sparse

def estimate_rho(resSize, density):
    print('Estimating rho for reservoir of size', resSize, '...')
    x = np.array([10, 100, 300, 600, 900, 1000], dtype=int)
    y = np.empty(x.shape, dtype=np.float32)
    for i in range(len(x)):
        W_sparse = initialize_reservoir(x[i], density)
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
        y[i] = rhoW
        print(x[i], '->', y[i])
    
    a, b = estimate_power_law(x, y)
    print('y =', a, '*', resSize, '^', b)

    rho = a*(resSize**b)

    # plot_power_law(np.concatenate([x, [resSize]]), np.concatenate([y, np.array([rho], dtype=np.float32)]), a, b)

    print('done:', rho)
    return rho

def initialize_weights_sparse(resSize, inSize, density, rho=None):
    Win = (torch.rand(resSize, 1 + inSize, dtype=torch.float32) - 0.5) * 1

    # Initialize W as a sparse matrix
    W_sparse = initialize_reservoir(resSize, density)

    # Normalize W
    if rho is None:
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
    else:
        rhoW = rho
    W_sparse = W_sparse * (1.25 / rhoW)

    return Win, W_sparse

def run_reservoir_sparse(data, Win, W_sparse, trainLen, initLen, resSize, a):
    print('Running reservoir...')
    X = torch.zeros((1 + inSize + resSize, trainLen - initLen), dtype=torch.float32)
    x = torch.zeros((resSize, 1), dtype=torch.float32)
    for t in range(trainLen):
        u = data[t]
        x = (1 - a) * x + a * torch.tanh(Win @ torch.vstack((torch.tensor([1.0], dtype=torch.float32), u.view(1, 1))) + W_sparse @ x)
        if t >= initLen:
            X[:, t - initLen] = torch.vstack((torch.tensor([1.0], dtype=torch.float32), u.view(1, 1), x))[:, 0]
    print('done.')
    return X, x

def train_output(X, Yt, reg):
    identity_matrix = torch.eye(1 + inSize + resSize, dtype=torch.float32)
    X_T = X.T
    return torch.linalg.solve(X @ X_T + reg * identity_matrix, X @ Yt)

def train_output_with_gd(X, Yt, learning_rate, epochs):
    input_size = X.shape[0]
    output_size = Yt.shape[1]
    
    model = SimpleNN(input_size, output_size)
    criterion = torch.nn.MSELoss()
    # optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate)
    optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-8)
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X.T)
        loss = criterion(output, Yt)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')
    return model

def run_generative_mode(data, model, Win, W, testLen, trainLen, a, initial_x):
    Y = torch.zeros((outSize, testLen), dtype=torch.float32)
    u = data[trainLen]
    x = initial_x
    for t in range(testLen):
        x = (1 - a) * x + a * torch.tanh(Win @ torch.vstack((torch.tensor([1.0], dtype=torch.float32), u.view(1, 1))) + W @ x)
        input_to_model = torch.vstack((torch.tensor([1.0], dtype=torch.float32), u.view(1, 1), x)).T.flatten()
        y = model(input_to_model.view(1, -1)).view(-1)
        Y[:, t] = y
        u = y
    return Y


def compute_mse(data, Y, trainLen, errorLen):
    return torch.mean((data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]) ** 2).item()

def plot_results(data, Y, X, model, trainLen, testLen):
    figure(1).clear()
    plot(data[trainLen + 1:trainLen + testLen + 1].numpy(), 'g')
    plot(Y.T.detach().numpy(), 'b')
    title('Target and generated signals $y(n)$ starting at $n=0$')
    legend(['Target signal', 'Free-running predicted signal'])

    figure(2).clear()
    plot(X[0:20, 0:200].T.numpy())
    title('Some reservoir activations $\\mathbf{x}(n)$')

    figure(3).clear()
    bar(range(1 + inSize + resSize), model.linear.weight.detach().numpy().squeeze())
    title('Output weights $\\mathbf{W}^{out}$')

    show()

def main():
    torch.manual_seed(42)
    global inSize, outSize, resSize
    inSize = outSize = 1
    resSize = 10000
    a = 0.3
    reg = 1e-8
    trainLen = 2000
    testLen = 2000
    initLen = 100
    errorLen = 500
    learning_rate = 0.0001
    epochs = 50000
    density = 0.1

    data = load_data('../data/MackeyGlass_t17.txt')
    Win, W = initialize_weights_sparse(resSize, inSize, density, rho=estimate_rho(resSize, density))
    X, final_x_state = run_reservoir_sparse(data, Win, W, trainLen, initLen, resSize, a)
    Yt = data[None, initLen + 1:trainLen + 1].clone().detach().to(dtype=torch.float32).T

    # Choose one of the training methods
    # Wout = train_output(X, Yt, reg)
    model = train_output_with_gd(X, Yt, learning_rate, epochs)

    Y = run_generative_mode(data, model, Win, W, testLen, trainLen, a, final_x_state)
    mse = compute_mse(data, Y, trainLen, errorLen)

    print('MSE =', mse)
    plot_results(data, Y, X, model, trainLen, testLen)

if __name__ == "__main__":
    main()
