import argparse
import torch
import numpy as np
# from adabelief_pytorch import AdaBelief
from matplotlib.pyplot import *
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from tqdm import tqdm

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
def load_data(file_path, dtype=torch.float64):
    data = np.loadtxt(file_path)
    return torch.tensor(data, dtype=dtype)

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

def initialize_reservoir(resSize, density, dtype=torch.float64, csr=True):
    indices = torch.randint(0, resSize, (2, int(resSize**2 * density)), dtype=torch.long)
    values = (torch.rand(int(resSize**2 * density), dtype=dtype) - 0.5)
    W_sparse = torch.sparse_coo_tensor(indices, values, (resSize, resSize), dtype=dtype)
    if csr: W_sparse = W_sparse.to_sparse_csr()
    return W_sparse

def estimate_rho(resSize, density, dtype=torch.float64):
    print('Estimating rho for reservoir of size', resSize, '...')
    # x = np.array([100, 300, 600, 900, 1000], dtype=int)
    x = torch.from_numpy(np.array([1000, 1200, 1400, 1600], dtype=int))
    y = torch.empty(x.shape, dtype=dtype)
    for i in range(len(x)):
        W_sparse = initialize_reservoir(x[i], density)
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
        y[i] = rhoW
        print(x[i], '->', y[i])
    
    a, b = estimate_power_law(x, y)
    print('y =', a, '*', resSize, '^', b)

    rho = a*(resSize**b)

    # plot_power_law(np.concatenate([x, [resSize]]), np.concatenate([y, np.array([rho], dtype=dtype)]), a, b)

    print('done:', rho)
    return rho

def initialize_weights_sparse(resSize, inSize, inInterSize, density, rho=None, dtype=torch.float64):
    Win = (torch.rand(inInterSize, 1 + inSize, dtype=dtype) - 0.5) * 1

    # Initialize W as a sparse matrix
    W_sparse = initialize_reservoir(resSize, density, dtype=dtype)

    # Normalize W
    if rho is None:
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).to(dtype=dtype).item()
    else:
        rhoW = rho
    W_sparse = W_sparse * (1.25 / rhoW)

    return Win, W_sparse

def run_reservoir_sparse(data, Win, W_sparse, trainLen, initLen, resSize, a, dtype=torch.float64):
    print('Running reservoir...')
    X = torch.zeros((1 + inSize + resSize, trainLen - initLen), dtype=dtype)
    x = torch.zeros((resSize, 1), dtype=dtype)
    Win_padding = torch.zeros((resSize-Win.shape[0], Win.shape[1]-1), dtype=dtype)
    for t in tqdm(range(trainLen)):
        u = data[t]
        Win_out = Win @ torch.vstack((torch.tensor([1.0], dtype=dtype), u.view(1, 1)))
        Win_out = torch.concatenate([Win_out, Win_padding])
        x = (1 - a) * x + a * torch.tanh(Win_out + W_sparse @ x)
        if t >= initLen:
            X[:, t - initLen] = torch.vstack((torch.tensor([1.0], dtype=dtype), u.view(1, 1), x))[:, 0]
    print('done.')
    return X, x

def train_output(X, Yt, input_size, r_out_size, reg, dtype=torch.float64):
    if X.shape[0] < 1+input_size+r_out_size: r_out_size = X.shape[0]
    else: X = X[:1+input_size+r_out_size]
    identity_matrix = torch.eye(X.shape[0], dtype=dtype)
    X_T = X.T
    return torch.linalg.solve(X @ X_T + reg * identity_matrix, X @ Yt)

def train_output_with_gd(X, Yt, input_size, r_out_size, learning_rate, epochs, args, dtype=torch.float64):
    if X.shape[0] < 1+input_size+r_out_size: r_out_size = X.shape[0]
    else: X = X[:1+input_size+r_out_size]
    output_size = Yt.shape[1]
    
    model = SimpleNN(1+input_size+r_out_size, output_size).to(dtype)
    criterion = torch.nn.MSELoss()
    
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    if args.opt == 'adamw':
        optimizer = torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True)
    elif args.opt == 'adagrad':
        optimizer = torch.optim.Adagrad(model.parameters(), lr=learning_rate)
    elif args.opt == 'rprop':
        optimizer = torch.optim.Rprop(model.parameters(), lr=learning_rate)
    elif args.opt == 'rmsprop':
        optimizer = torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    else:
        print('Optimizer not implemented:', args.opt)
    # optimizer = AdaBelief(model.parameters(), lr=learning_rate, eps=1e-8)
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

def run_generative_mode(data, model, Win, W, r_out_size, testLen, trainLen, a, initial_x, args, dtype=torch.float64):
    # Win_full = torch.concatenate([Win, torch.zeros((W.shape[0]-Win.shape[0], Win.shape[1]))])
    Y = torch.zeros((outSize, testLen), dtype=dtype)
    u = data[trainLen]
    x = initial_x
    for t in range(testLen):
        Win_out = Win @ torch.vstack((torch.tensor([1.0], dtype=dtype), u.view(1, 1)))
        Win_out = torch.concatenate([Win_out, torch.zeros((W.shape[0]-Win_out.shape[0], Win_out.shape[1]))])
        x = (1 - a) * x + a * torch.tanh(Win_out + W @ x)
        input_to_model = torch.vstack((torch.tensor([1.0], dtype=dtype), u.view(1, 1), x[:r_out_size])).T.flatten()
        if args.opt == 'lr':
            y = model.T @ input_to_model # direct method
        else:
            y = model(input_to_model.view(1, -1)).view(-1)
        Y[:, t] = y
        u = y
    return Y


def compute_mse(data, Y, trainLen, errorLen):
    return torch.mean((data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]) ** 2).item()

def plot_results(data, Y, X, model, r_out_size, trainLen, testLen, args):
    figure(1).clear()
    plot(data[trainLen + 1:trainLen + testLen + 1].numpy(), 'g')
    plot(torch.clip(Y.T.detach(), -10, 10).numpy(), 'b') # clip to ignore extremely large values when plotting
    title('Target and generated signals $y(n)$ starting at $n=0$')
    legend(['Target signal', 'Free-running predicted signal'])

    figure(2).clear()
    plot(X[0:20, 0:200].T.numpy())
    title('Some reservoir activations $\\mathbf{x}(n)$')

    figure(3).clear()
    if args.opt == 'lr':
        bar(range(1 + inSize + r_out_size), model.numpy().squeeze())
    else:
        bar(range(1 + inSize + r_out_size), model.linear.weight.detach().numpy().squeeze())
    
    title('Output weights $\\mathbf{W}^{out}$')

    show()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('-fp', type=int, default=64, choices=[16, 32, 64], help='float precision')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adamw', 'adagrad', 'rprop', 'rmsprop', 'lr'], help='optimisation')
    args = parser.parse_args()

    torch.manual_seed(42)
    global inSize, outSize, resSize
    inSize = outSize = 1
    resSize = 1000
    a = 0.3
    reg = 1e-8
    trainLen = 2000
    testLen = 2000
    initLen = 100
    errorLen = 500
    learning_rate = args.lr
    epochs = 10000
    density = 0.1
    inInterSize = 1000
    outInterSize = 1000
    dtype = {64:torch.float64, 32:torch.float32, 16:torch.float16}[args.fp]

    data = load_data('../data/MackeyGlass_t17.txt', dtype=dtype)
    Win, W = initialize_weights_sparse(resSize, inSize, inInterSize, density, rho=estimate_rho(resSize, density, dtype=dtype), dtype=dtype)
    X, final_x_state = run_reservoir_sparse(data, Win, W, trainLen, initLen, resSize, a, dtype=dtype)
    Yt = data[None, initLen + 1:trainLen + 1].clone().detach().to(dtype=dtype).T

    # Choose one of the training methods
    if args.opt == 'lr':
        model = train_output(X, Yt, inSize, outInterSize, reg, dtype=dtype)
    else:
        model = train_output_with_gd(X, Yt, inSize, outInterSize, learning_rate, epochs, args, dtype=dtype)

    Y = run_generative_mode(data, model, Win, W, outInterSize, testLen, trainLen, a, final_x_state, args=args, dtype=dtype)
    mse = compute_mse(data, Y, trainLen, errorLen)

    print('MSE =', mse)
    plot_results(data, Y, X, model, outInterSize, trainLen, testLen, args)

if __name__ == "__main__":
    main()
