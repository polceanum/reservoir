import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
from tqdm import tqdm

from plot_utils import plot_results, plot_power_law, plot_geometric_graph

# Neural Network Definition
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.linear(x)

# Transformer-based Language Model
class SimpleTransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, num_heads=8, hidden_size=256):
        super(SimpleTransformerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)  # Added linear layer
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(1) # Transformer expects (seq_len, batch, embedding)
        x = self.transformer_decoder(x, x)
        x = x.squeeze(1)
        return self.fc2(x)

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

def initialize_reservoir(resSize, density, topology, dtype=torch.float64, csr=True, allow_plot=False):
    dtype2tensor = {torch.float16:torch.HalfTensor, torch.float32:torch.FloatTensor, torch.float64:torch.DoubleTensor}
    dtype2sparsetensor = {torch.float16:torch.sparse.HalfTensor, torch.float32:torch.sparse.FloatTensor, torch.float64:torch.sparse.DoubleTensor}

    if topology == 'uniform':
        indices = torch.randint(0, resSize, (2, int(resSize**2 * density)), dtype=torch.long)
        values = (torch.rand(int(resSize**2 * density), dtype=dtype) - 0.5)
        W_sparse = torch.sparse_coo_tensor(indices, values, (resSize, resSize), dtype=dtype)
    elif topology == 'geometric':
        G = nx.thresholded_random_geometric_graph(resSize, 0.05, 0.0001)
        A = nx.to_scipy_sparse_array(G, format='coo')

        idx = torch.LongTensor(np.vstack((A.row, A.col)))
        val = dtype2tensor[dtype](A.data)
        shape = A.shape
        W_sparse = dtype2sparsetensor[dtype](idx, val, torch.Size(shape))

        if allow_plot: plot_geometric_graph(G)
    elif topology == 'smallworld':
        G = nx.navigable_small_world_graph(resSize, p=3)
        A = nx.to_scipy_sparse_array(G, format='coo')

        idx = torch.LongTensor(np.vstack((A.row, A.col)))
        val = dtype2tensor[dtype](A.data)
        shape = A.shape
        W_sparse = dtype2sparsetensor[dtype](idx, val, torch.Size(shape))

        if allow_plot: plot_geometric_graph(G)
    if csr: W_sparse = W_sparse.to_sparse_csr()
    return W_sparse

def estimate_rho(resSize, density, topology, dtype=torch.float64, allow_plot=False):
    print('Estimating rho for reservoir of size', resSize, '...')
    # x = torch.from_numpy(np.array([1000, 1500, 2000, 2500, 3000], dtype=int))
    x = torch.from_numpy(np.array([1000, 1250, 1500, 1750, 2000], dtype=int))
    y = torch.empty(x.shape, dtype=dtype)
    for i in range(len(x)):
        W_sparse = initialize_reservoir(x[i], density, topology, allow_plot=allow_plot)
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
        y[i] = rhoW
        print(x[i], '->', y[i])
    
    a, b = estimate_power_law(x, y)
    print('y =', a, '*', resSize, '^', b)

    rho = a*(resSize**b)

    if allow_plot: plot_power_law(np.concatenate([x, [resSize]]), np.concatenate([y, np.array([rho])]), a, b)

    return rho

def initialize_weights_sparse(resSize, inSize, inInterSize, density, topology, rho=None, dtype=torch.float64, allow_plot=True):
    Win = (torch.rand(inInterSize, 1 + inSize, dtype=dtype) - 0.5) * 1

    # Initialize W as a sparse matrix
    W_sparse = initialize_reservoir(resSize, density, topology, dtype=dtype, allow_plot=allow_plot)

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
    
    if args.read_out == 'linear':
        model = model = SimpleNN(1+input_size+r_out_size, output_size).to(dtype)
    elif args.read_out == 'transformer':
        model = SimpleTransformerModel(1+input_size+r_out_size, output_size).to(dtype)
    else:
        print('Read-out not implemented:', args.read_out)

    criterion = torch.nn.MSELoss()
    
    if args.opt == 'adam':
        optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True)
    elif args.opt == 'adamw':
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


def main():
    parser = argparse.ArgumentParser()

    # Training
    parser.add_argument('-fp', type=int, default=64, choices=[16, 32, 64], help='float precision')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adamw', 'adagrad', 'rprop', 'rmsprop', 'lr'], help='optimisation')
    parser.add_argument('-epochs', type=int, default=10000, help='number of epochs')

    # Reservoir
    parser.add_argument('-top', type=str, default='uniform', choices=['uniform', 'geometric', 'smallworld'], help='reservoir topology')
    parser.add_argument('-dim-res', type=int, default=1000, help='reservoir size')
    parser.add_argument('-rho', type=float, default=0.01, help='reservoir density')
    parser.add_argument('-alpha', type=float, default=0.3, help='reservoir leak rate')
    parser.add_argument('-rest', action='store_true', help='reservoir spectral radius estimation')

    # Read-out
    parser.add_argument('-read-out', type=str, default='linear', choices=['linear', 'transformer'], help='reservoir readout architecture')

    # Valves
    parser.add_argument('-valve-in', type=int, default=1000, help='input valve size (size of input slice to reservoir)')
    parser.add_argument('-valve-out', type=int, default=1000, help='output valve size (size of output slice from reservoir)')

    # Data in/out dimensions
    parser.add_argument('-dim-in', type=int, default=1, help='input size')
    parser.add_argument('-dim-out', type=int, default=1, help='output size')

    # Visualization
    parser.add_argument('-viz', action='store_true', help='plot reservoir information')
    
    args = parser.parse_args()

    torch.manual_seed(42)
    global inSize, outSize, resSize
    inSize = args.dim_in
    outSize = args.dim_out
    resSize = args.dim_res
    a = args.alpha
    reg = 1e-8
    trainLen = 2000
    testLen = 2000
    initLen = 100
    errorLen = 500
    learning_rate = args.lr
    epochs = args.epochs
    density = args.rho
    inInterSize = args.valve_in
    outInterSize = args.valve_out
    dtype = {64:torch.float64, 32:torch.float32, 16:torch.float16}[args.fp]

    data = load_data('../data/MackeyGlass_t17.txt', dtype=dtype)

    Win, W = initialize_weights_sparse(resSize, inSize, inInterSize, density, topology=args.top, rho=(estimate_rho(resSize, density, topology=args.top, dtype=dtype) if args.rest else None), dtype=dtype, allow_plot=args.viz)
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
    if args.viz:
        plot_results(data, Y, X, model, inSize, outInterSize, trainLen, testLen, args)

if __name__ == "__main__":
    main()
