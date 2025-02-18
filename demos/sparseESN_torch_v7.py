import argparse
import torch
import torch.nn as nn
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
from tqdm import tqdm

from plot_utils_batched import plot_results, plot_power_law, plot_geometric_graph

# --------------------------
# Neural Network Definitions
# --------------------------
class SimpleNN(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleNN, self).__init__()
        self.linear = nn.Linear(input_size, output_size, bias=True)

    def forward(self, x):
        return self.linear(x)


class SimpleTransformerModel(nn.Module):
    def __init__(self, input_size, output_size, num_layers=1, num_heads=8, hidden_size=256):
        super(SimpleTransformerModel, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        # Using batch_first=True so inputs are (batch, seq, embedding)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(
            d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x):
        # x is expected to be of shape (batch, input_size)
        x = self.fc1(x)                     # (batch, hidden_size)
        x = x.unsqueeze(1)                  # (batch, 1, hidden_size)
        x = self.transformer_decoder(x, x)   # (batch, 1, hidden_size)
        x = x.squeeze(1)                    # (batch, hidden_size)
        return self.fc2(x)                  # (batch, output_size)


# --------------------------
# Data and Helper Functions
# --------------------------
def load_data(file_path, dtype, device):
    # Load data from file. Assumes a single-column time series.
    data = np.loadtxt(file_path)
    return torch.tensor(data, dtype=dtype, device=device)


def estimate_power_law(x, y):
    log_x = np.log(x)
    log_y = np.log(y)
    log_x = log_x.reshape(-1, 1)
    model = LinearRegression().fit(log_x, log_y)
    b = model.coef_[0]
    a = np.exp(model.intercept_)
    return a, b


def initialize_reservoir(resSize, density, topology, dtype, device, csr=True, allow_plot=False):
    tensor_func = {
        torch.float16: torch.HalfTensor,
        torch.float32: torch.FloatTensor,
        torch.float64: torch.DoubleTensor
    }
    sparsetensor_func = {
        torch.float16: torch.sparse.HalfTensor,
        torch.float32: torch.sparse.FloatTensor,
        torch.float64: torch.sparse.DoubleTensor
    }

    if topology == 'uniform':
        num_elements = int(resSize**2 * density)
        indices = torch.randint(0, resSize, (2, num_elements), dtype=torch.long, device=device)
        values = (torch.rand(num_elements, dtype=dtype, device=device) - 0.5)
        W_sparse = torch.sparse_coo_tensor(indices, values, (resSize, resSize),
                                             dtype=dtype, device=device)
    elif topology in ['geometric', 'smallworld']:
        if topology == 'geometric':
            G = nx.thresholded_random_geometric_graph(resSize, 0.05, 0.0001)
        else:
            G = nx.navigable_small_world_graph(resSize, p=3)
        A = nx.to_scipy_sparse_array(G, format='coo')
        idx = torch.LongTensor(np.vstack((A.row, A.col))).to(device)
        val = tensor_func[dtype](A.data).to(device)
        W_sparse = sparsetensor_func[dtype](idx, val, torch.Size(A.shape)).to(device)
        if allow_plot:
            plot_geometric_graph(G)
    else:
        raise ValueError("Unknown topology: " + topology)
    
    if csr:
        W_sparse = W_sparse.to_sparse_csr()
    return W_sparse


def estimate_rho(resSize, density, topology, dtype, device, allow_plot=False):
    print('Estimating rho for reservoir of size', resSize, '...')
    x_vals = np.array([1000, 1250, 1500, 1750, 2000], dtype=int)
    y_vals = np.empty(x_vals.shape, dtype=np.float64)
    for i, current_size in enumerate(x_vals):
        W_sparse = initialize_reservoir(current_size, density, topology, dtype, device, allow_plot=allow_plot)
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
        y_vals[i] = rhoW
        print(current_size, '->', rhoW)
    
    a, b = estimate_power_law(x_vals, y_vals)
    print(f'y = {a} * resSize^{b}')
    rho = a * (resSize ** b)
    if allow_plot:
        xs = np.concatenate([x_vals, [resSize]])
        ys = np.concatenate([y_vals, np.array([rho])])
        plot_power_law(xs, ys, a, b)
    return rho


def initialize_weights_sparse(resSize, inSize, inInterSize, density, topology, dtype, device, rho=None, allow_plot=True):
    # Win: shape (inInterSize, 1 + inSize)
    Win = (torch.rand(inInterSize, 1 + inSize, dtype=dtype, device=device) - 0.5)
    W_sparse = initialize_reservoir(resSize, density, topology, dtype, device, allow_plot=allow_plot)
    if rho is None:
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_sparse.to_dense()))).to(dtype=dtype).item()
    else:
        rhoW = rho
    W_sparse = W_sparse * (1.25 / rhoW)
    return Win, W_sparse


def run_reservoir_sparse(data, Win, W_sparse, trainLen, initLen, resSize, a, inSize, dtype, device):
    """
    data: shape (trainLen, batch, inSize)
    Reservoir state x: shape (batch, resSize)
    For each time step, form input [1, u] and update reservoir state.
    """
    batch_size = data.shape[1]
    num_steps = trainLen - initLen
    feature_dim = 1 + inSize + resSize  # concatenated vector: [1, u, x]
    X = torch.zeros((batch_size, num_steps, feature_dim), dtype=dtype, device=device)
    x = torch.zeros((batch_size, resSize), dtype=dtype, device=device)
    one = torch.ones((batch_size, 1), dtype=dtype, device=device)
    inInterSize = Win.shape[0]
    win_padding = torch.zeros((batch_size, resSize - inInterSize), dtype=dtype, device=device)
    
    # Convert W_sparse to COO for supported sparse multiplication.
    W_coo = W_sparse.to_sparse_coo()

    for t in tqdm(range(trainLen)):
        u = data[t]  # shape (batch, inSize)
        input_vec = torch.cat([one, u], dim=1)  # (batch, 1+inSize)
        Win_out = input_vec @ Win.T             # (batch, inInterSize)
        Win_full = torch.cat([Win_out, win_padding], dim=1)  # (batch, resSize)
        # Use sparse multiplication: compute (W_coo @ x.T).T which is (batch, resSize)
        x = (1 - a) * x + a * torch.tanh(Win_full + torch.sparse.mm(W_coo, x.T).T)
        if t >= initLen:
            X[:, t - initLen, :] = torch.cat([one, u, x], dim=1)
    print('Reservoir run complete.')
    return X, x


def train_output(X, Yt, reg, dtype, device):
    """
    X: shape (batch, num_steps, feature_dim)
    Yt: shape (batch, num_steps, output_size)
    Flatten batch and time dimensions and solve for readout weights.
    """
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Yt.reshape(-1, Yt.shape[-1])
    A = X_flat.T @ X_flat + reg * torch.eye(X_flat.shape[-1], dtype=dtype, device=device)
    B = X_flat.T @ Y_flat
    W_out = torch.linalg.solve(A, B)
    return W_out


def train_output_with_gd(X, Yt, learning_rate, epochs, args, dtype, device):
    """
    X: shape (batch, num_steps, feature_dim)
    Yt: shape (batch, num_steps, output_size)
    Flatten batch and time dimensions for training.
    """
    X_flat = X.reshape(-1, X.shape[-1])
    Y_flat = Yt.reshape(-1, Yt.shape[-1])
    feature_dim = X_flat.shape[-1]
    output_size = Y_flat.shape[-1]
    
    if args.read_out == 'linear':
        model = SimpleNN(feature_dim, output_size).to(device).to(dtype)
    elif args.read_out == 'transformer':
        model = SimpleTransformerModel(feature_dim, output_size).to(device).to(dtype)
    else:
        raise NotImplementedError(f"Read-out not implemented: {args.read_out}")

    criterion = nn.MSELoss()
    optimizers = {
        'adam': torch.optim.Adam(model.parameters(), lr=learning_rate, amsgrad=True),
        'adamw': torch.optim.AdamW(model.parameters(), lr=learning_rate, amsgrad=True),
        'adagrad': torch.optim.Adagrad(model.parameters(), lr=learning_rate),
        'rprop': torch.optim.Rprop(model.parameters(), lr=learning_rate),
        'rmsprop': torch.optim.RMSprop(model.parameters(), lr=learning_rate)
    }
    optimizer = optimizers.get(args.opt)
    if optimizer is None:
        raise NotImplementedError(f"Optimizer not implemented: {args.opt}")
    
    for epoch in range(epochs):
        model.train()
        optimizer.zero_grad()
        output = model(X_flat)
        loss = criterion(output, Y_flat)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')
    return model


def run_generative_mode(data, model, Win, W, r_out_size, testLen, trainLen, a, initial_x, args, inSize, outSize, dtype, device):
    """
    data: shape (time, batch, inSize)
    initial_x: reservoir state, shape (batch, resSize)
    For each generative time step, update reservoir state in batch.
    Readout input: [1, u, x[:, :r_out_size]]
    """
    batch_size = data.shape[1]
    Y = torch.zeros((batch_size, testLen, outSize), dtype=dtype, device=device)
    one = torch.ones((batch_size, 1), dtype=dtype, device=device)
    inInterSize = Win.shape[0]
    resSize = W.shape[0]
    win_padding = torch.zeros((batch_size, resSize - inInterSize), dtype=dtype, device=device)
    
    # Convert reservoir weights to COO.
    W_coo = W.to_sparse_coo()
    
    u = data[trainLen]  # shape (batch, inSize)
    x = initial_x       # shape (batch, resSize)
    
    for t in range(testLen):
        input_vec = torch.cat([one, u], dim=1)           # (batch, 1+inSize)
        Win_out = input_vec @ Win.T                        # (batch, inInterSize)
        Win_full = torch.cat([Win_out, win_padding], dim=1)  # (batch, resSize)
        x = (1 - a) * x + a * torch.tanh(Win_full + torch.sparse.mm(W_coo, x.T).T)
        input_to_model = torch.cat([one, u, x[:, :r_out_size]], dim=1)  # (batch, 1+inSize+r_out_size)
        if args.opt == 'lr':
            # For linear readout, model is the weight matrix.
            y = input_to_model @ model
        else:
            y = model(input_to_model)
        Y[:, t, :] = y
        u = y  # Feedback for next step.
    return Y


def compute_mse(data, Y, trainLen, errorLen):
    """
    data: shape (time, batch, outSize)
    Y: shape (batch, testLen, outSize)
    """
    target = data[trainLen+1: trainLen+errorLen+1]  # (errorLen, batch, outSize)
    target = target.permute(1, 0, 2)  # (batch, errorLen, outSize)
    mse = torch.mean((target - Y[:, :errorLen, :]) ** 2).item()
    return mse


# --------------------------
# Main Function
# --------------------------
def main():
    parser = argparse.ArgumentParser()

    # Training options
    parser.add_argument('-fp', type=int, default=64, choices=[16, 32, 64], help='float precision')
    parser.add_argument('-lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('-opt', type=str, default='adam', choices=['adam', 'adamw', 'adagrad', 'rprop', 'rmsprop', 'lr'], help='optimizer')
    parser.add_argument('-epochs', type=int, default=10000, help='number of epochs')

    # Reservoir options
    parser.add_argument('-top', type=str, default='uniform', choices=['uniform', 'geometric', 'smallworld'], help='reservoir topology')
    parser.add_argument('-dim-res', type=int, default=1000, help='reservoir size')
    parser.add_argument('-rho', type=float, default=0.01, help='reservoir density')
    parser.add_argument('-alpha', type=float, default=0.3, help='reservoir leak rate')
    parser.add_argument('-rest', action='store_true', help='enable reservoir spectral radius estimation')

    # Read-out
    parser.add_argument('-read-out', type=str, default='linear', choices=['linear', 'transformer'], help='readout architecture')

    # Valve sizes
    parser.add_argument('-valve-in', type=int, default=1000, help='input valve size')
    parser.add_argument('-valve-out', type=int, default=1000, help='output valve size')

    # Data dimensions
    parser.add_argument('-dim-in', type=int, default=1, help='input size')
    parser.add_argument('-dim-out', type=int, default=1, help='output size')

    # Batch size
    parser.add_argument('-batch-size', type=int, default=1, help='batch size')

    # Visualization
    parser.add_argument('-viz', action='store_true', help='plot reservoir information')

    # Device option
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model (e.g. "cpu" or "cuda")')

    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(42)
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
    outInterSize = args.valve_out  # used as r_out_size in generative mode
    dtype = {64: torch.float64, 32: torch.float32, 16: torch.float16}[args.fp]
    batch_size = args.batch_size

    # Load data and add batch dimension.
    data = load_data('../data/MackeyGlass_t17.txt', dtype, device)  # shape (T,) if single-column
    # Reshape to (T, 1, inSize)
    data = data.unsqueeze(1).unsqueeze(2)
    if batch_size > 1:
        # Replicate the same time series for all batch samples.
        data = data.repeat(1, batch_size, 1)
    
    # Estimate reservoir spectral radius if requested.
    rho_value = (estimate_rho(resSize, density, args.top, dtype, device, allow_plot=args.viz)
                 if args.rest else None)
    # Initialize input weights (Win) and reservoir weights (W)
    Win, W = initialize_weights_sparse(resSize, inSize, inInterSize, density, args.top, dtype, device,
                                         rho=rho_value, allow_plot=args.viz)
    # Run the reservoir. X has shape (batch, (trainLen-initLen), (1+inSize+resSize))
    X, final_x_state = run_reservoir_sparse(data, Win, W, trainLen, initLen, resSize, a, inSize, dtype, device)
    # Prepare target outputs from data: time steps initLen+1 to trainLen.
    Yt = data[initLen+1: trainLen+1].clone().detach()  # shape (trainLen-initLen, batch, inSize)
    Yt = Yt.permute(1, 0, 2)  # shape (batch, trainLen-initLen, inSize)

    # Train the readout.
    if args.opt == 'lr':
        model = train_output(X, Yt, reg, dtype, device)
    else:
        model = train_output_with_gd(X, Yt, learning_rate, epochs, args, dtype, device)

    # Run generative mode. initial_x is the final reservoir state; use outInterSize as r_out_size.
    Y = run_generative_mode(data, model, Win, W, outInterSize, testLen, trainLen, a, final_x_state,
                            args, inSize, outSize, dtype, device)
    mse = compute_mse(data, Y, trainLen, errorLen)
    print('MSE =', mse)
    if args.viz:
        plot_results(data, Y, X, model, inSize, outInterSize, trainLen, testLen, args)


if __name__ == "__main__":
    main()
