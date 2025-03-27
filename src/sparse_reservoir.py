import argparse
import torch
import torch.nn as nn
import torch.profiler
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
from tqdm import tqdm

from plot_utils import plot_results, plot_power_law, plot_geometric_graph

# New imports for Wikipedia dataset and pretrained embeddings
from datasets import load_dataset
from transformers import AutoTokenizer, AutoModel

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
        # Using batch_first=True to work with inputs of shape (batch, seq, embedding)
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x):
        # Assume x is of shape (batch, input_size)
        x = self.fc1(x)
        # Add sequence dimension: here we consider the entire batch as a sequence of length 1 per sample
        x = x.unsqueeze(1)  # Now shape is (batch, 1, hidden_size)
        x = self.transformer_decoder(x, x)
        x = x.squeeze(1)
        return self.fc2(x)

# --------------------------
# Wikipedia Data Loader with Pretrained Embeddings
# --------------------------
def load_wikipedia_data(tokenizer, model, max_length=128, num_samples=1000):
    """
    Loads a subset of the Wikipedia dataset and converts texts to embeddings.
    Uses mean pooling over the last hidden states.
    """
    # Load a small subset for speed (adjust the split as needed)
    dataset = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_samples}]", trust_remote_code=True)
    texts = [sample["text"] for sample in dataset]
    # Tokenize the texts
    inputs = tokenizer(texts, return_tensors="pt", padding=True, truncation=True, max_length=max_length)
    with torch.no_grad():
        outputs = model(**inputs)
    # Mean-pool the last hidden state to get a single embedding per sample
    embeddings = outputs.last_hidden_state.mean(dim=1)
    return embeddings

# --------------------------
# Data and Helper Functions
# --------------------------
def load_data(file_path, dtype, device):
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
    tensor_func = {torch.float16: torch.HalfTensor, torch.float32: torch.FloatTensor, torch.float64: torch.DoubleTensor}
    sparsetensor_func = {torch.float16: torch.sparse.HalfTensor, torch.float32: torch.sparse.FloatTensor, torch.float64: torch.sparse.DoubleTensor}

    if topology == 'uniform':
        num_elements = int(resSize**2 * density)
        indices = torch.randint(0, resSize, (2, num_elements), dtype=torch.long, device=device)
        values = (torch.rand(num_elements, dtype=dtype, device=device) - 0.5)
        W_sparse = torch.sparse_coo_tensor(indices, values, (resSize, resSize), dtype=dtype, device=device)
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
    x = np.array([1000, 1250, 1500, 1750, 2000], dtype=int)
    y = np.empty(x.shape, dtype=np.float64)
    for i, current_size in enumerate(x):
        W_sparse = initialize_reservoir(current_size, density, topology, dtype, device, allow_plot=allow_plot)
        W_dense = W_sparse.to_dense()
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_dense))).item()
        y[i] = rhoW
        print(current_size, '->', rhoW)
    
    a, b = estimate_power_law(x, y)
    print(f'y = {a} * resSize^{b}')
    rho = a * (resSize ** b)
    if allow_plot:
        xs = np.concatenate([x, [resSize]])
        ys = np.concatenate([y, np.array([rho])])
        plot_power_law(xs, ys, a, b)
    return rho

def initialize_weights_sparse(resSize, inSize, inInterSize, density, topology, dtype, device, rho=None, allow_plot=True):
    Win = (torch.rand(inInterSize, 1 + inSize, dtype=dtype, device=device) - 0.5)
    W_sparse = initialize_reservoir(resSize, density, topology, dtype, device, allow_plot=allow_plot)
    if rho is None:
        rhoW = torch.max(torch.abs(torch.linalg.eigvals(W_sparse.to_dense()))).to(dtype=dtype).item()
    else:
        rhoW = rho
    W_sparse = W_sparse * (1.25 / rhoW)
    return Win, W_sparse

def run_reservoir_sparse(data, Win, W_sparse, trainLen, initLen, resSize, a, inSize, dtype, device):
    X = torch.zeros((1 + inSize + resSize, trainLen - initLen), dtype=dtype, device=device)
    x = torch.zeros((resSize, 1), dtype=dtype, device=device)
    # Precompute constant tensors outside the loop. Create padding with shape (resSize - Win.shape[0], 1)
    Win_padding = torch.zeros((resSize - Win.shape[0], 1), dtype=dtype, device=device)
    one = torch.tensor([1.0], dtype=dtype, device=device)
    for t in tqdm(range(trainLen)):
        u = data[t]
        Win_out = Win @ torch.vstack((one, u.view(-1, 1)))
        Win_out = torch.cat([Win_out, Win_padding])
        x = (1 - a) * x + a * torch.tanh(Win_out + W_sparse @ x)
        if t >= initLen:
            X[:, t - initLen] = torch.vstack((one, u.view(-1, 1), x))[:, 0]
    print('Reservoir run complete.')
    return X, x


def train_output(X, Yt, input_size, r_out_size, reg, dtype, device):
    if X.shape[0] < 1 + input_size + r_out_size:
        r_out_size = X.shape[0]
    else:
        X = X[:1 + input_size + r_out_size]
    identity_matrix = torch.eye(X.shape[0], dtype=dtype, device=device)
    return torch.linalg.solve(X @ X.T + reg * identity_matrix, X @ Yt)

def train_output_with_gd(X, Yt, input_size, r_out_size, learning_rate, epochs, args, dtype, device):
    if X.shape[0] < 1 + input_size + r_out_size:
        r_out_size = X.shape[0]
    else:
        X = X[:1 + input_size + r_out_size]
    output_size = Yt.shape[1]
    
    if args.read_out == 'linear':
        model = SimpleNN(1 + input_size + r_out_size, output_size).to(device).to(dtype)
    elif args.read_out == 'transformer':
        model = SimpleTransformerModel(1 + input_size + r_out_size, output_size).to(device).to(dtype)
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
        output = model(X.T)
        loss = criterion(output, Yt)
        loss.backward()
        optimizer.step()
        if epoch % 100 == 0:
            print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.8f}')
    return model

def run_generative_mode(data, model, Win, W, r_out_size, testLen, trainLen, a, initial_x, args, inSize, outSize, dtype, device):
    Y = torch.zeros((outSize, testLen), dtype=dtype, device=device)
    one = torch.tensor([1.0], dtype=dtype, device=device)
    # Precompute win_padding outside the loop:
    pad_rows = W.shape[0] - Win.shape[0]
    win_padding = torch.zeros((pad_rows, 1), dtype=dtype, device=device)

    u = data[trainLen]
    for t in range(testLen):
        Win_out = Win @ torch.vstack((one, u.view(-1, 1)))
        Win_out = torch.cat([Win_out, win_padding])
        x = (1 - a) * initial_x + a * torch.tanh(Win_out + W @ initial_x)
        # Concatenate one, input u, and reservoir state (truncated to r_out_size)
        input_to_model = torch.vstack((one, u.view(-1, 1), x[:r_out_size])).T.flatten()
        if args.opt == 'lr':
            y = model.T @ input_to_model
        else:
            y = model(input_to_model.view(1, -1)).view(-1)
        Y[:, t] = y
        u = y
        initial_x = x  # update the reservoir state
    return Y

def compute_mse(data, Y, trainLen, errorLen):
    # Handle multi-dimensional targets (e.g., when using Wikipedia embeddings)
    target = data[trainLen + 1: trainLen + errorLen + 1]
    if target.dim() > 1:
        target = target.T  # shape now (embedding_dim, errorLen)
    return torch.mean((target - Y[:, :errorLen]) ** 2).item()

# --------------------------
# Main Function
# --------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data file path
    parser.add_argument('--data-file', type=str, default='../data/MackeyGlass_t17.txt', help='Path to data file')
 
    parser.add_argument('--dataset', type=str, default='timeseries', choices=['timeseries', 'wikipedia'],
                        help='Select dataset type: "timeseries" or "wikipedia"')

    # Training options
    parser.add_argument('--fp', type=int, default=64, choices=[16, 32, 64], help='float precision')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--opt', type=str, default='adam', choices=['adam', 'adamw', 'adagrad', 'rprop', 'rmsprop', 'lr'], help='optimizer')
    parser.add_argument('--epochs', type=int, default=10000, help='number of epochs')

    # Reservoir options
    parser.add_argument('--top', type=str, default='uniform', choices=['uniform', 'geometric', 'smallworld'], help='reservoir topology')
    parser.add_argument('--dim-res', type=int, default=1000, help='reservoir size')
    parser.add_argument('--rho', type=float, default=0.01, help='reservoir density')
    parser.add_argument('--alpha', type=float, default=0.3, help='reservoir leak rate')
    parser.add_argument('--rest', action='store_true', help='enable reservoir spectral radius estimation')

    # Read-out
    parser.add_argument('--read-out', type=str, default='linear', choices=['linear', 'transformer'], help='readout architecture')

    # Valve sizes
    parser.add_argument('--valve-in', type=int, default=1000, help='input valve size')
    parser.add_argument('--valve-out', type=int, default=1000, help='output valve size')

    # Data dimensions
    parser.add_argument('--dim-in', type=int, default=1, help='input size (ignored if --dataset is "wikipedia")')
    parser.add_argument('--dim-out', type=int, default=1, help='output size (ignored if --dataset is "wikipedia")')

    # Visualization
    parser.add_argument('--viz', action='store_true', help='plot reservoir information')

    # Device option
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model (e.g. "cpu" or "cuda")')

    # Enable pytorch profiler option
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiling for reservoir and readout training')

    args = parser.parse_args()
    device = torch.device(args.device)

    torch.manual_seed(42)
    dtype = {64: torch.float64, 32: torch.float32, 16: torch.float16}[args.fp]

    # Load data: load Wikipedia data if dataset is wikipedia, else load timeseries data.
    if args.dataset == "wikipedia":
        print("Loading Wikipedia dataset with pretrained embeddings...")
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        model_emb = AutoModel.from_pretrained("bert-base-uncased").to(device).to(dtype)
        model_emb.eval()
        data = load_wikipedia_data(tokenizer, model_emb, max_length=128, num_samples=1000).to(dtype=dtype, device=device)
        # Set input and output dimensions to match the embedding size (typically 768 for BERT)
        inSize = data.shape[1]
        outSize = data.shape[1]
    else:
        data = load_data(args.data_file, dtype, device)
        inSize = args.dim_in
        outSize = args.dim_out

    resSize = args.dim_res
    a = args.alpha
    reg = 1e-8
    # Adjust trainLen, testLen, etc. For Wikipedia, ensure there are enough samples.
    trainLen = 2000 if args.dataset == "timeseries" else min(500, data.shape[0]-1)
    testLen = 2000 if args.dataset == "timeseries" else min(400, data.shape[0]-1)
    initLen = 100
    errorLen = 500 if args.dataset == "timeseries" else min(100, data.shape[0]-initLen-1)
    learning_rate = args.lr
    epochs = args.epochs
    density = args.rho
    inInterSize = args.valve_in
    outInterSize = args.valve_out

    rho_value = (estimate_rho(resSize, density, args.top, dtype, device, allow_plot=args.viz)
                 if args.rest else None)
    Win, W = initialize_weights_sparse(resSize, inSize, inInterSize, density, args.top, dtype, device, rho=rho_value, allow_plot=args.viz)

    # --------------------------
    # Profile Reservoir Run (if enabled)
    # --------------------------
    if args.profile:
        print("Profiling reservoir run...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True
        ) as prof:
            X, final_x_state = run_reservoir_sparse(data, Win, W, trainLen, initLen, resSize, a, inSize, dtype, device)
            prof.step()
        table = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(table)
        reservoir_trace_file = "reservoir_trace.json"
        prof.export_chrome_trace(reservoir_trace_file)
        print("Reservoir chrome trace exported to", reservoir_trace_file)
    else:
        X, final_x_state = run_reservoir_sparse(data, Win, W, trainLen, initLen, resSize, a, inSize, dtype, device)

    # Prepare target outputs.
    Yt = data[initLen + 1: trainLen + 1].clone().detach().to(dtype=dtype, device=device)
    if Yt.ndim == 1:
        Yt = Yt.unsqueeze(1)
    # For multi-dimensional data (e.g. Wikipedia embeddings), ensure Yt is of shape (trainLen - initLen, outSize)
    if Yt.dim() > 1 and Yt.shape[1] != outSize:
        Yt = Yt.view(-1, outSize)

    # --------------------------
    # Profile Readout Training (if enabled)
    # --------------------------
    if args.profile:
        print("Profiling readout training...")
        with torch.profiler.profile(
            activities=[torch.profiler.ProfilerActivity.CPU,
                        torch.profiler.ProfilerActivity.CUDA] if device.type == "cuda" else [torch.profiler.ProfilerActivity.CPU],
            record_shapes=True,
            with_stack=True
        ) as prof:
            if args.opt == 'lr':
                model = train_output(X, Yt, inSize, outInterSize, reg, dtype, device)
            else:
                model = train_output_with_gd(X, Yt, inSize, outInterSize, learning_rate, epochs, args, dtype, device)
            prof.step()
        table = prof.key_averages().table(sort_by="self_cpu_time_total")
        print(table)
        readout_trace_file = "readout_trace.json"
        prof.export_chrome_trace(readout_trace_file)
        print("Readout chrome trace exported to", readout_trace_file)
    else:
        if args.opt == 'lr':
            model = train_output(X, Yt, inSize, outInterSize, reg, dtype, device)
        else:
            model = train_output_with_gd(X, Yt, inSize, outInterSize, learning_rate, epochs, args, dtype, device)

    Y = run_generative_mode(data, model, Win, W, outInterSize, testLen, trainLen, a, final_x_state,
                            args, inSize, outSize, dtype, device)
    mse = compute_mse(data, Y, trainLen, errorLen)
    print('MSE =', mse)
    if args.viz:
        plot_results(data, Y, X, model, inSize, outInterSize, trainLen, testLen, args)


if __name__ == "__main__":
    main()
