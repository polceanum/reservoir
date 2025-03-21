import argparse
import os
import random
import torch
import torch.nn as nn
import torch.profiler
import numpy as np
from sklearn.linear_model import LinearRegression
import networkx as nx
from tqdm import tqdm

from plot_utils import plot_results, plot_power_law, plot_geometric_graph

# If using a text dataset, we need Hugging Face transformers:
try:
    from transformers import AutoTokenizer, AutoModel
except ImportError:
    print("Transformers not installed. If you want to use the text dataset option, please install transformers.")

# --------------------------
# Automatic Download Function
# --------------------------
def download_wikipedia_sample(file_path):
    """
    Downloads a sample text file using the WikiText-2 dataset.
    Each non-empty line in the file will be a sample.
    """
    try:
        from datasets import load_dataset
    except ImportError:
        print("Hugging Face 'datasets' library not installed. Please install it via 'pip install datasets'")
        exit(1)
    print("Downloading Wikipedia sample dataset using WikiText-2...")
    dataset = load_dataset("wikitext", "wikitext-2-raw-v1", split="train")
    with open(file_path, "w", encoding="utf-8") as f:
        for example in dataset:
            text = example["text"].strip()
            if text:
                f.write(text + "\n")
    print(f"Downloaded sample file to {file_path}")

# --------------------------
# Caching Helpers
# --------------------------
def load_or_cache_text_data(file_path, dtype, device, tokenizer, model):
    """
    Loads text data and computes embeddings. If a cached version exists (file_path + ".embeddings.pt"),
    then it is loaded instead.
    """
    cache_file = file_path + ".embeddings.pt"
    if os.path.exists(cache_file):
        print("Loading cached text embeddings from", cache_file)
        data = torch.load(cache_file, map_location=device)
    else:
        data = load_text_data(file_path, dtype, device, tokenizer, model)
        torch.save(data, cache_file)
        print("Cached text embeddings saved to", cache_file)
    return data

def get_cached_weights(resSize, inSize, inInterSize, density, topology, dtype, device, rho, allow_plot, cache_dir="cache"):
    """
    Checks for cached reservoir weights. If found, loads and returns them.
    Otherwise, computes the weights using initialize_weights_sparse and caches them.
    """
    os.makedirs(cache_dir, exist_ok=True)
    filename = os.path.join(cache_dir, f"weights_res{resSize}_in{inSize}_inInter{inInterSize}_density{density}_{topology}.pt")
    if os.path.exists(filename):
        print("Loading cached reservoir weights from", filename)
        cache = torch.load(filename, map_location=device)
        Win = cache["Win"]
        W = cache["W"]
    else:
        Win, W = initialize_weights_sparse(resSize, inSize, inInterSize, density, topology, dtype, device, rho=rho, allow_plot=allow_plot)
        torch.save({"Win": Win, "W": W}, filename)
        print("Cached reservoir weights saved to", filename)
    return Win, W

# --------------------------
# Additional Decoding Helpers
# --------------------------
def load_text_lines(file_path):
    """Load raw text lines from a file."""
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f if line.strip()]
    return lines

def decode_embedding(embedding, embeddings, texts):
    """
    Given a predicted embedding, finds the nearest neighbor (cosine similarity) in embeddings,
    and returns the corresponding text from texts.
    """
    embedding_norm = embedding / embedding.norm()
    embeddings_norm = embeddings / embeddings.norm(dim=1, keepdim=True)
    similarity = torch.matmul(embeddings_norm, embedding_norm)
    idx = torch.argmax(similarity).item()
    return texts[idx]

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
        self.transformer_decoder_layer = nn.TransformerDecoderLayer(d_model=hidden_size, nhead=num_heads, batch_first=True)
        self.transformer_decoder = nn.TransformerDecoder(self.transformer_decoder_layer, num_layers=num_layers)
        self.fc2 = nn.Linear(hidden_size, output_size, bias=True)
        
    def forward(self, x):
        x = self.fc1(x)
        x = x.unsqueeze(1)  # shape: (batch, 1, hidden_size)
        x = self.transformer_decoder(x, x)
        x = x.squeeze(1)
        return self.fc2(x)

# --------------------------
# Data and Helper Functions
# --------------------------
def load_data(file_path, dtype, device):
    data = np.loadtxt(file_path)
    return torch.tensor(data, dtype=dtype, device=device)

def load_text_data(file_path, dtype, device, tokenizer, model):
    """
    Loads a text file where each line is a document/sentence,
    computes embeddings using a pretrained model, and returns a tensor of embeddings.
    """
    embeddings = []
    with open(file_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()
    for line in tqdm(lines, desc="Embedding texts"):
        line = line.strip()
        if not line:
            continue
        encoded_input = tokenizer(line, return_tensors='pt', truncation=True, padding=True)
        for key in encoded_input:
            encoded_input[key] = encoded_input[key].to(device)
        with torch.no_grad():
            output = model(**encoded_input)
        emb = output.last_hidden_state[:, 0, :]  # shape: (1, embedding_dim)
        embeddings.append(emb.squeeze(0))
    data = torch.stack(embeddings)
    return data.to(dtype=dtype, device=device)

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
    one = torch.tensor([1.0], dtype=dtype, device=device)
    for t in tqdm(range(trainLen), desc="Running reservoir"):
        u = data[t]
        Win_out = Win @ torch.vstack((one, u.view(-1, 1)))  # Shape: (inInterSize, 1)
        # Pad Win_out to match reservoir size: pad with zeros in a single column.
        if resSize > Win.shape[0]:
            pad = torch.zeros((resSize - Win.shape[0], 1), dtype=dtype, device=device)
            Win_out = torch.cat([Win_out, pad], dim=0)
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
    
    if args.opt == 'lr':
        model = SimpleNN(1 + input_size + r_out_size, output_size).to(device).to(dtype)
    else:
        model = SimpleTransformerModel(1 + input_size + r_out_size, output_size).to(device).to(dtype)
    
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
    pad_rows = W.shape[0] - Win.shape[0]
    win_padding = torch.zeros((pad_rows, 1), dtype=dtype, device=device) if pad_rows > 0 else None

    u = data[trainLen]
    x = initial_x
    for t in range(testLen):
        Win_out = Win @ torch.vstack((one, u.view(-1, 1)))
        if win_padding is not None:
            Win_out = torch.cat([Win_out, win_padding], dim=0)
        x = (1 - a) * x + a * torch.tanh(Win_out + W @ x)
        input_to_model = torch.vstack((one, u.view(-1, 1), x[:r_out_size])).T.flatten()
        if args.opt == 'lr':
            y = model.T @ input_to_model
        else:
            y = model(input_to_model.view(1, -1)).view(-1)
        Y[:, t] = y
        u = y
    return Y

def compute_mse(data, Y, trainLen, errorLen):
    target = data[trainLen + 1:trainLen + errorLen + 1]
    if target.ndim > 1:
        target = target.transpose(0, 1)
    else:
        target = target.view(1, -1)
    return torch.mean((target - Y[:, :errorLen]) ** 2).item()

# --------------------------
# Main Function
# --------------------------
def main():
    parser = argparse.ArgumentParser()

    # Data file path and dataset type
    parser.add_argument('--data-file', type=str, default='./data/wikipedia_sample.txt', help='Path to data file')
    parser.add_argument('--dataset', type=str, default='timeseries', choices=['timeseries', 'wikipedia'], help='Dataset type to train on')
    parser.add_argument('--pretrained-model', type=str, default='distilbert-base-uncased', help='Pretrained model name for text embeddings (if using a text dataset)')

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
    parser.add_argument('--dim-in', type=int, default=1, help='input size (if using numerical data, or set to embedding dimension for text)')
    parser.add_argument('--dim-out', type=int, default=1, help='output size (if using numerical data, or set to embedding dimension for text)')

    # Visualization
    parser.add_argument('--viz', action='store_true', help='plot reservoir information')

    # Device option
    parser.add_argument('--device', type=str, default='cpu', help='Device to run the model (e.g. "cpu" or "cuda")')

    # Enable pytorch profiler option
    parser.add_argument('--profile', action='store_true', help='Enable PyTorch profiling for reservoir and readout training')

    # Print sample outputs (works for both timeseries and language modes)
    parser.add_argument('--print-samples', action='store_true', help='Print a few sample outputs')

    # Save/load model options.
    parser.add_argument('--save-model', action='store_true', help='Save trained model to file')
    parser.add_argument('--model-save-path', type=str, default='trained_model.pt', help='Path to save or load trained model')
    parser.add_argument('--load-model', action='store_true', help='Load trained model from file (skip training)')

    args = parser.parse_args()
    device = torch.device(args.device)

    # Fix seeds for reproducibility.
    torch.manual_seed(42)
    random.seed(42)
    np.random.seed(42)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(42)

    dtype = {64: torch.float64, 32: torch.float32, 16: torch.float16}[args.fp]

    # Load data based on dataset type.
    if args.dataset == 'wikipedia':
        if not os.path.exists(args.data_file):
            download_wikipedia_sample(args.data_file)
        tokenizer = AutoTokenizer.from_pretrained(args.pretrained_model)
        try:
            model_pt = AutoModel.from_pretrained(args.pretrained_model)
        except Exception as e:
            print("Error downloading pretrained model:", e)
            exit(1)
        model_pt.eval()
        model_pt.to(device)
        data = load_or_cache_text_data(args.data_file, dtype, device, tokenizer, model_pt)
        embedding_dim = data.shape[1]
        args.dim_in = embedding_dim
        args.dim_out = embedding_dim
        print(f"Loaded text data with {data.shape[0]} samples, each of dimension {embedding_dim}.")
    else:
        data = load_data(args.data_file, dtype, device)
        # Ensure timeseries data is 2D (Nx1) for consistency.
        if data.ndim == 1:
            data = data.unsqueeze(1)

    inSize = args.dim_in
    outSize = args.dim_out
    resSize = args.dim_res
    a = args.alpha
    reg = 1e-8
    total_samples = data.shape[0]
    trainLen = min(2000, total_samples // 2)
    testLen = min(2000, total_samples - trainLen)
    initLen = 100
    errorLen = 500
    learning_rate = args.lr
    epochs = args.epochs
    density = args.rho
    inInterSize = args.valve_in
    outInterSize = args.valve_out

    if args.rest:
        rho_value = estimate_rho(resSize, density, args.top, dtype, device, allow_plot=args.viz)
    else:
        rho_value = None

    # Always cache reservoir weights.
    Win, W = get_cached_weights(resSize, inSize, inInterSize, density, args.top, dtype, device, rho_value, allow_plot=args.viz)

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

    Yt = data[initLen + 1: trainLen + 1].clone().detach().to(dtype=dtype, device=device)

    if args.load_model:
        print("Loading trained model from", args.model_save_path)
        saved = torch.load(args.model_save_path, map_location=device)
        if isinstance(saved, dict) and 'state_dict' in saved:
            state = saved['state_dict']
            if 'opt' in saved:
                readout_type = saved['opt']
            else:
                if "linear.weight" in state:
                    readout_type = "lr"
                else:
                    readout_type = "transformer"
            if readout_type == 'lr':
                model = SimpleNN(1 + inSize + outInterSize, outSize).to(device).to(dtype)
            else:
                model = SimpleTransformerModel(1 + inSize + outInterSize, outSize).to(device).to(dtype)
            model.load_state_dict(state)
        else:
            if "linear.weight" in saved:
                model = SimpleNN(1 + inSize + outInterSize, outSize).to(device).to(dtype)
            else:
                model = SimpleTransformerModel(1 + inSize + outInterSize, outSize).to(device).to(dtype)
            model.load_state_dict(saved)
    else:
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
        if args.save_model:
            torch.save({'state_dict': model.state_dict(), 'opt': args.opt}, args.model_save_path)
            print("Trained model saved to", args.model_save_path)

    Y = run_generative_mode(data, model, Win, W, outInterSize, testLen, trainLen, a, final_x_state,
                            args, inSize, outSize, dtype, device)
    mse = compute_mse(data, Y, trainLen, errorLen)
    print('MSE =', mse)

    if args.print_samples:
        print("\nSample Outputs:")
        if args.dataset == 'wikipedia':
            text_lines = load_text_lines(args.data_file)
            num_samples_to_print = min(5, Y.shape[1])
            for i in range(num_samples_to_print):
                pred_embedding = Y[:, i]
                decoded_text = decode_embedding(pred_embedding, data, text_lines)
                print(f"Sample {i+1}: {decoded_text}")
        else:
            num_samples_to_print = min(5, Y.shape[1])
            for i in range(num_samples_to_print):
                sample_output = Y[:, i].cpu().numpy()
                print(f"Sample {i+1}: {sample_output}")

    if args.viz:
        plot_results(data, Y, X, model, inSize, outInterSize, trainLen, testLen, args)

if __name__ == "__main__":
    main()

