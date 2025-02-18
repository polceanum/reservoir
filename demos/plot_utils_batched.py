import numpy as np
import torch
import matplotlib.pyplot as plt
import networkx as nx

def plot_power_law(x, y, a, b):
    # Generate a range of x values for plotting the power law curve
    x_range = np.linspace(min(x), max(x), 100)
    y_range = a * x_range ** b

    # Plot the original data points
    plt.scatter(x, y, color='blue', label='Original Data Points + extrapolation')
    # Plot the power law curve
    plt.plot(x_range, y_range, color='red', label=f'Estimated Power Law (y = {a:.2f}x^{b:.2f})')
    
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Power Law Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(data, Y, X, model, in_size, r_out_size, trainLen, testLen, args):
    """
    data: Tensor of shape (T, batch, outSize)
    Y: Tensor of shape (batch, testLen, outSize)
    X: Tensor of shape (batch, (trainLen - initLen), feature_dim) where feature_dim = 1 + in_size + resSize
    For plotting we use the first batch element.
    """
    # Convert tensors to numpy arrays using the first batch element.
    data_np = data[:, 0, :].cpu().numpy()      # shape: (T, outSize)
    Y_np = Y[0].detach().cpu().numpy()           # shape: (testLen, outSize)
    X_np = X[0].cpu().numpy()                    # shape: (num_steps, feature_dim)

    # Figure 1: Target and predicted signals
    plt.figure(1)
    plt.clf()
    target_signal = data_np[trainLen+1: trainLen+testLen+1, 0]  # use first output dimension
    predicted_signal = np.clip(Y_np[:, 0], -10, 10)
    plt.plot(target_signal, 'g', label='Target signal')
    plt.plot(predicted_signal, 'b', label='Predicted signal')
    plt.title('Target and Free-running Predicted Signal $y(n)$ starting at $n=0$')
    plt.legend()

    # Figure 2: Reservoir activations
    plt.figure(2)
    plt.clf()
    # Extract reservoir activations (columns after the first 1+in_size entries)
    reservoir_activations = X_np[:, 1+in_size:]
    num_neurons = min(20, reservoir_activations.shape[1])
    activation_subset = reservoir_activations[:200, :num_neurons].T  # shape: (num_neurons, time)
    for neuron_activation in activation_subset:
        plt.plot(neuron_activation)
    plt.title('Some Reservoir Activations $\\mathbf{x}(n)$')

    # Figure 3: Output weights
    plt.figure(3)
    plt.clf()
    if args.opt == 'lr':
        # In the 'lr' case, model is a weight matrix whose shape should be (1+in_size+r_out_size, output_size).
        weights = model.cpu().numpy().squeeze()
        x_range = range(weights.shape[0])
    elif args.read_out == 'linear':
        # For a linear read-out architecture, plot the linear layer's weights.
        weights = model.linear.weight.detach().cpu().numpy().squeeze()
        x_range = range(weights.shape[0])
    elif args.read_out == 'transformer':
        # For transformer read-out, we plot the final fc2 layer's weights.
        weights = model.fc2.weight.detach().cpu().numpy().squeeze()
        # weights shape is (output_size, hidden_size) and if output_size==1, then shape becomes (hidden_size,)
        x_range = range(weights.shape[0])
    else:
        weights = None
        x_range = None

    if weights is not None and x_range is not None:
        plt.bar(x_range, weights)
        plt.title('Output Weights')
    else:
        plt.text(0.5, 0.5, 'No output weights to display', 
                 horizontalalignment='center', verticalalignment='center')

    plt.show()

def plot_geometric_graph(G):
    # Position is stored as node attribute "pos"
    pos = nx.get_node_attributes(G, "pos")
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
        if d < dmin:
            ncenter = n
            dmin = d

    # Color nodes by shortest-path length from ncenter.
    p = dict(nx.single_source_shortest_path_length(G, ncenter))

    plt.figure(figsize=(8, 8))
    nx.draw_networkx_edges(G, pos, alpha=0.4)
    nx.draw_networkx_nodes(
        G,
        pos,
        nodelist=list(p.keys()),
        node_size=10,
        node_color=list(p.values()),
        cmap=plt.cm.Reds_r,
    )

    plt.xlim(-0.05, 1.05)
    plt.ylim(-0.05, 1.05)
    plt.axis("off")
    plt.show()
