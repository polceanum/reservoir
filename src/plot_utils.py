import numpy as np
import torch
from matplotlib.pyplot import *
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

    # Additional plot settings
    plt.xlabel('x')
    plt.ylabel('y')
    plt.title('Power Law Estimation')
    plt.legend()
    plt.grid(True)
    plt.show()

def plot_results(data, Y, X, model, in_size, r_out_size, trainLen, testLen, args):
    plt.figure(1)
    plt.clf()
    
    # Process target signal: if multidimensional, use the first column
    target = data[trainLen + 1:trainLen + testLen + 1].numpy()
    if target.ndim > 1:
        target = target[:, 0]
    plt.plot(target, 'g', label='Target signal')
    
    # Process predicted signal: if multidimensional, use the first column
    pred = torch.clip(Y.T.detach(), -10, 10).numpy()
    note = ''
    if pred.ndim > 1:
        num_cols = pred.shape[1]
        pred = pred[:, 0]
        note = ' (only first column shown out of {})'.format(num_cols)
    plt.plot(pred, 'b', label='Free-running predicted signal')
    
    plt.title('Target and generated signals $y(n)$ starting at $n=0$'+note)
    plt.legend()
    
    plt.figure(2)
    plt.clf()
    plt.plot(X[0:20, 0:200].T.numpy())
    plt.title('Some reservoir activations $\\mathbf{x}(n)$')
    
    plt.figure(3)
    plt.clf()
    if args.opt == 'lr':
        w = model.numpy().squeeze()
        plt.bar(range(len(w)), w)
        plt.title('Output weights $\\mathbf{W}^{out}$')
    elif args.read_out == 'linear':
        w = model.linear.weight.detach().numpy()
        if w.ndim > 1:
            w = np.mean(w, axis=0)  # average over output neurons if necessary
        else:
            w = w.squeeze()
        plt.bar(range(len(w)), w)
        plt.title('Output weights $\\mathbf{W}^{out}$')
    
    plt.show()




def plot_geometric_graph(G):
    # position is stored as node attribute data for random_geometric_graph
    pos = nx.get_node_attributes(G, "pos")

    # find node near center (0.5,0.5)
    dmin = 1
    ncenter = 0
    for n in pos:
        x, y = pos[n]
        d = (x - 0.5) ** 2 + (y - 0.5) ** 2
        if d < dmin:
            ncenter = n
            dmin = d

    # color by path length from node near center
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