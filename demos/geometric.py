import matplotlib.pyplot as plt
import networkx as nx
from pyvis.network import Network

G = nx.thresholded_random_geometric_graph(1000, 0.05, 0.5)
# position is stored as node attribute data for random_geometric_graph
pos = nx.get_node_attributes(G, "pos")

print('done generation')

plt.figure(figsize=(8, 8))
nx.draw_networkx_edges(G, pos, alpha=0.4)
nx.draw_networkx_nodes(G, pos, node_size=5)

net = Network(notebook=True)
net.from_nx(G)
net.show_buttons(filter_=['physics'])
net.show('example.html')

plt.xlim(-0.05, 1.05)
plt.ylim(-0.05, 1.05)
plt.axis("off")
plt.show()