import networkx as nx
import matplotlib.pyplot as plt

G = nx.cycle_graph(10)
A = nx.adjacency_matrix(G)*0.5
print(A.todense())

G_numpy = nx.from_numpy_array(A.todense())

nx.draw_networkx(G_numpy, with_labels=False, node_size=10, arrowsize=2)
plt.show()