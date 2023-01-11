# -*- coding: utf-8 -*-
"""
A minimalistic Echo State Networks demo with Mackey-Glass (delay 17) data 
in "plain" scientific Python.
by Mantas LukoÅ¡eviÄ?ius 2012
http://minds.jacobs-university.de/mantas
"""
from numpy import *
from matplotlib.pyplot import *
import scipy.linalg
import networkx as nx
from pyvis.network import Network
from scipy import sparse
from scipy import stats

# load the data
trainLen = 2000
testLen = 2000
initLen = 100

data = loadtxt('MackeyGlass_t17.txt')

# plot some of it
figure(10).clear()
plot(data[0:1000])
title('A sample of data')

# generate the ESN reservoir
inSize = outSize = 1
resSize = 2000
a = 0.3 # leaking rate
k_connectivity = 3
sparsity = 1-k_connectivity/resSize
out_size = 1000

random.seed(42)
Win = (random.rand(resSize,1+inSize)-0.5) * 1
# W = random.rand(resSize,resSize)-0.5 
# # sparse_mask = random.rand(resSize,resSize)
# # sparse_mask[sparse_mask<sparsity] = 0
# # sparse_mask[sparse_mask>0] = 1

# # G = nx.thresholded_random_geometric_graph(resSize, 0.2, 0.5)
# # G = nx.random_geometric_graph(resSize, 0.05)
# G = nx.generators.geometric.soft_random_geometric_graph(resSize, radius=0.1)
# sparse_mask = np.array(nx.adjacency_matrix(G).todense(), dtype=float32)
# sparse_mask[sparse_mask==0] = 0
# sparse_mask[sparse_mask!=0] = 1

# # for breaking undirectedness
# sparse_mask_2 = random.rand(resSize,resSize)
# sparse_mask_2[sparse_mask_2<0.5] = 0
# sparse_mask_2[sparse_mask_2>0] = 1

# W *= sparse_mask*sparse_mask_2
# ##############
# # G = nx.thresholded_random_geometric_graph(resSize, 0.03, 0.5)
# # W = np.array(nx.adjacency_matrix(G).todense(), dtype=float32)

#W = sparse.random(resSize, resSize, density=0.01, data_rvs=stats.uniform(loc=-0.5, scale=1.0))

density = 0.01
non_zeros = int(resSize * resSize * density)
print('non zeros', non_zeros)
rows = np.random.randint(0, resSize, size=non_zeros)
cols = np.random.randint(0, resSize, size=non_zeros)
data = np.random.rand(non_zeros)-0.5
print(data.mean(), data.std())
W = sparse.coo_matrix((data, (rows, cols)), shape=(resSize, resSize)).tocsr()


# ------------------- #
# figure(4).clear()
# print('plotting graph')
# pos = nx.get_node_attributes(G, "pos")
# nx.draw_networkx(G, pos=pos, with_labels=False, node_size=2, arrowsize=5, edge_color=(0.0, 0.0, 0.0, 0.05))
# net = Network(notebook=True)
# net.from_nx(G)
# net.show_buttons(filter_=['physics'])
# net.show('example.html')
# exit()

# ------------------- #

# Option 1 - direct scaling (quick&dirty, reservoir-specific):
#W *= 0.135 
# Option 2 - normalizing and setting spectral radius (correct, slow):
print('Computing spectral radius...')
# rhoW = max(abs(linalg.eig(W)[0]))
rhoW = max(abs(sparse.linalg.eigs(W, k=1)[0]))
# rhoW = 5.785 #0.92 #4.600 #3.751 #2.666
print('done. rhoW =', rhoW)
W *= 1.25 / rhoW

# allocated memory for the design (collected states) matrix
X = zeros((1+inSize+resSize,trainLen-initLen))
# set the corresponding target matrix directly
Yt = data[None,initLen+1:trainLen+1] 

# run the reservoir with the data and collect X
x = zeros((resSize,1))
for t in range(trainLen):
    u = data[t]
    # x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + W.dot( x ) )
    if t >= initLen:
        X[:,t-initLen] = vstack((1,u,x))[:,0]
    
# train the output
reg = 1e-8  # regularization coefficient
X = X[:out_size, :]
X_T = X.T
# print(X_T.shape)
# exit()
Wout = dot( dot(Yt,X_T), linalg.inv( dot(X,X_T) + \
    reg*eye(out_size) ) ) #*eye(1+inSize+resSize) ) )
#Wout = dot( Yt, linalg.pinv(X) )

# run the trained ESN in a generative mode. no need to initialize here, 
# because x is initialized with training data and we continue from there.
Y = zeros((outSize,testLen))
u = data[trainLen]
for t in range(testLen):
    # x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + dot( W, x ) )
    x = (1-a)*x + a*tanh( dot( Win, vstack((1,u)) ) + W.dot( x ) )
    y = dot( Wout, vstack((1,u,x))[:out_size, :] )
    Y[:,t] = y
    # generative mode:
    u = y
    ## this would be a predictive mode:
    #u = data[trainLen+t+1] 

# compute MSE for the first errorLen time steps
errorLen = 500
mse = sum( square( data[trainLen+1:trainLen+errorLen+1] - Y[0,0:errorLen] ) ) / errorLen
print('MSE = ' + str( mse ))
    
# plot some signals
figure(1).clear()
plot( data[trainLen+1:trainLen+testLen+1], 'g' )
plot( Y.T, 'b' )
title('Target and generated signals $y(n)$ starting at $n=0$')
legend(['Target signal', 'Free-running predicted signal'])

figure(2).clear()
plot( X[0:20,0:200].T )
title('Some reservoir activations $\mathbf{x}(n)$')

figure(3).clear()
bar( range(out_size), Wout.T.squeeze(1) )
title('Output weights $\mathbf{W}^{out}$')

show()