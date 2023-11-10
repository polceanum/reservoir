import jax
import jax.numpy as jnp
from jax import random
from jax.experimental import sparse
# from jax.scipy.linalg import eigh
from jax.scipy.sparse.linalg import cg
from matplotlib.pyplot import *
import numpy as np

jax.config.update('jax_enable_x64', True)
# jax.config.update('jax_platform_name', 'cpu')
# jax.config.update('jax_disable_jit', True)

# Load the data
data = np.loadtxt('../data/MackeyGlass_t17.txt').astype(np.float64)
data = jnp.array(data)

# Initialize the random key
key = random.PRNGKey(42)

# Parameters
trainLen = 2000
testLen = 2000
initLen = 100
inSize = outSize = 1
resSize = 1000
a = 0.3  # leaking rate
reg = 1e-8  # regularization coefficient
density = 1.0  # Density for the sparse matrix

# Initialize input weights
key, subkey = random.split(key)
Win = (random.uniform(subkey, (resSize, 1 + inSize)) - 0.5) * 1

# Initialize reservoir weights as sparse matrix
key, subkey = random.split(key)
W_dense = random.uniform(subkey, (resSize, resSize)) - 0.5
W_sparse = sparse.BCOO.fromdense(W_dense, nse=int(resSize**2 * density))
# W_sparse = jnp.array(W_dense)
# rhoW = max(abs(eigh(W_dense, eigvals_only=True)))
rhoW = np.max(np.abs(jnp.linalg.eig(W_dense)[0]))
W_sparse = W_sparse * (1.25 / rhoW)

# Allocate memory for the design matrix
X = jnp.zeros((1 + inSize + resSize, trainLen - initLen))
Yt = data[None, initLen + 1:trainLen + 1].T

# Run the reservoir
x = jnp.zeros((resSize, 1))
for t in range(trainLen):
    u = data[t]
    x = (1 - a) * x + a * jnp.tanh(Win @ jnp.vstack((1, u)) + W_sparse @ x)
    if t >= initLen:
        X = X.at[:, t - initLen].set(jnp.vstack((1, u, x))[:, 0])

# Train the output
identity_matrix = jnp.eye(1 + inSize + resSize)
X_T = X.T
# Wout = np.dot( np.dot(Yt,X_T), np.linalg.inv( np.dot(X,X_T) + reg*np.eye(1+inSize+resSize) ) )
Wout = jnp.linalg.solve(X @ X_T + reg * identity_matrix, X @ Yt)

# Run in generative mode
Y = jnp.zeros((outSize, testLen))
u = data[trainLen]
for t in range(testLen):
    x = (1 - a) * x + a * jnp.tanh(Win @ jnp.vstack((1, u)) + W_sparse @ x)
    y = Wout.T @ jnp.vstack((1, u, x))
    Y = Y.at[:, t].set(y[0])
    u = y[0]


# Compute MSE
errorLen = 500
mse = jnp.mean(jnp.square(data[trainLen + 1:trainLen + errorLen + 1] - Y[0, 0:errorLen]))
print('MSE = ' + str(mse))

# Plotting
figure(1).clear()
plot(data[trainLen + 1:trainLen + testLen + 1], 'g')
plot(Y.T, 'b')
title('Target and generated signals $y(n)$ starting at $n=0$')
legend(['Target signal', 'Free-running predicted signal'])

figure(2).clear()
plot(X[0:20, 0:200].T)
title('Some reservoir activations $\\mathbf{x}(n)$')

figure(3).clear()
bar(range(1 + inSize + resSize), Wout.T.squeeze())
title('Output weights $\\mathbf{W}^{out}$')

show()
