import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import networkx as nx

# --------------------------
# Sparse ESN reservoir init
# --------------------------
def initialize_reservoir(resSize, density, topology, dtype, device):
    if topology == 'uniform':
        num_elements = int(resSize * resSize * density)
        idx = torch.randint(0, resSize, (2, num_elements), device=device)
        val = (torch.rand(num_elements, dtype=dtype, device=device) - 0.5)
        W = torch.sparse_coo_tensor(idx, val, (resSize, resSize), dtype=dtype, device=device).coalesce()
    elif topology in ['geometric', 'smallworld']:
        if topology == 'geometric':
            G = nx.thresholded_random_geometric_graph(resSize, 0.05, 0.0001)
        else:
            G = nx.navigable_small_world_graph(resSize, p=3)
        A = nx.to_scipy_sparse_array(G, format='coo')
        idx = torch.tensor(np.vstack([A.row, A.col]), dtype=torch.long, device=device)
        val = torch.tensor(A.data, dtype=dtype, device=device)
        W = torch.sparse_coo_tensor(idx, val, torch.Size(A.shape), dtype=dtype, device=device).coalesce()
    else:
        raise ValueError(f"Unknown topology: {topology}")
    return W.to_sparse_csr()

@torch.no_grad()
def power_iteration_spectral_radius(W_csr, iters=50):
    """
    Estimate spectral radius (|lambda_max|) via power iteration on sparse CSR.
    W_csr: (N,N) sparse_csr_tensor
    """
    N = W_csr.size(0)
    x = torch.randn(N, 1, device=W_csr.device, dtype=W_csr.dtype)
    x = x / (x.norm() + 1e-12)
    for _ in range(iters):
        x = torch.sparse.mm(W_csr, x)
        n = x.norm()
        if n < 1e-20:
            break
        x = x / n
    # Rayleigh quotient gives the eigenvalue estimate
    Wx = torch.sparse.mm(W_csr, x)
    num = (x.T @ Wx).abs().item()
    den = (x.T @ x).abs().item() + 1e-12
    return num / den

def initialize_weights_sparse(resSize, inSize, density, topology, dtype, device, rho=None):
    # Input weights: map [1; u_t] directly into resSize (no padding later)
    Win = torch.empty(resSize, 1 + inSize, dtype=dtype, device=device).uniform_(-0.5, 0.5)
    W_sparse = initialize_reservoir(resSize, density, topology, dtype, device)

    if rho is None:
        rhoW = power_iteration_spectral_radius(W_sparse, iters=50)
    else:
        rhoW = float(rho)

    # Scale to echo state regime (you can tune 1.25 factor)
    scale = 1.25 / (rhoW + 1e-12)
    W_sparse = W_sparse * scale
    return Win, W_sparse

# --------------------------
# Reservoir-Attention Layer
# --------------------------
class ReservoirAttention(nn.Module):
    """
    Forward signature matches your original:
      query: (seq_len, batch, embed_dim)
      reservoir_state: (batch, resSize, 1)
    Returns:
      attn_output: (seq_len, batch, embed_dim)
      new_state:   (batch, resSize, 1)
    """
    def __init__(self, embed_dim, num_heads, resSize, inSize,
                 density=0.01, topology='uniform',
                 a=0.3, rho=None, dtype=torch.float32, device='cpu'):
        super().__init__()
        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.resSize = resSize
        self.inSize = inSize
        self.a = a

        # Optional input projection to inSize
        self.input_proj = None
        if embed_dim != inSize:
            self.input_proj = nn.Linear(embed_dim, inSize, bias=False, device=device, dtype=dtype)

        # ESN weights (non-trainable)
        Win, W_sparse = initialize_weights_sparse(
            resSize, inSize, density, topology, dtype, device, rho
        )
        self.register_buffer("Win", Win, persistent=False)
        self.register_buffer("W_sparse", W_sparse, persistent=False)

        # Attention projections:
        #   - Query comes from input (dynamic)
        #   - Keys are learned positional embeddings per reservoir position (static per step)
        #   - Values are state-weighted positional embeddings (dynamic)
        self.query_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

        # Learned positional embeddings for keys/values per head
        # Shape: (resSize, num_heads, head_dim)
        self.Ek = nn.Parameter(torch.empty(resSize, num_heads, self.head_dim, dtype=dtype, device=device))
        self.Ev = nn.Parameter(torch.empty(resSize, num_heads, self.head_dim, dtype=dtype, device=device))
        nn.init.xavier_uniform_(self.Ek.view(resSize, -1))
        nn.init.xavier_uniform_(self.Ev.view(resSize, -1))

        # Optional output projection (identity by reshape is fine; add this if you want a learned mix)
        # self.out_proj = nn.Linear(embed_dim, embed_dim, bias=False, device=device, dtype=dtype)

    def _esn_step(self, u_t, state):
        """
        u_t: (batch, embed_dim) before optional proj
        state: (batch, resSize, 1)
        """
        if self.input_proj is not None:
            u_t = self.input_proj(u_t)  # (batch, inSize)

        # Build [1; u_t] -> (batch, 1+inSize)
        cat = torch.cat([torch.ones(u_t.size(0), 1, device=u_t.device, dtype=u_t.dtype), u_t], dim=1)
        # Win @ [1; u_t]: (resSize, 1+inSize) @ (batch, 1+inSize)^T
        # -> (resSize, batch) -> (batch, resSize)
        Win_u = torch.matmul(cat, self.Win.T)  # (batch, resSize)

        # Sparse reservoir: for each batch, W_sparse @ state_b
        # state_b: (resSize, 1) -> result (resSize, 1)
        bsz = state.size(0)
        Wx = []
        s_flat = state.view(bsz, self.resSize, 1)
        for b in range(bsz):
            Wx_b = torch.sparse.mm(self.W_sparse, s_flat[b])  # (resSize, 1)
            Wx.append(Wx_b.squeeze(-1))
        Wx = torch.stack(Wx, dim=0)  # (batch, resSize)

        pre = Win_u + Wx  # (batch, resSize)
        new_state = (1.0 - self.a) * state.squeeze(-1) + self.a * torch.tanh(pre)
        return new_state.unsqueeze(-1)  # (batch, resSize, 1)

    def _attend(self, q_t, state):
        """
        q_t: (batch, embed_dim)
        state: (batch, resSize, 1)
        Multi-head attention over reservoir positions:
          - Keys: learned per-position embeddings (static)
          - Values: state-weighted per-position embeddings (dynamic)
        """
        bsz = q_t.size(0)

        # Q: (batch, num_heads, head_dim)
        Q = self.query_proj(q_t).view(bsz, self.num_heads, self.head_dim)

        # K: (batch, num_heads, resSize, head_dim)  [broadcast learned Ek]
        K = self.Ek.permute(1, 0, 2).unsqueeze(0).expand(bsz, -1, -1, -1)

        # V: (batch, num_heads, resSize, head_dim)  [state-weighted Ev]
        # state: (batch, resSize, 1) -> (batch, 1, resSize, 1)
        S = state.permute(0, 2, 1)  # (batch, 1, resSize)
        Ev = self.Ev.permute(1, 0, 2).unsqueeze(0)  # (1, num_heads, resSize, head_dim)
        V = S.unsqueeze(-1) * Ev  # (batch, num_heads, resSize, head_dim)

        # Scores: (batch, num_heads, resSize)
        # Q · K^T across head_dim
        scores = (Q.unsqueeze(2) * K).sum(-1) / (self.head_dim ** 0.5)
        attn = torch.softmax(scores, dim=-1)

        # Context: sum over positions -> (batch, num_heads, head_dim)
        context = (attn.unsqueeze(-1) * V).sum(dim=2)

        # Merge heads -> (batch, embed_dim)
        out = context.reshape(bsz, self.embed_dim)
        # If you want an extra learned mix:
        # out = self.out_proj(out)
        return out

    def forward(self, query, reservoir_state):
        """
        query: (seq_len, batch, embed_dim)
        reservoir_state: (batch, resSize, 1)
        """
        seq_len, bsz, _ = query.size()
        outputs = []
        state = reservoir_state

        for t in range(seq_len):
            q_t = query[t]  # (batch, embed_dim)

            # ESN step
            state = self._esn_step(q_t, state)  # (batch, resSize, 1)

            # Attention over reservoir positions
            y_t = self._attend(q_t, state)  # (batch, embed_dim)

            outputs.append(y_t.unsqueeze(0))  # (1, batch, embed)

        attn_output = torch.cat(outputs, dim=0)  # (seq_len, batch, embed_dim)
        return attn_output, state

# --------------------------
# Demo
# --------------------------
if __name__ == "__main__":
    torch.manual_seed(42)
    device = "cpu"
    dtype = torch.float32

    embed_dim = 16
    inSize = 8  # intentionally different to test input_proj
    num_heads = 4
    seq_len = 5
    batch_size = 2
    resSize = 32

    attn = ReservoirAttention(embed_dim, num_heads, resSize, inSize,
                              density=0.05, topology='uniform',
                              a=0.3, dtype=dtype, device=device)

    reservoir_state = torch.zeros((batch_size, resSize, 1), dtype=dtype, device=device)
    x = torch.randn(seq_len, batch_size, embed_dim, dtype=dtype, device=device)

    out, new_state = attn(x, reservoir_state)
    print("Output shape:", out.shape)           # (seq_len, batch, embed_dim)
    print("New reservoir state shape:", new_state.shape)  # (batch, resSize, 1)
