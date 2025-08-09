import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
import os

# Make src importable
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from reservoir_attention import ReservoirAttention

def set_reproducible(seed=0, num_threads=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(num_threads)

@torch.no_grad()
def time_full_seq_mha(mha, x, warmup=2, runs=5):
    if x.device.type == "cuda": torch.cuda.synchronize()
    for _ in range(warmup): _ = mha(x, x, x)[0]
    if x.device.type == "cuda": torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs): _ = mha(x, x, x)[0]
    if x.device.type == "cuda": torch.cuda.synchronize()
    end = time.time()
    return (end - start) / runs

@torch.no_grad()
def time_full_seq_reservoir(ra, x, reservoir_state, warmup=2, runs=5):
    if x.device.type == "cuda": torch.cuda.synchronize()
    for _ in range(warmup): _ = ra(x, reservoir_state.clone())[0]
    if x.device.type == "cuda": torch.cuda.synchronize()
    start = time.time()
    for _ in range(runs): _ = ra(x, reservoir_state.clone())[0]
    if x.device.type == "cuda": torch.cuda.synchronize()
    end = time.time()
    return (end - start) / runs

@torch.no_grad()
def stream_mha_prefix_outputs(mha, x):
    """Return per-step outputs for MHA (prefix recompute). Shape: (L,B,E)."""
    outs = []
    for t in range(1, x.shape[0] + 1):
        y_t = mha(x[:t], x[:t], x[:t])[0][-1]  # (B,E)
        outs.append(y_t.unsqueeze(0))
    return torch.cat(outs, dim=0)

def _project_qkv_from_mha(mha: nn.MultiheadAttention, x_step: torch.Tensor):
    E = mha.embed_dim
    H = mha.num_heads
    Dh = E // H
    W = mha.in_proj_weight
    b = mha.in_proj_bias
    proj = F.linear(x_step, W, b)  # (B,3E)
    q, k, v = proj.split(E, dim=-1)
    def reshape(t): return t.view(t.size(0), H, 1, Dh)
    return reshape(q), reshape(k), reshape(v)

@torch.no_grad()
def stream_mha_cached_outputs(mha, x):
    """Return per-step outputs for MHA with KV caching. Shape: (L,B,E)."""
    device = x.device
    B = x.size(1); E = mha.embed_dim; H = mha.num_heads; Dh = E // H
    K_cache = torch.empty(B, H, 0, Dh, device=device, dtype=x.dtype)
    V_cache = torch.empty(B, H, 0, Dh, device=device, dtype=x.dtype)
    outs = []
    for t in range(x.shape[0]):
        step = x[t]  # (B,E)
        q, k, v = _project_qkv_from_mha(mha, step)  # (B,H,1,Dh)
        K_cache = torch.cat([K_cache, k], dim=2)
        V_cache = torch.cat([V_cache, v], dim=2)
        attn = F.scaled_dot_product_attention(q, K_cache, V_cache, attn_mask=None, is_causal=False)  # (B,H,1,Dh)
        y_t = mha.out_proj(attn.reshape(B, 1, H * Dh)).squeeze(1)  # (B,E)
        outs.append(y_t.unsqueeze(0))
    return torch.cat(outs, dim=0)

@torch.no_grad()
def stream_reservoir_outputs(ra, x, reservoir_state):
    """Return per-step outputs for RA (one token at a time). Shape: (L,B,E)."""
    state = reservoir_state.clone()
    outs = []
    for t in range(x.shape[0]):
        step = x[t:t+1]  # (1,B,E)
        y_step, state = ra(step, state)  # (1,B,E), (B,R,1)
        outs.append(y_step)  # already (1,B,E)
    return torch.cat(outs, dim=0), state

@torch.no_grad()
def full_reservoir_outputs(ra, x, reservoir_state):
    """Return full-seq outputs for RA in one call. Shape: (L,B,E)."""
    y, new_state = ra(x, reservoir_state.clone())
    return y, new_state

@torch.no_grad()
def mse_all(a, b):
    return F.mse_loss(a, b).item()

@torch.no_grad()
def mse_single_step(a, b, t):
    # a,b: (L,B,E)
    return F.mse_loss(a[t], b[t]).item()

def main():
    # --- Config ---
    device = "cpu"  # or "cuda"
    dtype = torch.float32
    seed = 0
    threads = 1

    seq_len = 128
    batch_size = 1
    embed_dim = 768
    num_heads = 4

    resSize = 256
    inSize = embed_dim
    density = 0.01
    topology = "uniform"
    a = 0.3

    set_reproducible(seed, threads)

    x = torch.randn(seq_len, batch_size, embed_dim, device=device, dtype=dtype)

    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads,
                                batch_first=False, device=device, dtype=dtype)

    ra = ReservoirAttention(embed_dim=embed_dim, num_heads=num_heads,
                            resSize=resSize, inSize=inSize,
                            density=density, topology=topology,
                            a=a, dtype=dtype, device=device)

    reservoir_state = torch.zeros((batch_size, resSize, 1), dtype=dtype, device=device)

    # === Timings ===
    t_mha_full = time_full_seq_mha(mha, x)
    t_ra_full = time_full_seq_reservoir(ra, x, reservoir_state)

    print("\n[Full sequence]")
    print(f"MHA               : {t_mha_full*1000:.2f} ms")
    print(f"ReservoirAttention: {t_ra_full*1000:.2f} ms")

    t_mha_prefix = None
    t_mha_cached = None
    t_ra_stream  = None

    # Streaming timings
    if device == "cuda": torch.cuda.synchronize()
    start = time.time(); outs_mha_prefix = stream_mha_prefix_outputs(mha, x); end = time.time()
    t_mha_prefix = end - start

    if device == "cuda": torch.cuda.synchronize()
    start = time.time(); outs_mha_cached = stream_mha_cached_outputs(mha, x); end = time.time()
    t_mha_cached = end - start

    if device == "cuda": torch.cuda.synchronize()
    start = time.time(); outs_ra_stream, _ = stream_reservoir_outputs(ra, x, reservoir_state); end = time.time()
    t_ra_stream = end - start

    print("\n[Streaming, step-by-step]")
    print(f"MHA (prefix)      : {t_mha_prefix*1000:.2f} ms")
    print(f"MHA (cached KV)   : {t_mha_cached*1000:.2f} ms")
    print(f"ReservoirAttention: {t_ra_stream*1000:.2f} ms")

    # === MSE checks ===
    # MHA: prefix vs cached (should be ~0)
    mse_mha_all = mse_all(outs_mha_prefix, outs_mha_cached)
    t_pick = seq_len // 2
    mse_mha_step = mse_single_step(outs_mha_prefix, outs_mha_cached, t_pick)

    # RA: full vs streaming (should be ~0)
    outs_ra_full, _ = full_reservoir_outputs(ra, x, reservoir_state)
    mse_ra_all = mse_all(outs_ra_full, outs_ra_stream)
    mse_ra_step = mse_single_step(outs_ra_full, outs_ra_stream, t_pick)

    print("\n[MSE consistency checks]")
    print(f"MHA  prefix vs cached  (all steps): {mse_mha_all:.6e}")
    print(f"MHA  prefix vs cached  (t={t_pick}): {mse_mha_step:.6e}")
    print(f"RA   full   vs stream  (all steps): {mse_ra_all:.6e}")
    print(f"RA   full   vs stream  (t={t_pick}): {mse_ra_step:.6e}")

if __name__ == "__main__":
    main()
