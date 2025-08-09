#!/usr/bin/env python3
"""
Compare PyTorch MHA vs ReservoirAttention on real Wikipedia embeddings.

Reports:
- Full-sequence timings (per sample)
- Streaming timings: MHA (prefix), MHA (cached KV), ReservoirAttention (incremental)
- Single-step consistency MSEs
- Multistep closed-loop consistency MSEs (seed_len, horizon)

Usage examples:
  python benchmark_wikipedia_reservoir_vs_mha_multistep.py --samples 2 --seq_len 256
  python benchmark_wikipedia_reservoir_vs_mha_multistep.py --device cuda --amp --samples 4 --seq_len 512
  python benchmark_wikipedia_reservoir_vs_mha_multistep.py --device cuda --amp --res_size 512 --density 0.005 --topology smallworld
"""
import os, sys, time, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# Make ../src importable for reservoir_attention.py
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from reservoir_attention import ReservoirAttention

# ---------------- Helpers ----------------
def set_repro(seed=0, threads=1):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)

def sync(device_type: str):
    if device_type == "cuda":
        torch.cuda.synchronize()

@torch.no_grad()
def timeit(fn, *args, warmup=1, runs=1, device_type="cpu"):
    for _ in range(warmup):
        _ = fn(*args)
    sync(device_type)
    t0 = time.time()
    out = None
    for _ in range(runs):
        out = fn(*args)
    sync(device_type)
    return (time.time() - t0) / max(1, runs), out

def pad_or_truncate(seq, target_len):
    L, D = seq.shape
    if L == target_len:
        return seq
    if L > target_len:
        return seq[:target_len]
    pad = torch.zeros(target_len - L, D, device=seq.device, dtype=seq.dtype)
    return torch.cat([seq, pad], dim=0)

# ---------------- Embeddings (HF Wikipedia + Transformers) ----------------
def load_wiki_embeddings(model_name, num_samples, max_length, device, dtype, use_amp=False):
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device)
    if device == "cuda" and not use_amp:
        mdl = mdl.to(dtype)

    ds = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_samples}]")
    texts = [ex["text"] for ex in ds]

    batches = []
    with torch.no_grad():
        for txt in texts:
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            if device == "cuda" and use_amp:
                with torch.autocast(device_type="cuda", dtype=torch.float16):
                    last = mdl(**enc).last_hidden_state.squeeze(0)  # (L,E_model)
            else:
                last = mdl(**enc).last_hidden_state.squeeze(0)
            if last.dtype != dtype:
                last = last.to(dtype)
            batches.append(last)
    return batches  # list[(L,E)]

# ---------------- MHA variants ----------------
@torch.no_grad()
def mha_full_seq(mha, x):
    # x: (L,B,E)
    return mha(x, x, x)[0]

@torch.no_grad()
def mha_stream_prefix(mha, x):
    outs = []
    for t in range(1, x.size(0) + 1):
        outs.append(mha(x[:t], x[:t], x[:t])[0][-1].unsqueeze(0))
    return torch.cat(outs, dim=0)  # (L,B,E)

def _project_qkv_from_mha(mha: nn.MultiheadAttention, step):
    # step: (B,E)
    E = mha.embed_dim
    H = mha.num_heads
    Dh = E // H
    proj = F.linear(step, mha.in_proj_weight, mha.in_proj_bias)  # (B,3E)
    q, k, v = proj.split(E, dim=-1)
    view = lambda t: t.view(t.size(0), H, 1, Dh)
    return view(q), view(k), view(v)  # (B,H,1,Dh) each

@torch.no_grad()
def mha_stream_cached(mha, x):
    B = x.size(1)
    E = mha.embed_dim
    H = mha.num_heads
    Dh = E // H
    K = torch.empty(B, H, 0, Dh, device=x.device, dtype=x.dtype)
    V = torch.empty(B, H, 0, Dh, device=x.device, dtype=x.dtype)
    outs = []
    for t in range(x.size(0)):
        step = x[t]  # (B,E)
        q, k, v = _project_qkv_from_mha(mha, step)
        K = torch.cat([K, k], dim=2)
        V = torch.cat([V, v], dim=2)
        attn = F.scaled_dot_product_attention(q, K, V)  # (B,H,1,Dh)
        y = mha.out_proj(attn.reshape(B, 1, H * Dh)).squeeze(1)  # (B,E)
        outs.append(y.unsqueeze(0))
    return torch.cat(outs, dim=0)  # (L,B,E)

# ---------------- ReservoirAttention variants ----------------
@torch.no_grad()
def ra_full_seq(ra, x, state0):
    return ra(x, state0.clone())  # (L,B,E), state

@torch.no_grad()
def ra_stream(ra, x, state0):
    state = state0.clone()
    outs = []
    for t in range(x.size(0)):
        y, state = ra(x[t:t+1], state)  # (1,B,E)
        outs.append(y)
    return torch.cat(outs, dim=0), state  # (L,B,E), state

# ---------------- Multistep closed-loop helpers ----------------
@torch.no_grad()
def init_mha_cache_from_seq(mha, x_seq):
    B = x_seq.size(1)
    E = mha.embed_dim
    H = mha.num_heads
    Dh = E // H
    K = torch.empty(B, H, 0, Dh, device=x_seq.device, dtype=x_seq.dtype)
    V = torch.empty(B, H, 0, Dh, device=x_seq.device, dtype=x_seq.dtype)
    for t in range(x_seq.size(0)):
        q, k, v = _project_qkv_from_mha(mha, x_seq[t])
        K = torch.cat([K, k], dim=2)
        V = torch.cat([V, v], dim=2)
    return K, V  # (B,H,L,Dh)

@torch.no_grad()
def rollout_mha_prefix(mha, seed_seq, horizon):
    seq = seed_seq.clone()          # (L0,B,E)
    outs = []
    for _ in range(horizon):
        y = mha(seq, seq, seq)[0][-1]  # (B,E)
        outs.append(y.unsqueeze(0))
        seq = torch.cat([seq, y.unsqueeze(0)], dim=0)
    return torch.cat(outs, dim=0)   # (H,B,E)

@torch.no_grad()
def rollout_mha_cached(mha, seed_seq, horizon):
    B = seed_seq.size(1)
    E = mha.embed_dim
    Hh = mha.num_heads
    Dh = E // Hh
    # Build initial cache from the seed (including the last token)
    K, V = init_mha_cache_from_seq(mha, seed_seq)  # (B,Hh,L0,Dh)
    x_curr = seed_seq[-1]  # (B,E)
    outs = []
    for _ in range(horizon):
        q, k, v = _project_qkv_from_mha(mha, x_curr)  # (B,Hh,1,Dh)
        K = torch.cat([K, k], dim=2)
        V = torch.cat([V, v], dim=2)
        attn = F.scaled_dot_product_attention(q, K, V)  # (B,Hh,1,Dh)
        y = mha.out_proj(attn.reshape(B, 1, Hh * Dh)).squeeze(1)  # (B,E)
        outs.append(y.unsqueeze(0))
        x_curr = y
    return torch.cat(outs, dim=0)  # (H,B,E)

@torch.no_grad()
def init_ra_state_from_seq(ra, x_seq, state0):
    state = state0.clone()
    if x_seq.size(0) > 0:
        _, state = ra(x_seq, state)
    return state

@torch.no_grad()
def rollout_ra_stream(ra, seed_seq, horizon, state_after_seed):
    state = state_after_seed.clone()
    last_token = seed_seq[-1].unsqueeze(0)  # (1,B,E)
    outs = []
    for _ in range(horizon):
        y, state = ra(last_token, state)  # (1,B,E)
        outs.append(y)
        last_token = y
    return torch.cat(outs, dim=0)  # (H,B,E)

@torch.no_grad()
def rollout_ra_full(ra, seed_seq, horizon, state0):
    seq = seed_seq.clone()
    outs = []
    for _ in range(horizon):
        y_full, _ = ra(seq, state0.clone())  # (L,B,E)
        y = y_full[-1].unsqueeze(0)
        outs.append(y)
        seq = torch.cat([seq, y], dim=0)
    return torch.cat(outs, dim=0)  # (H,B,E)

# ---------------- Main benchmark ----------------
def run_benchmark(args):
    set_repro(args.seed, args.threads)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if (device == "cuda" and args.amp) else torch.float32

    print(f"Device={device}, dtype={dtype}, samples={args.samples}, seq_len={args.seq_len}, embed_model={args.embed_model}")

    # 1) Load embeddings
    emb_list = load_wiki_embeddings(args.embed_model, args.samples, args.seq_len, device, dtype, use_amp=(device=="cuda" and args.amp))

    # Conform to (L,B,E)
    x_list = [pad_or_truncate(emb, args.seq_len).unsqueeze(1) for emb in emb_list]

    # 2) Modules (match embed dim)
    embed_dim = x_list[0].size(-1)
    mha = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=args.heads,
                                batch_first=False, device=device, dtype=dtype)
    ra = ReservoirAttention(embed_dim=embed_dim, num_heads=args.heads,
                            resSize=args.res_size, inSize=embed_dim,
                            density=args.density, topology=args.topology,
                            a=args.leak, dtype=dtype, device=device)

    # 3) Per-sample timings
    full_mha_times, full_ra_times = [], []
    str_prefix_times, str_cached_times, str_ra_times = [], [], []
    total_tokens = 0

    for x in x_list:
        state0 = torch.zeros((x.size(1), args.res_size, 1), dtype=dtype, device=device)
        total_tokens += x.size(0) * x.size(1)

        dt, _ = timeit(mha_full_seq, mha, x, warmup=args.warmup, runs=args.runs, device_type=device)
        full_mha_times.append(dt)

        dt, _ = timeit(ra_full_seq, ra, x, state0, warmup=args.warmup, runs=args.runs, device_type=device)
        full_ra_times.append(dt)

        dt, _ = timeit(mha_stream_prefix, mha, x, warmup=0, runs=1, device_type=device)
        str_prefix_times.append(dt)

        dt, _ = timeit(mha_stream_cached, mha, x, warmup=0, runs=1, device_type=device)
        str_cached_times.append(dt)

        dt, _ = timeit(ra_stream, ra, x, state0, warmup=0, runs=1, device_type=device)
        str_ra_times.append(dt)

    def avg(xs): return sum(xs) / max(1, len(xs))
    def tokens_per_sec(total_tok, total_time): return total_tok / total_time

    full_mha_avg = avg(full_mha_times)
    full_ra_avg  = avg(full_ra_times)
    str_pref_avg = avg(str_prefix_times)
    str_cach_avg = avg(str_cached_times)
    str_ra_avg   = avg(str_ra_times)

    print("\n=== Wikipedia Benchmark ===")
    print(f"Samples: {len(x_list)} | Seq len: {args.seq_len} | Embed dim: {embed_dim}")
    print("\n[Full sequence] (mean per sample)")
    print(f"MHA                : {full_mha_avg*1000:.2f} ms   | throughput ≈ {tokens_per_sec(total_tokens, full_mha_avg*len(x_list)):.1f} tok/s")
    print(f"ReservoirAttention : {full_ra_avg*1000:.2f} ms   | throughput ≈ {tokens_per_sec(total_tokens, full_ra_avg*len(x_list)):.1f} tok/s")

    print("\n[Streaming] (mean per sample)")
    print(f"MHA (prefix)       : {str_pref_avg*1000:.2f} ms")
    print(f"MHA (cached KV)    : {str_cach_avg*1000:.2f} ms")
    print(f"ReservoirAttention : {str_ra_avg*1000:.2f} ms")

    # 4) Single-step MSE (consistency)
    with torch.no_grad():
        x0 = x_list[0]
        st0 = torch.zeros((x0.size(1), args.res_size, 1), dtype=dtype, device=device)

        y_pref = mha_stream_prefix(mha, x0)
        y_cach = mha_stream_cached(mha, x0)
        mse_mha_single = F.mse_loss(y_pref, y_cach).item()

        y_full, _ = ra_full_seq(ra, x0, st0)
        y_step, _ = ra_stream(ra, x0, st0)
        mse_ra_single = F.mse_loss(y_full, y_step).item()

    print("\n[Single-step MSE consistency on sample 0]")
    print(f"MHA  prefix vs cached : {mse_mha_single:.3e}")
    print(f"RA   full   vs stream : {mse_ra_single:.3e}")

    # 5) Multistep closed-loop MSE (consistency)
    seed_len = min(args.seed_len, x0.size(0))
    horizon  = args.horizon
    seed_seq = x0[:seed_len]  # (L0,B,E)

    gen_mha_prefix = rollout_mha_prefix(mha, seed_seq, horizon)   # (H,B,E)
    gen_mha_cached = rollout_mha_cached(mha, seed_seq, horizon)   # (H,B,E)
    mse_mha_multi  = F.mse_loss(gen_mha_prefix, gen_mha_cached).item()

    st_after_seed = init_ra_state_from_seq(ra, seed_seq, st0)
    gen_ra_stream = rollout_ra_stream(ra, seed_seq, horizon, st_after_seed)  # (H,B,E)
    gen_ra_full   = rollout_ra_full(ra, seed_seq, horizon, st0)              # (H,B,E)
    mse_ra_multi  = F.mse_loss(gen_ra_full, gen_ra_stream).item()

    print("\n[Multistep closed-loop MSE on sample 0]")
    print(f"MHA  prefix vs cached : {mse_mha_multi:.3e}  (seed_len={seed_len}, horizon={horizon})")
    print(f"RA   full   vs stream : {mse_ra_multi:.3e}  (seed_len={seed_len}, horizon={horizon})")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true", help="use fp16 on CUDA for embedding model + tensors")
    ap.add_argument("--samples", type=int, default=4)
    ap.add_argument("--seq_len", type=int, default=256)
    ap.add_argument("--embed_model", type=str, default="bert-base-uncased")
    ap.add_argument("--heads", type=int, default=4)
    ap.add_argument("--res_size", type=int, default=256)
    ap.add_argument("--density", type=float, default=0.01)
    ap.add_argument("--topology", type=str, default="uniform", choices=["uniform","geometric","smallworld"])
    ap.add_argument("--leak", type=float, default=0.3)
    ap.add_argument("--warmup", type=int, default=1)
    ap.add_argument("--runs", type=int, default=1)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--threads", type=int, default=1)
    ap.add_argument("--seed_len", type=int, default=16, help="prefix length used as seed for rollout")
    ap.add_argument("--horizon",  type=int, default=32, help="number of generated steps for closed-loop test")
    args = ap.parse_args()
    run_benchmark(args)

if __name__ == "__main__":
    main()
