# benchmark_wikipedia_reservoir_vs_mha.py
import os, sys, time, math, argparse
import torch
import torch.nn as nn
import torch.nn.functional as F

# repo-local import
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from reservoir_attention import ReservoirAttention

# ---- Helpers ----
def set_repro(seed=0, threads=1):
    torch.manual_seed(seed); torch.cuda.manual_seed_all(seed)
    torch.set_num_threads(threads)

@torch.no_grad()
def mha_full_seq(mha, x):
    return mha(x, x, x)[0]

@torch.no_grad()
def ra_full_seq(ra, x, state0):
    return ra(x, state0.clone())

@torch.no_grad()
def mha_stream_prefix(mha, x):
    outs = []
    for t in range(1, x.size(0) + 1):
        outs.append(mha(x[:t], x[:t], x[:t])[0][-1].unsqueeze(0))
    return torch.cat(outs, 0)

def _project_qkv_from_mha(mha: nn.MultiheadAttention, step):
    E = mha.embed_dim; H = mha.num_heads; Dh = E // H
    proj = F.linear(step, mha.in_proj_weight, mha.in_proj_bias)  # (B,3E)
    q, k, v = proj.split(E, dim=-1)
    view = lambda t: t.view(t.size(0), H, 1, Dh)
    return view(q), view(k), view(v)

@torch.no_grad()
def mha_stream_cached(mha, x):
    B = x.size(1); E = mha.embed_dim; H = mha.num_heads; Dh = E // H
    K = torch.empty(B, H, 0, Dh, device=x.device, dtype=x.dtype)
    V = torch.empty(B, H, 0, Dh, device=x.device, dtype=x.dtype)
    outs = []
    for t in range(x.size(0)):
        step = x[t]                                   # (B,E)
        q, k, v = _project_qkv_from_mha(mha, step)    # (B,H,1,Dh)
        K = torch.cat([K, k], 2); V = torch.cat([V, v], 2)
        attn = F.scaled_dot_product_attention(q, K, V)  # (B,H,1,Dh)
        y = mha.out_proj(attn.reshape(B, 1, H*Dh)).squeeze(1)  # (B,E)
        outs.append(y.unsqueeze(0))
    return torch.cat(outs, 0)

@torch.no_grad()
def ra_stream(ra, x, state0):
    state = state0.clone()
    outs = []
    for t in range(x.size(0)):
        y, state = ra(x[t:t+1], state)  # (1,B,E), state
        outs.append(y)
    return torch.cat(outs, 0), state

def sync(dev): 
    if dev == "cuda": torch.cuda.synchronize()

def timeit(fn, *args, warmup=1, runs=1, device_type="cpu"):
    # Warmup
    for _ in range(warmup): _ = fn(*args)
    sync(device_type)
    t0 = time.time()
    for _ in range(runs): out = fn(*args)
    sync(device_type)
    dt = (time.time() - t0) / runs
    return dt, out

# ---- Data: Wikipedia embeddings via HF ----
def load_wiki_embeddings(model_name, num_samples, max_length, device, dtype):
    from datasets import load_dataset
    from transformers import AutoTokenizer, AutoModel

    tok = AutoTokenizer.from_pretrained(model_name)
    mdl = AutoModel.from_pretrained(model_name).to(device).to(dtype)
    mdl.eval()

    ds = load_dataset("wikipedia", "20220301.en", split=f"train[:{num_samples}]")
    texts = [ex["text"] for ex in ds]

    batches = []
    with torch.no_grad():
        for txt in texts:
            enc = tok(txt, return_tensors="pt", truncation=True, max_length=max_length)
            enc = {k: v.to(device) for k, v in enc.items()}
            last = mdl(**enc).last_hidden_state.squeeze(0)   # (L,E_model)
            batches.append(last)
    return batches  # list of (L,E_model) tensors

def pad_or_truncate(seq, target_len):
    L, D = seq.shape
    if L == target_len: return seq
    if L > target_len:  return seq[:target_len]
    # pad with zeros
    pad = torch.zeros(target_len - L, D, device=seq.device, dtype=seq.dtype)
    return torch.cat([seq, pad], dim=0)

# ---- Benchmark runner ----
def run_benchmark(args):
    set_repro(args.seed, args.threads)
    device = "cuda" if (args.device == "cuda" and torch.cuda.is_available()) else "cpu"
    dtype = torch.float16 if (device == "cuda" and args.amp) else torch.float32

    print(f"Device={device}, dtype={dtype}, samples={args.samples}, seq_len={args.seq_len}, embed_model={args.embed_model}")

    # 1) Load embeddings
    emb_list = load_wiki_embeddings(args.embed_model, args.samples, args.seq_len, device, dtype)
    # Conform all to (L,B,E)
    x_list = []
    for emb in emb_list:
        emb = pad_or_truncate(emb, args.seq_len)              # (L,E)
        x_list.append(emb.unsqueeze(1))                       # (L,1,E)

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

    for i, x in enumerate(x_list):
        state0 = torch.zeros((x.size(1), args.res_size, 1), dtype=dtype, device=device)
        total_tokens += x.size(0) * x.size(1)

        # full-seq
        dt, _ = timeit(mha_full_seq, mha, x, warmup=args.warmup, runs=args.runs, device_type=device)
        full_mha_times.append(dt)

        dt, _ = timeit(ra_full_seq, ra, x, state0, warmup=args.warmup, runs=args.runs, device_type=device)
        full_ra_times.append(dt)

        # streaming (prefix)
        dt, _ = timeit(mha_stream_prefix, mha, x, warmup=0, runs=1, device_type=device)
        str_prefix_times.append(dt)

        # streaming (cached)
        dt, _ = timeit(mha_stream_cached, mha, x, warmup=0, runs=1, device_type=device)
        str_cached_times.append(dt)

        # streaming (RA)
        dt, _ = timeit(ra_stream, ra, x, state0, warmup=0, runs=1, device_type=device)
        str_ra_times.append(dt)

    # 4) Aggregate + tokens/sec
    def avg(xs): return sum(xs) / max(1, len(xs))
    def tps(total_tok, total_time): return total_tok / total_time

    full_mha_avg = avg(full_mha_times)
    full_ra_avg  = avg(full_ra_times)
    str_pref_avg = avg(str_prefix_times)
    str_cach_avg = avg(str_cached_times)
    str_ra_avg   = avg(str_ra_times)

    print("\n=== Wikipedia Benchmark ===")
    print(f"Samples: {len(x_list)} | Seq len: {args.seq_len} | Embed dim: {embed_dim}")
    print("\n[Full sequence] (mean per sample)")
    print(f"MHA                : {full_mha_avg*1000:.2f} ms   | throughput ≈ {tps(total_tokens, full_mha_avg*len(x_list)):.1f} tok/s")
    print(f"ReservoirAttention : {full_ra_avg*1000:.2f} ms   | throughput ≈ {tps(total_tokens, full_ra_avg*len(x_list)):.1f} tok/s")

    print("\n[Streaming] (mean per sample)")
    print(f"MHA (prefix)       : {str_pref_avg*1000:.2f} ms")
    print(f"MHA (cached KV)    : {str_cach_avg*1000:.2f} ms")
    print(f"ReservoirAttention : {str_ra_avg*1000:.2f} ms")

    # 5) Optional consistency MSE on first sample (sanity)
    with torch.no_grad():
        x0 = x_list[0]
        st0 = torch.zeros((x0.size(1), args.res_size, 1), dtype=dtype, device=device)
        # MHA: prefix vs cached
        y_pref = mha_stream_prefix(mha, x0)
        y_cach = mha_stream_cached(mha, x0)
        mse_mha = F.mse_loss(y_pref, y_cach).item()
        # RA: full vs stream
        y_full, _ = ra_full_seq(ra, x0, st0)
        y_step, _ = ra_stream(ra, x0, st0)
        mse_ra = F.mse_loss(y_full, y_step).item()
    print("\n[MSE consistency checks on sample 0]")
    print(f"MHA  prefix vs cached : {mse_mha:.3e}")
    print(f"RA   full   vs stream : {mse_ra:.3e}")

if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--device", choices=["cpu","cuda"], default="cpu")
    ap.add_argument("--amp", action="store_true", help="use fp16 on CUDA")
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
    args = ap.parse_args()
    run_benchmark(args)
