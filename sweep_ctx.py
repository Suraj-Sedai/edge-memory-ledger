#!/usr/bin/env python3
"""
Context-length sweep for llama-cpp-python on the Coralboard (SL2610).

Each (n_ctx, repeat) runs in a FRESH child process, because VmHWM is a
per-process high-water mark and never goes down. Running several n_ctx values
in one process would make every row after the first meaningless.

For each run it records:
  - predicted KV bytes (written to the terminal BEFORE the run starts)
  - runtime-reported buffers parsed from llama.cpp's own log
    (model buffers by type, KV, compute, output)
  - OS-side memory: VmRSS split into RssAnon / RssFile at each stage,
    VmHWM at the end, sampled peak RssAnon / RssFile, min MemAvailable
  - timing: load, warm-up, prefill tok/s, decode tok/s

Usage on the board:
  python3 sweep_ctx.py --model /home/root/qwen2.5-0.5b-instruct-q4_k_m.gguf \
      --ctx 128 512 1024 2048 4096 --fill 0.9 --repeats 3 --out fill_sweep.csv
  python3 sweep_ctx.py --model ... --ctx 128 1024 4096 8192 16384 32768 \
      --prompt-tokens 64 --repeats 3 --out alloc_sweep.csv
"""
import argparse, csv, json, os, re, subprocess, sys, threading, time

MiB = 1024 * 1024

# ----------------------------------------------------------------- /proc
def read_status():
    """Return selected /proc/self/status fields in kB."""
    out = {}
    with open("/proc/self/status") as f:
        for line in f:
            k, _, v = line.partition(":")
            if k in ("VmRSS", "VmHWM", "RssAnon", "RssFile", "RssShmem", "VmSize"):
                out[k] = int(v.split()[0])
    return out

def mem_available_kb():
    with open("/proc/meminfo") as f:
        for line in f:
            if line.startswith("MemAvailable:"):
                return int(line.split()[1])
    return -1

class Sampler(threading.Thread):
    """Polls RSS components and MemAvailable. VmHWM is exact (kernel-tracked);
    this adds the anon/file split at the peak, which the kernel does not keep."""
    def __init__(self, period):
        super().__init__(daemon=True)
        self.period, self.stop = period, threading.Event()
        self.peak_anon = self.peak_file = self.peak_rss = 0
        self.min_avail = 1 << 62
    def run(self):
        while not self.stop.is_set():
            s = read_status()
            self.peak_anon = max(self.peak_anon, s.get("RssAnon", 0))
            self.peak_file = max(self.peak_file, s.get("RssFile", 0))
            self.peak_rss = max(self.peak_rss, s.get("VmRSS", 0))
            self.min_avail = min(self.min_avail, mem_available_kb())
            time.sleep(self.period)

# ------------------------------------------------------------ prediction
def predict_kv_bytes(n_ctx, layers, kv_heads, head_dim, bytes_per_el):
    # 2 (K and V) x layers x kv_heads x head_dim x bytes x tokens
    return 2 * layers * kv_heads * head_dim * bytes_per_el * n_ctx

# --------------------------------------------------------- log parsing
# Patterns are deliberately loose: llama.cpp renames these lines between
# versions (llama_kv_cache_unified -> llama_kv_cache, etc.).
RE_MODEL_BUF   = re.compile(r"(\S+) model buffer size\s*=\s*([\d.]+) MiB")
RE_KV_BUF      = re.compile(r"(\S+) KV buffer size\s*=\s*([\d.]+) MiB")
RE_KV_SUMMARY  = re.compile(r"size\s*=\s*([\d.]+) MiB \(\s*(\d+) cells")
RE_K           = re.compile(r"K \((\w+)\):\s*([\d.]+) MiB")
RE_V           = re.compile(r"V \((\w+)\):\s*([\d.]+) MiB")
RE_COMPUTE_BUF = re.compile(r"(\S+) compute buffer size\s*=\s*([\d.]+) MiB")
RE_OUTPUT_BUF  = re.compile(r"(\S+)\s+output buffer size\s*=\s*([\d.]+) MiB")
RE_NCTX        = re.compile(r"\bn_ctx\s*=\s*(\d+)")
RE_FLASH       = re.compile(r"flash_attn\s*=\s*(\S+)")

def parse_llama_log(text):
    r = {}
    mb = RE_MODEL_BUF.findall(text)
    r["model_bufs"] = ";".join(f"{n}={v}" for n, v in mb)
    r["model_buf_total_mib"] = round(sum(float(v) for _, v in mb), 2)
    kvb = RE_KV_BUF.findall(text)
    r["kv_buf_reported_mib"] = round(sum(float(v) for _, v in kvb), 2) if kvb else None
    m = RE_KV_SUMMARY.search(text)
    r["kv_cells"] = int(m.group(2)) if m else None
    if r["kv_buf_reported_mib"] is None and m:
        r["kv_buf_reported_mib"] = float(m.group(1))
    k, v = RE_K.search(text), RE_V.search(text)
    r["kv_dtype"] = k.group(1) if k else None
    r["k_mib"] = float(k.group(2)) if k else None
    r["v_mib"] = float(v.group(2)) if v else None
    cb = RE_COMPUTE_BUF.findall(text)
    r["compute_bufs"] = ";".join(f"{n}={v}" for n, v in cb)
    r["compute_buf_total_mib"] = round(sum(float(v) for _, v in cb), 2)
    ob = RE_OUTPUT_BUF.findall(text)
    r["output_buf_total_mib"] = round(sum(float(v) for _, v in ob), 2)
    nc = RE_NCTX.findall(text)
    r["n_ctx_runtime"] = int(nc[-1]) if nc else None
    fa = RE_FLASH.findall(text)
    r["flash_attn_runtime"] = fa[-1] if fa else None
    return r

# ---------------------------------------------------------------- worker
def worker(a):
    """Runs inside the child. Prints one line 'RESULT {json}' to stdout.
    llama.cpp's verbose log goes to stderr, which the parent captures."""
    kb = lambda s, k: round(s.get(k, 0) / 1024, 2)  # kB -> MiB
    res = {}
    s0 = read_status()
    from llama_cpp import Llama            # import cost counted in baseline
    s1 = read_status()
    res.update(rss_preimport_mib=kb(s0, "VmRSS"), rss_base_mib=kb(s1, "VmRSS"),
               anon_base_mib=kb(s1, "RssAnon"), file_base_mib=kb(s1, "RssFile"))

    sampler = Sampler(a.sample_ms / 1000); sampler.start()

    t = time.perf_counter()
    m = Llama(a.model, n_ctx=a.n_ctx, n_batch=a.n_batch, n_ubatch=a.n_ubatch,
              n_threads=a.threads, n_threads_batch=a.threads,
              use_mmap=not a.no_mmap, flash_attn=a.flash_attn,
              logits_all=False, verbose=True)
    res["load_s"] = round(time.perf_counter() - t, 3)
    s2 = read_status()
    res.update(rss_loaded_mib=kb(s2, "VmRSS"), anon_loaded_mib=kb(s2, "RssAnon"),
               file_loaded_mib=kb(s2, "RssFile"))

    md = m.metadata or {}
    arch = md.get("general.architecture", "")
    def g(key):
        v = md.get(f"{arch}.{key}")
        return int(v) if v is not None else None
    n_embd, n_head = g("embedding_length"), g("attention.head_count")
    res.update(arch=arch, gguf_layers=g("block_count"), gguf_kv_heads=g("attention.head_count_kv"),
               gguf_head_dim=(n_embd // n_head) if n_embd and n_head else None,
               n_ctx_train=g("context_length"))

    # Synthetic prompt. Content is irrelevant to memory; only the length matters.
    seed = ("The quick brown fox jumps over the lazy dog while the memory "
            "controller waits for the next cache line to arrive. ")
    base = m.tokenize(seed.encode(), add_bos=False)
    need = a.prompt_len
    toks = (base * (need // len(base) + 1))[:need]
    bos = m.token_bos()
    if bos is not None and bos >= 0:
        toks = [bos] + toks[:-1]

    # Warm-up: one short pass touches every weight page (cold mmap reads
    # from eMMC would otherwise pollute the prefill timing). Then reset.
    t = time.perf_counter(); m.eval(toks[:8]); res["warmup_s"] = round(time.perf_counter() - t, 3)
    m.reset()

    t = time.perf_counter(); m.eval(toks); dt = time.perf_counter() - t
    res.update(prompt_tokens=len(toks), prefill_s=round(dt, 3),
               prefill_tps=round(len(toks) / dt, 3))
    s3 = read_status()
    res.update(rss_after_prefill_mib=kb(s3, "VmRSS"), anon_after_prefill_mib=kb(s3, "RssAnon"))

    # Decode a fixed number of tokens, greedy, ignoring EOS (we want a fixed
    # workload, not a sensible answer).
    n = 0; t = time.perf_counter()
    for _ in range(a.gen):
        tok = m.sample(temp=0.0)
        m.eval([tok]); n += 1
    dt = time.perf_counter() - t
    res.update(gen_tokens=n, decode_s=round(dt, 3), decode_tps=round(n / dt, 3) if n else None)

    sampler.stop.set(); sampler.join()
    s4 = read_status()
    res.update(rss_end_mib=kb(s4, "VmRSS"), vmhwm_mib=kb(s4, "VmHWM"),
               peak_rss_sampled_mib=round(sampler.peak_rss / 1024, 2),
               peak_anon_sampled_mib=round(sampler.peak_anon / 1024, 2),
               peak_file_sampled_mib=round(sampler.peak_file / 1024, 2),
               memavail_min_mib=round(sampler.min_avail / 1024, 2))
    sys.stderr.flush()
    print("RESULT " + json.dumps(res), flush=True)

# ---------------------------------------------------------------- parent
COLUMNS = [
    "n_ctx", "repeat", "status", "prompt_tokens", "gen_tokens",
    "kv_pred_mib", "kv_buf_reported_mib", "kv_err_pct", "kv_cells", "kv_dtype", "k_mib", "v_mib",
    "model_buf_total_mib", "model_bufs", "compute_buf_total_mib", "compute_bufs", "output_buf_total_mib",
    "runtime_total_mib",
    "rss_preimport_mib", "rss_base_mib", "anon_base_mib", "file_base_mib",
    "rss_loaded_mib", "anon_loaded_mib", "file_loaded_mib",
    "rss_after_prefill_mib", "anon_after_prefill_mib", "rss_end_mib",
    "vmhwm_mib", "peak_rss_sampled_mib", "peak_anon_sampled_mib", "peak_file_sampled_mib",
    "vmhwm_minus_base_mib", "peak_anon_minus_base_mib", "memavail_min_mib",
    "load_s", "warmup_s", "prefill_s", "prefill_tps", "decode_s", "decode_tps",
    "n_ctx_runtime", "n_ctx_train", "flash_attn_runtime", "n_batch", "n_ubatch", "threads", "use_mmap",
    "arch", "gguf_layers", "gguf_kv_heads", "gguf_head_dim", "wall_s", "log_file",
]

def run_one(a, n_ctx, rep, logdir):
    if a.prompt_tokens is not None:
        plen = a.prompt_tokens
    else:
        plen = int(a.fill * n_ctx) - a.gen
    plen = max(8, min(plen, n_ctx - a.gen - 1))

    pred = predict_kv_bytes(n_ctx, a.layers, a.kv_heads, a.head_dim, a.kv_bytes)
    print(f"[predict] n_ctx={n_ctx:>6} rep={rep}  KV = {pred:,} B = {pred / MiB:.3f} MiB  "
          f"(prompt {plen} + gen {a.gen} tokens)", flush=True)

    cmd = [sys.executable, os.path.abspath(__file__), "--_worker",
           "--model", a.model, "--n-ctx", str(n_ctx), "--prompt-len", str(plen),
           "--gen", str(a.gen), "--n-batch", str(a.n_batch), "--n-ubatch", str(a.n_ubatch),
           "--threads", str(a.threads), "--sample-ms", str(a.sample_ms)]
    if a.no_mmap: cmd.append("--no-mmap")
    if a.flash_attn: cmd.append("--flash-attn")

    t = time.perf_counter()
    try:
        p = subprocess.run(cmd, capture_output=True, text=True, timeout=a.timeout)
        rc, out, err = p.returncode, p.stdout, p.stderr
        status = "ok" if rc == 0 else ("killed(OOM?)" if rc in (-9, 137) else f"rc={rc}")
    except subprocess.TimeoutExpired as e:
        out = (e.stdout or b"").decode() if isinstance(e.stdout, bytes) else (e.stdout or "")
        err = (e.stderr or b"").decode() if isinstance(e.stderr, bytes) else (e.stderr or "")
        status = "timeout"
    wall = time.perf_counter() - t

    log_file = os.path.join(logdir, f"ctx{n_ctx}_rep{rep}.log")
    with open(log_file, "w") as f:
        f.write(err); f.write("\n--- stdout ---\n"); f.write(out)

    row = dict(n_ctx=n_ctx, repeat=rep, status=status, wall_s=round(wall, 1), log_file=log_file,
               kv_pred_mib=round(pred / MiB, 4), n_batch=a.n_batch, n_ubatch=a.n_ubatch,
               threads=a.threads, use_mmap=not a.no_mmap)
    row.update(parse_llama_log(err))           # partial info even if the run died
    for line in out.splitlines():
        if line.startswith("RESULT "):
            row.update(json.loads(line[7:]))

    if row.get("kv_buf_reported_mib"):
        row["kv_err_pct"] = round(100 * (row["kv_buf_reported_mib"] - row["kv_pred_mib"]) / row["kv_pred_mib"], 3)
    row["runtime_total_mib"] = round(sum(row.get(k) or 0 for k in
        ("model_buf_total_mib", "kv_buf_reported_mib", "compute_buf_total_mib", "output_buf_total_mib")), 2)
    if row.get("vmhwm_mib") is not None:
        row["vmhwm_minus_base_mib"] = round(row["vmhwm_mib"] - row["rss_base_mib"], 2)
        row["peak_anon_minus_base_mib"] = round(row["peak_anon_sampled_mib"] - row["anon_base_mib"], 2)
    if row.get("gguf_layers") and (row["gguf_layers"], row["gguf_kv_heads"], row["gguf_head_dim"]) \
            != (a.layers, a.kv_heads, a.head_dim):
        print(f"  !! GGUF shape {row['gguf_layers']}/{row['gguf_kv_heads']}/{row['gguf_head_dim']} "
              f"differs from --layers/--kv-heads/--head-dim; prediction is wrong", flush=True)

    print(f"  -> {status}  KV rep {row.get('kv_buf_reported_mib')} MiB  "
          f"compute {row.get('compute_buf_total_mib')}  VmHWM {row.get('vmhwm_mib')}  "
          f"peakAnon {row.get('peak_anon_sampled_mib')}  "
          f"prefill {row.get('prefill_tps')} t/s  decode {row.get('decode_tps')} t/s  [{wall:.0f}s]",
          flush=True)
    return row

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--ctx", type=int, nargs="+", default=[128, 512, 1024, 2048, 4096])
    ap.add_argument("--repeats", type=int, default=3)
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--fill", type=float, default=0.9, help="prompt = fill*n_ctx - gen (default)")
    g.add_argument("--prompt-tokens", type=int, default=None, help="fixed prompt length instead of --fill")
    ap.add_argument("--gen", type=int, default=32)
    ap.add_argument("--n-batch", type=int, default=512)
    ap.add_argument("--n-ubatch", type=int, default=512)
    ap.add_argument("--threads", type=int, default=2,
                    help="llama-cpp-python defaults to cpu_count//2 = 1 on this board")
    ap.add_argument("--no-mmap", action="store_true")
    ap.add_argument("--flash-attn", action="store_true")
    ap.add_argument("--sample-ms", type=float, default=20)
    ap.add_argument("--timeout", type=float, default=3600)
    # model shape for the pre-run prediction (Qwen2.5-0.5B defaults)
    ap.add_argument("--layers", type=int, default=24)
    ap.add_argument("--kv-heads", type=int, default=2)
    ap.add_argument("--head-dim", type=int, default=64)
    ap.add_argument("--kv-bytes", type=int, default=2, help="f16 = 2")
    ap.add_argument("--out", default="sweep.csv")
    # internal
    ap.add_argument("--_worker", action="store_true", help=argparse.SUPPRESS)
    ap.add_argument("--n-ctx", type=int, help=argparse.SUPPRESS)
    ap.add_argument("--prompt-len", type=int, help=argparse.SUPPRESS)
    a = ap.parse_args()

    if a._worker:
        return worker(a)

    logdir = os.path.splitext(a.out)[0] + "_logs"
    os.makedirs(logdir, exist_ok=True)
    new = not os.path.exists(a.out)
    with open(a.out, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=COLUMNS, extrasaction="ignore")
        if new: w.writeheader()
        for n_ctx in a.ctx:
            for rep in range(a.repeats):
                w.writerow(run_one(a, n_ctx, rep, logdir)); f.flush()
    print(f"wrote {a.out}; raw llama.cpp logs in {logdir}/")

if __name__ == "__main__":
    main()
