# Time torch.mul and torch.addmm on COO inputs, median of 100 runs (after
# 10 warm-up runs), CPU and CUDA, nnz 100 and 10^6. Run once with main's torch
# (in-tree kernels) and once with the patched torch plus sparse-ext (out-of-tree); prints JSON.
#   python bench.py LABEL > bench-LABEL.json
import json, os, statistics, sys, time, torch

def coo(m, k, nnz, dev, seed):
    g = torch.Generator().manual_seed(seed)
    lin = torch.randperm(m * k, generator=g)[:nnz].sort().values   # distinct, sorted
    idx = torch.stack([lin // k, lin % k])
    vals = torch.randn(nnz, generator=g, dtype=torch.float64)
    return torch.sparse_coo_tensor(idx, vals, (m, k), is_coalesced=True).to(dev)

def bench(f, dev, reps=100, warm=10):
    sync = torch.cuda.synchronize if dev == "cuda" else (lambda: None)
    for _ in range(warm):
        f()
    sync()
    ts = []
    for _ in range(reps):
        t0 = time.perf_counter(); f(); sync(); ts.append(time.perf_counter() - t0)
    return statistics.median(ts)

rows = []
for dev in ("cpu", "cuda"):
    for nnz, (m, k, n) in ((100, (1000, 1000, 16)), (10**6, (10**4, 10**4, 16))):
        a = coo(m, k, nnz, dev, 1)
        b = torch.sparse_coo_tensor(a._indices(), torch.randn(nnz, dtype=torch.float64, device=dev), a.shape, is_coalesced=True)
        D = torch.randn(k, n, dtype=torch.float64, device=dev)
        T = torch.randn(m, n, dtype=torch.float64, device=dev)
        cases = {
            "mul(sparse, sparse)": lambda: torch.mul(a, b),
            "mul(sparse, 2.0)": lambda: torch.mul(a, 2.0),
            "addmm(T, sparse, D)": lambda: torch.addmm(T, a, D, beta=0.5, alpha=2.0),
        }
        for name, f in cases.items():
            rows.append(dict(device=dev, nnz=nnz, shape=[m, k, n], op=name, median_s=bench(f, dev)))
            print(dev, nnz, name, f"{rows[-1]['median_s'] * 1e6:.1f} us", file=sys.stderr)
print(json.dumps(dict(label=sys.argv[1], torch=os.path.dirname(torch.__file__), threads=torch.get_num_threads(),
                      gpu=torch.cuda.get_device_name(0), dtype="float64", rows=rows), indent=1))
