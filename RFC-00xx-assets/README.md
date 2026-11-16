# Assets of RFC-00xx: two sparse kernel families moved out of core

An experiment: COO `mul` and `addmm`, CPU and CUDA, removed from ATen and registered
from a package built on the stable ABI, which `import torch` loads through a
`torch.backends` entry point. Measured on two families only; MPS and XPU not tried.

| File | What it is | Applies to |
|---|---|---|
| `pytorch-out-of-tree.diff` | Deletes the `SparseCPU`/`SparseCUDA` entries of `mul.out`, `addmm.out`, `addmm` and `addmm_` in `native_functions.yaml` (8 lines). The kernel sources stay. | pytorch `a67a74d3e71bf2891b0d02b126fe7390090e1f1c` |
| `sparse-ext.diff` | The package, `sparse-ext/` (1028 lines): stable-ABI kernels for `mul.out` and `addmm`, `TORCH_TARGET_VERSION` 2.13, and two routes to register `addmm`: Python (`torch.library`, stable only) and a C++ bridge (`register_addmm.cpp`, the one non-stable file). `TORCH_SPARSE_EXT_ADDMM=python` or `cpp` (default) picks the route. | a new directory; builds against the patched pytorch |
| `sparse-ext.diff`: `csrc/probe_scalar.cpp` | A stable-ABI kernel with a `Scalar` argument. Calling `torch.ops.sparse_ext_probe.scalar_arg(torch.ones(1), 2.0)` raises `RuntimeError: Not yet supported conversion from IValue to StableIValue for schema type: Scalar`. | — |
| `correctness.py` | 1414 checks of `mul` and `addmm` against dense references, CPU and CUDA | either torch |
| `bench.py` | the timings below; prints JSON | either torch |

## Reproduce

```
git -C pytorch checkout a67a74d3e71bf2891b0d02b126fe7390090e1f1c
git -C pytorch apply pytorch-out-of-tree.diff     # then build pytorch from source
git apply sparse-ext.diff                           # creates sparse-ext/
pip install --no-build-isolation -v sparse-ext
(cd pytorch && python -m pytest test/test_sparse.py)
python correctness.py
python bench.py out-of-tree > bench-out-of-tree.json   # and with an unpatched torch
TORCH_SPARSE_EXT_ADDMM=python python bench.py python > bench-python.json
```

## Tests

| Run | main | patched + package, C++ bridge | patched + package, Python route |
|---|---|---|---|
| all of `test/test_sparse.py` | 5513 passed, 668 skipped, 2 xfailed | the same | the same |
| `mul`/`addmm` against dense references (1414 checks) | 1414/1414 | 1414/1414 | 1414/1414 |

With `TORCH_DEVICE_BACKEND_AUTOLOAD=0` the package is not loaded, and sparse `mul` and
`addmm` raise `NotImplementedError`.

## Timings

Median of 100 runs after 10 warm-up runs; median of three alternating rounds per side.
float64 COO, coalesced. Small: 1000×1000, nnz 100. Large: 10⁴×10⁴, nnz 10⁶. `addmm`:
`D` with 16 columns, `beta=0.5`, `alpha=2.0`. AMD Ryzen 9 9950X (16 cores, 32 torch
threads), NVIDIA RTX 5060 Ti. Ratios are out-of-tree over in-tree.

| device | nnz | op | in-tree | C++ bridge | ratio | Python route | ratio |
|---|---|---|---|---|---|---|---|
| cpu | 100 | `mul(sparse, sparse)` | 2.5 µs | 7.0 µs | 2.75× | 7.1 µs | 2.77× |
| cpu | 100 | `mul(sparse, 2.0)` | 2.2 µs | 4.9 µs | 2.18× | 4.9 µs | 2.20× |
| cpu | 100 | `addmm(T, sparse, D)` | 3.2 µs | 4.5 µs | 1.43× | 6.2 µs | 1.96× |
| cpu | 10⁶ | `mul(sparse, sparse)` | 7,990 µs | 16,779 µs | 2.10× | 17,986 µs | 2.25× |
| cpu | 10⁶ | `mul(sparse, 2.0)` | 60.2 µs | 87.4 µs | 1.45× | 86.5 µs | 1.44× |
| cpu | 10⁶ | `addmm(T, sparse, D)` | 11,987 µs | 3,899 µs | 0.33× | 3,916 µs | 0.33× |
| cuda | 100 | `mul(sparse, sparse)` | 20.5 µs | 92.9 µs | 4.54× | 93.1 µs | 4.55× |
| cuda | 100 | `mul(sparse, 2.0)` | 9.5 µs | 16.0 µs | 1.68× | 15.9 µs | 1.67× |
| cuda | 100 | `addmm(T, sparse, D)` | 31.7 µs | 35.0 µs | 1.10× | 37.0 µs | 1.17× |
| cuda | 10⁶ | `mul(sparse, sparse)` | 402.8 µs | 880.2 µs | 2.19× | 879.6 µs | 2.18× |
| cuda | 10⁶ | `mul(sparse, 2.0)` | 116.5 µs | 146.7 µs | 1.26× | 146.7 µs | 1.26× |
| cuda | 10⁶ | `addmm(T, sparse, D)` | 632.5 µs | 633.9 µs | 1.00× | 638.6 µs | 1.01× |

- `mul` is slower because the stable ABI has no setter for a sparse tensor's indices
  and values (one more copy per call), reads metadata through dispatcher calls, and
  has no cuSPARSE handle (the package keeps its own pool).
- CPU `addmm` at nnz 10⁶ is faster out-of-tree because the port inlines a loop that
  in-tree runs as one BLAS call per nonzero; expect that to reverse for wide `D`.
