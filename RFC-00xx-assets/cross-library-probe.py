"""Probe each Array API 2025.12 function (main namespace and linalg extension) on a small sparse
input in one library, and classify the result:
  sparse - the call works and returns the library's sparse type (or a scalar, or a tuple of those);
  dense  - the call works but returns a dense array;
  error  - the call raises, or the library has no such function;
  n/a    - the function takes no array (creation from shapes, dtype functions).

Usage: python cross-library-probe.py LIB OUT.json array-api-2025.12-functions.json
LIB is one of torch-coo, torch-csr, torch-coo-cuda, torch-csr-cuda, scipy-coo, scipy-csr,
pydata-coo, cupyx-csr. Versions used for the RFC: torch at a67a74d3e7, SciPy 1.18.1, PyData
sparse 0.19.3, CuPy 14.2.0, NumPy 2.3.5.

Inputs: a 4x4 float64 array with 6 nonzeros (one row empty); an SPD 4x4 for factorisations;
integer and boolean versions for bitwise and logical functions; complex for real/imag/conj; a
1-D vector for vector functions. Array API names are used, with each library's own spelling
where it differs (see REN for torch).
"""
import json, sys, traceback
import numpy as np

LIB, OUT = sys.argv[1], sys.argv[2]
AA = json.load(open(sys.argv[3]))
D = np.array([[1., 0, 2, 0], [0, 0, 3, 0], [0, 0, 0, 0], [4, 5, 0, 6]])
SPD = np.array([[4., 1, 0, 0], [1, 3, 0, 0], [0, 0, 2, 0], [0, 0, 0, 5]])
V = np.array([1., 0, 2, 0])

# ---- library adapters: make(np_array) -> sparse; ns -> namespace object; issparse(x)
if LIB.startswith("torch"):
    import torch
    lay = LIB.split("-")[1]
    dev = "cuda" if LIB.endswith("-cuda") else "cpu"
    def make(a):
        t = torch.tensor(a, device=dev)
        if t.ndim == 1 or lay == "coo":
            return t.to_sparse()
        return t.to_sparse_csr()
    def issparse(x):
        return isinstance(x, torch.Tensor) and x.layout != torch.strided
    def isdense(x):
        return isinstance(x, torch.Tensor) and x.layout == torch.strided
    class NS:
        pass
    ns = NS()
    REN = {"concat": "cat", "permute_dims": "permute", "bitwise_invert": "bitwise_not", "expand_dims": "unsqueeze",
           "take_along_axis": "take_along_dim", "cumulative_sum": "cumsum", "cumulative_prod": "cumprod",
           "broadcast_arrays": "broadcast_tensors", "unique_values": "unique", "max": "amax", "min": "amin",
           "acos": "acos", "pow": "pow", "atan2": "atan2"}
    REN.update({"repeat": "repeat_interleave", "unstack": "unbind"})
    def get(name, linalg=False):
        if linalg:
            if name == "matrix_transpose":
                return lambda x: x.mT
            return getattr(torch.linalg, name) if hasattr(torch.linalg, name) else getattr(torch, name)
        if name == "astype":
            return lambda x, dt: x.to(dt)
        if name == "matrix_transpose":
            return lambda x: x.mT
        if name == "vecdot":
            return torch.linalg.vecdot
        return getattr(torch, REN.get(name, name))
    F32, I64, BOOL = torch.float32, torch.int64, torch.bool
    def cast(x, dt):
        return x.to(dt)
elif LIB.startswith("scipy") or LIB.startswith("cupyx"):
    if LIB.startswith("scipy"):
        import scipy.sparse as S
        xp = np
        def tonp(a): return a
    else:
        import cupy as xp
        import cupyx.scipy.sparse as S
        def tonp(a): return xp.asarray(a)
    lay = LIB.split("-")[1]
    ctor = {"csr": S.csr_array if hasattr(S, "csr_array") else S.csr_matrix,
            "coo": S.coo_array if hasattr(S, "coo_array") else S.coo_matrix}[lay]
    def make(a):
        return ctor(tonp(a)) if a.ndim == 2 or hasattr(S, "csr_array") else ctor(tonp(a)[None])
    def issparse(x): return S.issparse(x)
    def isdense(x): return isinstance(x, (np.ndarray, xp.ndarray))
    def get(name, linalg=False):
        # a scipy.sparse function, else a method of the array, else the numpy/cupy function
        if linalg:
            import importlib
            spl = importlib.import_module(S.__name__ + ".linalg")
            if hasattr(spl, name):
                return getattr(spl, name)
            return getattr(xp.linalg, name)
        if hasattr(S, name) and callable(getattr(S, name)):
            return getattr(S, name)
        def m(x, *a, **k):
            if hasattr(x, name):
                return getattr(x, name)(*a, **k)
            return getattr(xp, name)(x, *a, **k)
        return m
    F32, I64, BOOL = np.float32, np.int64, np.bool_
    def cast(x, dt): return x.astype(dt)
elif LIB.startswith("pydata"):
    import sparse as S
    def make(a): return S.COO.from_numpy(a)
    def issparse(x): return isinstance(x, S.SparseArray)
    def isdense(x): return isinstance(x, np.ndarray)
    def get(name, linalg=False):
        return getattr(S.linalg if linalg else S, name)
    F32, I64, BOOL = np.float32, np.int64, np.bool_
    def cast(x, dt): return x.astype(dt)

A, P, v = make(D), make(SPD), make(V)
def _try(fn):
    try:
        return fn()
    except Exception as e:  # e.g. cupyx has no int64 sparse arrays
        print("input not constructible:", type(e).__name__, str(e)[:80], file=sys.stderr)
        return None
Ai = _try(lambda: cast(make(np.array([[1, 0, 2, 0], [0, 0, 3, 0], [0, 0, 0, 0], [4, 5, 0, 6]])), I64))
Ab = _try(lambda: cast(make(D != 0), BOOL))
Ac = make(D + 1j * D)

UNARY_BOOL = {"logical_not"}
BINARY_BOOL = {"logical_and", "logical_or", "logical_xor"}
INT_FUNCS = {"bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_left_shift", "bitwise_right_shift", "bitwise_invert"}
BINARY = {"add", "subtract", "multiply", "divide", "floor_divide", "pow", "remainder", "atan2", "copysign", "hypot",
          "logaddexp", "maximum", "minimum", "nextafter", "equal", "not_equal", "greater", "greater_equal", "less",
          "less_equal", "bitwise_and", "bitwise_or", "bitwise_xor", "bitwise_left_shift", "bitwise_right_shift",
          "logical_and", "logical_or", "logical_xor"}
NA = {"arange", "empty", "eye", "full", "linspace", "meshgrid", "ones", "zeros", "can_cast", "finfo", "iinfo",
      "isdtype", "result_type", "__array_namespace_info__", "broadcast_shapes"}
COMPLEX = {"real", "imag", "conj"}


def call(name, f, linalg):
    if name in NA:
        return "n/a"
    if linalg:
        spec = {"cholesky": (P,), "det": (P,), "eigh": (P,), "eig": (P,), "eigvals": (P,), "eigvalsh": (P,), "inv": (P,), "matrix_power": (P, 2),
                "matrix_rank": (P,), "pinv": (P,), "qr": (P,), "slogdet": (P,), "solve": (P, P), "svd": (A,),
                "svdvals": (A,), "cross": (make(np.array([[1., 0, 2], [0, 3, 0]])), make(np.array([[0., 1, 0], [2, 0, 1]]))),
                "outer": (v, v), "vecdot": (A, A), "vector_norm": (A,), "matrix_norm": (A,), "trace": (A,),
                "diagonal": (A,), "matmul": (A, A), "tensordot": (A, A), "matrix_transpose": (A,)}
        return f(*spec[name])
    if name in COMPLEX:
        return f(Ac)
    if name in BINARY:
        x = Ab if name in BINARY_BOOL else Ai if name in INT_FUNCS else A
        return f(x, x)
    if name in UNARY_BOOL:
        return f(Ab)
    if name in INT_FUNCS:
        return f(Ai)
    special = {
        "astype": lambda: f(A, F32), "concat": lambda: f([A, A]), "stack": lambda: f([A, A]),
        "reshape": lambda: f(A, (16,)), "permute_dims": lambda: f(A, (1, 0)), "expand_dims": lambda: f(A, 0),
        "squeeze": lambda: f(make(D[:1]), 0), "flip": lambda: f(A, (0,)) if LIB.startswith("torch") else f(A),
        "roll": lambda: f(A, 1), "broadcast_to": lambda: f(A, (2, 4, 4)), "broadcast_arrays": lambda: f(A, A),
        "moveaxis": lambda: f(A, 0, 1), "repeat": lambda: f(A, 2), "tile": lambda: f(A, (2, 1)),
        "unstack": lambda: f(A), "where": lambda: f(Ab, A, A), "searchsorted": lambda: f(make(np.array([0., 1, 2, 3])), make(np.array([1.5, 2.5]))),
        "take": lambda: f(A, make(np.array([0, 2])) if False else np.array([0, 2]) if not LIB.startswith("torch") else __import__("torch").tensor([0, 2]), axis=0) if not LIB.startswith("torch") else f(A, __import__("torch").tensor([0, 2])),
        "take_along_axis": lambda: f(A, np.zeros((4, 1), dtype=np.int64) if not LIB.startswith("torch") else __import__("torch").zeros((4, 1), dtype=__import__("torch").int64), axis=1) if not LIB.startswith("torch") else f(A, __import__("torch").zeros((4, 1), dtype=__import__("torch").int64), 1),
        "clip": lambda: f(A, 0.5, 3.5), "full_like": lambda: f(A, 7.0), "isin": lambda: f(A, A),
        "matmul": lambda: f(A, A), "tensordot": lambda: f(A, A, 1) if not LIB.startswith("torch") else f(A, A, dims=1),
        "vecdot": lambda: f(A, A), "diff": lambda: f(A), "cumulative_sum": lambda: f(A, axis=0) if not LIB.startswith("torch") else f(A, 0),
        "cumulative_prod": lambda: f(A, axis=0) if not LIB.startswith("torch") else f(A, 0), "from_dlpack": lambda: f(A),
        "asarray": lambda: f(A), "tril": lambda: f(A), "triu": lambda: f(A),
    }
    if name in special:
        return special[name]()
    return f(A)


def classify(r):
    if isinstance(r, (tuple, list)) or type(r).__name__.endswith("Result") or hasattr(r, "_fields"):
        parts = [classify(x) for x in r]
        return "sparse" if all(p == "sparse" for p in parts) else "dense" if "error" not in parts else "error"
    if issparse(r):
        return "sparse"
    if np.isscalar(r) or (hasattr(r, "ndim") and getattr(r, "ndim", 1) == 0):
        return "sparse"   # a scalar result of a reduction: nothing to densify
    if isdense(r):
        return "dense"
    return "dense"


res = {}
for group, names in AA.items():
    linalg = group == "linalg"
    for n in names:
        key = ("linalg." if linalg else "") + n
        if n in NA and not linalg:
            res[key] = ["n/a", ""]
            continue
        try:
            f = get(n, linalg)
        except Exception as e:
            res[key] = ["error", f"missing: {type(e).__name__}"]
            continue
        try:
            r = call(n, f, linalg)
            res[key] = ["n/a", ""] if isinstance(r, str) else [classify(r), type(r).__name__]
        except Exception as e:
            res[key] = ["error", f"{type(e).__name__}: {str(e)[:120]}"]
json.dump(res, open(OUT, "w"), indent=0)
import collections
print(LIB, dict(collections.Counter(v[0] for v in res.values())))
