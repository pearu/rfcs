# The package's mul / addmm against dense references, over the cases the in-tree
import warnings; warnings.filterwarnings("ignore")
# kernels handle (test_sparse_mul is skipped on Linux, so the test suite does not cover mul).
import itertools, torch
torch.manual_seed(0)
fails = 0; n = 0

def check(name, got, want):
    global fails, n
    n += 1
    g = got.to_dense() if got.layout == torch.sparse_coo else got
    try:
        torch.testing.assert_close(g, want, atol=1e-5, rtol=1e-5, check_dtype=True)
    except AssertionError as e:
        fails += 1; print("FAIL", name, str(e).splitlines()[0])

def rs(shape, sdim, nnz, dtype, dev, coalesced):
    idx = torch.stack([torch.randint(0, max(s, 1), (nnz,)) for s in shape[:sdim]])
    vals = torch.randn(nnz, *shape[sdim:], dtype=torch.double).to(dtype)
    t = torch.sparse_coo_tensor(idx, vals, shape, device=dev)
    return t.coalesce() if coalesced else t

for dev, dtype, coal in itertools.product(("cpu", "cuda"), (torch.float64, torch.complex128, torch.float32), (True, False)):
    tag = f"{dev} {dtype} coalesced={coal}"
    for shape, sdim in (((5, 6), 2), ((4, 5, 3), 2), ((3, 4, 5, 2), 3)):
        a = rs(shape, sdim, 7, dtype, dev, coal); b = rs(shape, sdim, 9, dtype, dev, coal)
        check(f"{tag} s*s {shape}", a * b, a.to_dense() * b.to_dense())
        d = torch.randn(shape, dtype=torch.double, device=dev).to(dtype)
        check(f"{tag} s*d {shape}", a * d, a.to_dense() * d)
        check(f"{tag} d*s {shape}", d * a, d * a.to_dense())
        check(f"{tag} s*2.5 {shape}", a * 2.5, a.to_dense() * 2.5)
        d1 = torch.randn(shape[-1], dtype=torch.double, device=dev).to(dtype)  # dense broadcast over last dim
        check(f"{tag} s*d[-1] {shape}", a * d1, a.to_dense() * d1)
        db = torch.randn((2,) + shape, dtype=torch.double, device=dev).to(dtype)  # dense has a batch dim
        check(f"{tag} s*d[batch] {shape}", a * db, a.to_dense() * db)
        c = a.clone(); c.mul_(b); check(f"{tag} s.mul_(s) {shape}", c, a.to_dense() * b.to_dense())
        z = torch.sparse_coo_tensor(torch.zeros(sdim, 0, dtype=torch.long), torch.zeros((0,) + shape[sdim:], dtype=dtype), shape, device=dev)
        check(f"{tag} s*empty {shape}", a * z, torch.zeros(shape, dtype=dtype, device=dev))
        s0 = torch.tensor(3.0, dtype=dtype, device=dev).to_sparse()
        check(f"{tag} s*0dim-sparse {shape}", a * s0, a.to_dense() * 3)
    # addmm
    for (m, k, nn), beta, alpha in itertools.product(((6, 7, 5), (6, 7, 0), (0, 7, 5)), (0, 1, 0.5), (1, 0.7)):
        S = rs((m, k), 2, 12 if m else 0, dtype, dev, coal)
        D = torch.randn(k, nn, dtype=torch.double, device=dev).to(dtype)
        T = torch.randn(m, nn, dtype=torch.double, device=dev).to(dtype)
        want = beta * T + alpha * (S.to_dense() @ D)
        check(f"{tag} addmm {m,k,nn} b={beta} a={alpha}", torch.addmm(T, S, D, beta=beta, alpha=alpha), want)
        out = torch.empty(0, dtype=dtype, device=dev)
        torch.addmm(T, S, D, beta=beta, alpha=alpha, out=out); check(f"{tag} addmm out=", out, want)
        T2 = T.clone(); T2.addmm_(S, D, beta=beta, alpha=alpha); check(f"{tag} addmm_", T2, want)
        check(f"{tag} addmm bcast", torch.addmm(T[:1] if m else T, S, D, beta=beta, alpha=alpha),
              beta * (T[:1] if m else T) + alpha * (S.to_dense() @ D))
        check(f"{tag} addmm D.T", torch.addmm(T, S, D.t().contiguous().t(), beta=beta, alpha=alpha), want)
    if dtype.is_complex:
        S = rs((6, 7), 2, 12, dtype, dev, coal); D = torch.randn(7, 5, dtype=dtype, device=dev); T = torch.randn(6, 5, dtype=dtype, device=dev)
        check(f"{tag} addmm complex alpha/beta", torch.addmm(T, S, D, beta=0.5 - 1j, alpha=2 + 1j), (0.5 - 1j) * T + (2 + 1j) * (S.to_dense() @ D))

for dtype in (torch.float16, torch.bfloat16, torch.int64):
    S = rs((6, 7), 2, 12, dtype, "cpu", True); D = torch.randn(7, 5).to(dtype); T = torch.randn(6, 5).to(dtype)
    want = torch.addmm(T.double(), S.to_dense().double(), D.double()).to(dtype)
    got = torch.addmm(T, S, D)
    n += 1
    if not torch.allclose(got.double(), want.double(), atol=0.1, rtol=0.05):
        fails += 1; print("FAIL cpu addmm", dtype)
    check(f"cpu mul {dtype}", S * S, S.to_dense() * S.to_dense())
print(f"{n - fails}/{n} checks passed")
