"""Gradcheck of log1p and sparse.softmax on COO, FakeTensorMode calls of
to_sparse_csr on dense and sin on CSR; counters before and after each."""
import sys, json, torch
from torch._subclasses.fake_tensor import FakeTensorMode
sx = sys.modules.get("torch_sparse_ext")
print(f"torch {torch.__file__}; package loaded: {sx is not None}")
gc = torch.sparse.as_sparse_gradcheck(torch.autograd.gradcheck)


def calls():
    if not sx:
        return {}
    return {f: {k: v for k, v in d.items() if v} for f, d in sx.calls().items() if any(d.values())}


def run(tag, f):
    sx and sx.calls(reset=True)
    try:
        r = f()
        print(f"  {tag}: {r}")
    except Exception as e:  # noqa: BLE001
        print(f"  {tag}: {type(e).__name__}: {str(e).splitlines()[0][:250]}")
    print(f"     package calls: {json.dumps(calls())}")


for dev in ("cpu", "cuda"):
    print(f"-- {dev}")
    d = torch.tensor([[0., 1.5, 0.], [2., 0., 3.]], dtype=torch.float64, device=dev)
    coo = d.to_sparse()
    x = coo.detach().requires_grad_(True)
    run("gradcheck log1p, COO, masked=False", lambda: gc(torch.log1p, (x,), masked=False))
    run("gradcheck sparse.softmax(dim=1), COO, masked=False",
        lambda: gc(lambda t: torch.sparse.softmax(t, 1), (x,), masked=False))
    run("gradcheck sparse.softmax(dim=1), COO, masked=True",
        lambda: gc(lambda t: torch.sparse.softmax(t, 1), (x,), masked=True))
    csr_real = d.to_sparse_csr() if sx else torch.sparse_csr_tensor(
        torch.tensor([0, 1, 3]), torch.tensor([1, 0, 2]), d[d != 0], d.shape, device=dev)

    def fake_to_csr():
        with FakeTensorMode() as m:
            r = m.from_tensor(d).to_sparse_csr()
            return f"{type(r).__name__} layout={r.layout} shape={tuple(r.shape)} device={r.device} nnz={r._nnz()}"

    def fake_sin_csr():
        with FakeTensorMode() as m:
            r = torch.sin(m.from_tensor(csr_real))
            return f"{type(r).__name__} layout={r.layout} shape={tuple(r.shape)} device={r.device} nnz={r._nnz()}"

    def fake_sin_csr_made_in_mode():
        # from_tensor() cannot take a real CSR tensor (main too); make the fake CSR in the mode
        with FakeTensorMode() as m:
            c = m.from_tensor(d).to_sparse_csr()
            r = torch.sin(c)
            return f"{type(r).__name__} layout={r.layout} shape={tuple(r.shape)} device={r.device} nnz={r._nnz()}"

    run("FakeTensorMode dense.to_sparse_csr()", fake_to_csr)
    run("FakeTensorMode torch.sin(from_tensor(CSR))", fake_sin_csr)
    run("FakeTensorMode torch.sin(fake dense.to_sparse_csr())", fake_sin_csr_made_in_mode)
