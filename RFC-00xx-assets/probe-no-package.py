"""One call per family and variant on COO and CSR (and dense/CSC for
_to_sparse_csr), CPU and CUDA: the exception type and first line, or the value."""
import os, sys, torch
sx = sys.modules.get("torch_sparse_ext")
print(f"torch {torch.__file__}; package loaded: {sx is not None}; uses_aten: {sx.uses_aten() if sx else None}; "
      f"SPARSETORCH_USE_ATEN={os.environ.get('SPARSETORCH_USE_ATEN')}")


def show(tag, f):
    try:
        r = f()
        desc = f"{r.layout}" if isinstance(r, torch.Tensor) else repr(r)
        print(f"  {tag}: RETURNED {desc}")
    except Exception as e:  # noqa: BLE001
        print(f"  {tag}: {type(e).__name__}: {str(e).splitlines()[0][:220]}")


for dev in ("cpu", "cuda"):
    d = torch.tensor([[0., 1., 0.], [2., 0., 3.]], device=dev)
    coo = torch.sparse_coo_tensor(d.nonzero().t(), d[d != 0], d.shape).coalesce()
    i = torch.tensor([0, 1, 3], device=dev); j = torch.tensor([1, 0, 2], device=dev)
    csr = torch.sparse_csr_tensor(i, j, d[d != 0], d.shape)
    csc = torch.sparse_csc_tensor(torch.tensor([0, 1, 2, 3], device=dev), torch.tensor([1, 0, 1], device=dev),
                                  torch.tensor([2., 1., 3.], device=dev), d.shape)
    print(f"-- {dev}")
    for name in ("sin", "abs", "log1p"):
        for lay, x in (("coo", coo), ("csr", csr)):
            show(f"{name}({lay})", lambda: getattr(torch, name)(x))
            show(f"{name}_({lay})", lambda: getattr(x.clone(), name + "_")())
            show(f"{name}.out({lay})", lambda: getattr(torch, name)(x, out=x.clone()))
    show("sparse.softmax(coo)", lambda: torch.sparse.softmax(coo, 1))
    show("sparse.softmax(csr)", lambda: torch.sparse.softmax(csr, 1))
    show("_sparse_softmax_backward_data(coo)", lambda: torch.ops.aten._sparse_softmax_backward_data(coo, coo, 1, coo))
    show("to_sparse_csr(dense)", lambda: d.to_sparse_csr())
    show("to_sparse_csr(coo)", lambda: coo.to_sparse_csr())
    show("to_sparse_csr(csr)", lambda: csr.to_sparse_csr())
    show("to_sparse_csr(csc)", lambda: csc.to_sparse_csr())
    show("_to_sparse_csr(dense)", lambda: torch.ops.aten._to_sparse_csr(d))
    show("to_sparse(layout=csr)(dense)", lambda: d.to_sparse(layout=torch.sparse_csr))
