"""Functional and in-place sin/log1p on each compressed layout, no package."""
import torch
d = torch.tensor([[0., 1.], [2., 3.]])
for lay in (torch.sparse_csr, torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc):
    x = d.to_sparse(layout=lay, blocksize=(1, 1) if lay in (torch.sparse_bsr, torch.sparse_bsc) else None) \
        if lay in (torch.sparse_csc, torch.sparse_bsr, torch.sparse_bsc) else None
    for name in ("sin", "sin_", "log1p", "log1p_"):
        try:
            if x is None:
                x = torch.sparse_csr_tensor(torch.tensor([0, 1, 3]), torch.tensor([1, 0, 1]), torch.tensor([1., 2., 3.]), (2, 2))
            getattr(x.clone(), name)()
            print(f"{lay} {name}: RETURNED")
        except Exception as e:  # noqa: BLE001
            print(f"{lay} {name}: {type(e).__name__}: {str(e).splitlines()[0][:150]}")
