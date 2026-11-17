# On main: export two programs and save them with their inputs and
# main's outputs. usage: python export-save.py OUTDIR
import sys, torch
o = sys.argv[1]
print("torch", torch.__version__, torch.__file__)


class AbsNet(torch.nn.Module):
    def forward(self, x):
        return torch.abs(x)


class ToCsrNet(torch.nn.Module):
    def forward(self, x):
        return x.to_sparse_csr()


coo = torch.tensor([[0.0, -1.5, 0.0], [2.0, 0.0, -3.0]]).to_sparse_coo()
dense = torch.tensor([[0.0, -1.5, 0.0], [2.0, 0.0, -3.0]])
for name, net, x in (("abs_coo", AbsNet(), coo), ("to_sparse_csr", ToCsrNet(), dense)):
    ep = torch.export.export(net, (x,), strict=True)
    print(f"== {name}\n{ep.graph_module.code}")
    torch.export.save(ep, f"{o}/{name}.pt2")
    y = ep.module()(x)
    print("result:", y)
    torch.save({"x": x, "y": y}, f"{o}/{name}.io.pt")
