# On main, or on the compiler-stack build with the package: AOTInductor-compile a module
# whose forward is x.to_sparse(), for CPU and CUDA. usage: python aoti-compile.py OUTDIR TAG
import glob, os, sys, zipfile, torch
o, tag = sys.argv[1], sys.argv[2]
m = sys.modules.get("torch_sparse_ext")
print("torch", torch.__version__, "| package loaded:", m is not None)


class ToSparse(torch.nn.Module):
    def forward(self, x):
        return x.to_sparse()


for dev in ("cpu", "cuda"):
    x = torch.tensor([[0.0, -1.5, 0.0], [2.0, 0.0, -3.0]], device=dev)
    path = f"{o}/{tag}-{dev}.pt2"
    try:
        ep = torch.export.export(ToSparse(), (x,), strict=True)
        print(f"== {dev}: exported graph\n{ep.graph_module.code.strip()}")
        p = torch._inductor.aoti_compile_and_package(ep, package_path=path)
        print(f"== {dev}: compiled {p}")
    except Exception as e:  # noqa: BLE001
        print(f"== {dev}: FAILED {type(e).__name__}: {str(e)[:1500]}")
        continue
    with zipfile.ZipFile(path) as z:
        cpps = [n for n in z.namelist() if n.endswith(".cpp") or n.endswith(".h")]
        print("archive sources:", cpps)
        for n in cpps:
            src = z.read(n).decode(errors="replace")
            hits = [l.strip() for l in src.splitlines() if "to_sparse" in l]
            if hits:
                print(f"   {n}:")
                for l in hits:
                    print("      ", l[:300])
    if m:
        print("counters during compile:", {f: c for f, c in m.calls().items() if any(c.values())})
