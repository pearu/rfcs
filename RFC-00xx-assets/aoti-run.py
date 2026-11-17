# On the compiler-stack build, with or without the package: load and run the
# AOTInductor packages. usage: python aoti-run.py DIR   (setting from the environment)
import glob, sys, torch
d = sys.argv[1]
m = sys.modules.get("torch_sparse_ext")
print("torch", torch.__version__, "| package loaded:", m is not None,
      "| uses_aten:", m.uses_aten() if m else None)
for path in sorted(glob.glob(f"{d}/*.pt2")):
    dev = "cuda" if path.endswith("-cuda.pt2") else "cpu"
    x = torch.tensor([[0.0, -1.5, 0.0], [2.0, 0.0, -3.0]], device=dev)
    print(f"== {path.split('/')[-1]}")
    if m:
        m.calls(reset=True)
    try:
        f = torch._inductor.aoti_load_package(path)
        y = f(x)
        y = y[0] if isinstance(y, (list, tuple)) else y
        ok = y.layout == torch.sparse_coo and torch.equal(y.to_dense(), x)
        print("run: ok | layout", y.layout, "| device", y.device, "| equals eager x.to_sparse() on main:", ok)
    except Exception as e:  # noqa: BLE001
        print("run: FAILED", type(e).__name__)
        print("error text:", str(e)[:1500])
    if m:
        print("counters:", {f: c for f, c in m.calls().items() if any(c.values())})
