# On the deletion experiment's build, with or without the package: load the saved
# programs and run them. usage: python export-load.py DIR   (setting from the environment)
import os, sys, traceback, torch
d = sys.argv[1]
m = sys.modules.get("torch_sparse_ext")
print("torch", torch.__version__, "| package loaded:", m is not None,
      "| uses_aten:", m.uses_aten() if m else None)
for name in ("abs_coo", "to_sparse_csr"):
    print(f"== {name}")
    if not os.path.exists(f"{d}/{name}.pt2"):
        print("not saved: torch.export.save failed on main"); continue
    io = torch.load(f"{d}/{name}.io.pt", weights_only=False)
    try:
        ep = torch.export.load(f"{d}/{name}.pt2")
        print("load: ok")
        print(ep.graph_module.code.strip())
    except Exception as e:  # noqa: BLE001
        print("load: FAILED", type(e).__name__, e); continue
    before = m.calls(reset=True) if m else None
    try:
        y = ep.module()(io["x"])
        same = (y.layout == io["y"].layout and torch.equal(y.to_dense(), io["y"].to_dense()))
        print("run: ok | layout", y.layout, "| equals main:", same)
    except Exception as e:  # noqa: BLE001
        print("run: FAILED", type(e).__name__)
        print("error text:", str(e).splitlines()[0])
        print("".join(traceback.format_exception_only(e)).strip()[:600])
    if m:
        print("counters:", {f: c for f, c in m.calls().items() if any(c.values())})
