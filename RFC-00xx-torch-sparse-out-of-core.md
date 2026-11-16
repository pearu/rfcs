> **Pre-plan for discussion, not yet an RFC for comment.** It proposes a direction
> and lists the questions to settle before the full RFC is written.

# Moving torch.sparse out of pytorch core

**Authors:**
* @pearu


All pytorch line numbers below are at commit
[`a67a74d3e7`](https://github.com/pytorch/pytorch/tree/a67a74d3e71bf2891b0d02b126fe7390090e1f1c)
(2026-11-02). Option letters (A–G) are defined under [Alternatives](#alternatives).

## **Summary**

- torch.sparse has no listed active maintainer; its emeritus maintainers still review
  much of it.
- Proposal: move sparse into a separate project (A), reached through an in-repo
  package first (C). At once: a maintenance policy (E) and a sparse test config (D).
  Alongside: semi-structured (2:4) sparsity moves to TorchAO (F).
- No Python name breaks: torch depends on the new package, so `import torch` brings
  `torch.sparse` as today. One C++ method for c10d backends is deprecated and removed
  (R2→R4).
- Four releases (R0–R3, about eight months), plus R4 for that C++ removal.
- Fallback, if core does not add three stable-ABI pieces: the torch-xpu-ops model (G).
- The asks of core: owners for the new repo; three stable-ABI pieces; building the
  package from a commit pin in CI, as for triton; small PRs that let sparse support a
  new backend.

## **Motivation**

**Maintenance and review come first.**
- All five "Sparse (torch.sparse)" maintainers are emeritus ([persons_of_interest.md:150-156][poi]).
  The "Sparse" merge rule is approved only by four of them ([merge_rules.yaml:159-180][mr]).
- They still review. Of 119 sparse PRs in 12 months, 84 % had a human review, and 50
  (42 %) were reviewed by an emeritus sparse maintainer.[^prs] The rest of the review
  falls on core maintainers.
- Sparse PRs get a first review as fast as other PRs. They may take longer to land
  (median 6 against 4 days; the 95 % interval of the difference is −0.1 to +6.6
  days). Time to land reflects reviewer availability, not difficulty. The cost is who
  reviews.[^prs]
- LLMs make parity PRs cheap. Holes in sparse support (MPS ops, CPU/CUDA parity) are
  easy to find and fill. Core reviewers then ask who needs the op, and there is no
  owner to answer or to maintain it afterwards.
- Sparse is small: 1.5 % of issues and 0.4 % of PRs[^issues]; ~27 k lines of C++/CUDA
  in ATen, 3.6 % of it.[^size]

**Users next.** The API is beta ([sparse.md:13-15][beta]); semi-structured and
`torch.masked` are prototype ([sparse.md:72][semi], [masked.md:18-19][masked]). It is
used: PyG, DGL, TorchAO and `torch_sparse` call it. Public Python files that mention
`torch.sparse_coo_tensor`: 29,504; `SparseAdam`: 24,512; `Embedding(` with
`sparse=True`: 11,968.[^cs] So sparse cannot be removed; it can move behind the same
names.

The aim is to keep this work, which users depend on, maintained by owners of its own.

**CI last.** A PR that touches no sparse file spends ~1.9 of its ~119 runner hours of
test jobs on sparse tests, 1.6 %.[^ci] Saving it is a side effect, not the reason.

## **Proposed Implementation**

### What stays in core

| Item | Why it stays |
|---|---|
| Layout values `sparse_coo`, `sparse_csr/csc/bsr/bsc` | part of the stable ABI's enum ([Layout.h:11-21][layout]) |
| Dispatch keys `Sparse`, `SparseCsr` | outside code cannot add a functionality key ([DispatchKey.h:211-213][dk]) |
| `SparseTensorImpl`, `SparseCsrTensorImpl`, with ~20 constructor and accessor ops | core autograd reads them |
| Generic gradient accumulation | picks in-place or out-of-place add by layout ([accumulate_grad.h:188-239][acc]) |
| DDP bucketing of sparse gradients | set in the reducer ([reducer.cpp:91-162][red]) |
| The serialization name `torch._utils._rebuild_sparse_tensor` | saved files name it ([_utils.py:369][rebuild]) |
| `torch.backends.cusparselt` | documented without a prototype label ([backends.md:286-301][cslt]); cuSPARSELt stays a torch wheel dependency ([torch/\_\_init\_\_.py:402][preload]) |
| `torch.masked` | prototype, mostly strided |
| The cuSPARSE handle pool | dense MAGMA code uses it ([MagmaUtils.h:35][magma]); exposed through the stable ABI in R1 |

Everything else moves: the kernels (~26 k lines), autograd for sparse ops,
`torch/sparse/*`, `SparseAdam`, and the c10d sparse kernels ([Ops.cpp:795-801][ops]).

### The migration

pytorch releases about every two months. R0 is the first release after the RFC is
accepted. The new repo is `pytorch/sparse-tensors`, package `sparse_tensors`.

| Release | What changes | `torch.sparse` imports as |
|---|---|---|
| **R0** | E: the policy in `sparse.md` and CONTRIBUTING. D: a `sparse` test config. The new repo gets named owners. F starts; the semi-structured API warns "moving to TorchAO". | today |
| **R1** | C: `torch/sparse/*`, `SparseAdam` and `sparse_adam` move into the in-repo package `sparse_tensors`, laid out as the new repo. | alias |
| **R2** | Kernels move, one family per PR. A family is an op with all its variants (functional, in-place, `out=`). Each PR drops one from ATen and bumps the package's pin. The package builds on the stable ABI and loads through a `torch.backends` entry point. F done. `Backend::allreduce_sparse` deprecated. | dependency of torch, pinned `==`, built by pytorch CI |
| **R3** | The rest: autograd for sparse-only ops, the c10d sparse kernels, the 14 MPS-only kernels. Sparse test files leave pytorch CI. | dependency, `>=` |
| **R4** | `Backend::allreduce_sparse` removed. | dependency |

Users installing the package themselves is not proposed; that is a later decision.

**The alias (R1 on)** re-exports, where each is documented today: `torch.sparse`;
`torch.optim.SparseAdam` ([optim/\_\_init\_\_.py:31][sa]); and
`torch.optim.functional.sparse_adam` ([functional.py:71][fsa]). Names that stay need
none: the `Tensor` methods, `torch.sparse_*_tensor`, `Embedding(sparse=True)` and
`torch.cuda.sparse` ([cuda/sparse.py:1][cudasparse]).

**How the package reaches torch (R2 on).** pytorch CI builds it from a commit pin, as
for triton ([build-triton-wheel.yml][triton], [binary_populate_env.sh:73-87][bpe]), so
dropping a family from ATen and adding it to the package is one PR. At release, the
package is built on torch N−1's stable ABI and released first; torch N pins it. It
skips families that the torch it runs on still has in ATen.

**Why four releases.** No Python name breaks, so the pace is set by the work, not by
deprecation: R1 is mechanical, R2–R3 carry the kernels. The C++ c10d method needs a
release of deprecation (R2 warns, R4 removes).

**XPU and MPS.** 156 of 173 `SparseXPU`, 128 of 139 `SparseCsrXPU`, 147 of 161
`SparseMPS` and 13 of 13 `SparseCsrMPS` entries in `native_functions.yaml` name the
same generic kernel as CPU/CUDA. The package registers each moved family for every key
that names it today. pytorch's XPU and MPS jobs test it during R2–R3; after that, the
package needs XPU and macOS runners, or those maintainers test it.

### What core adds

| Release | Core adds |
|---|---|
| R0 | the `sparse` test config; a marker for the 122 sparse tests in core test files |
| R1 | `sparse_tensors` in the wheel's `packages`, as `functorch` ([pyproject.toml:188][pyproj]); a merge rule and CODEOWNERS. **In the stable ABI: (1) `Scalar` as a kernel argument and for calling ops; (2) accessors and an in-place setter for a sparse tensor's indices, values and nnz; (3) the current cuSPARSE handle.** Any stable-ABI extension with a `Scalar` argument or a vendor library handle needs the same. |
| R2 | a CI job that builds the package from a pin, and the dependency in the wheel requirements; an "install `sparse_tensors`" error when it is absent; a way to skip sparse OpInfo samples without it; Inductor's 2:4 kernel choice through a registration ([mm.py:203-205][inductor]) |
| R3 | an extension point for `Adagrad`'s sparse step |

### New backends: an agreement with core

Core keeps the keys and the TensorImpls, so a backend is a set of kernels registered
from the package. But core must know the backend first. For MPS:

| Core piece | For MPS today |
|---|---|
| The device: allocator, streams, dense kernels | `MPS` |
| `Backend` enum entries and their keys | [Backend.h:44-45][be1], [:102][be2], [:184][be3] |
| The key's name for parsing | [DispatchKey.cpp:353][dkcpp] |
| Layout from backend | [Layout.h:15][c10layout] |
| Python tensor types, `new()`'s allowed keys | [tensor_types.cpp:39][tt], [tensor_new.cpp:566-581][tn] |
| Code generation of the keys | [model.py:309-310][tgm], [gen.py:3054][tgg] |

These exist today for MPS, XPU, HIP and `PrivateUse1`. Another device needs a core PR,
with no kernel code in it. The understanding to ask for:
1. Core accepts PRs whose only purpose is to let the sparse project support a backend.
2. The device's maintainers review them, with a sparse owner as co-reviewer.
3. The sparse project decides which backends to add, by its own bar: user demand and
   an owner who commits.

### Evidence: two kernel families moved

COO `mul` and `addmm`, CPU and CUDA, were removed from ATen and registered from a
stable-ABI package (target 2.13) that `import torch` loads. Measured on two families
only; diffs and results in [`RFC-00xx-assets/`](RFC-00xx-assets/README.md).

- [`pytorch-out-of-tree.diff`](RFC-00xx-assets/pytorch-out-of-tree.diff) deletes 8
  lines of `native_functions.yaml`, nothing else. Each R2 PR looks like this, plus
  deleting the sources. Removing `addmm.out` alone left `torch.addmm` in core: the
  family is the unit.
- [`sparse-ext.diff`](RFC-00xx-assets/sparse-ext.diff), 1028 lines: all of
  `test_sparse.py` gives the same result as main, 5513 passed.
- Its Python route for `addmm` (`torch.library`) loads only stable-ABI code: the
  stable ABI is enough today, at ~3 µs more per call than in-tree (CPU, nnz 100), and
  it takes the GIL.
- Its C++ bridge, `register_addmm.cpp`, is the one non-stable file. It shows what the
  stable ABI lacks: a kernel cannot receive a `Scalar` ([shim_common.cpp:77-186][shim];
  the package's `probe_scalar.cpp` shows the error).
- `addmm` costs the same on CUDA at nnz 10⁶ (1.00×); `mul` is 1.3–4.6× slower.
  `mul.cpp`'s extra copies (no setter) and `addmm_cuda.cu`'s own handle pool are the
  other two pieces.

## **Metrics**

- Sparse PRs and reviews reaching core reviewers: should drop to the part kept in core.
- Sparse share of pytorch CI time: from 1.6 % towards zero after R3.
- No regressions in the 122 sparse tests in core test files, run with and without the
  package.
- Downloads of `sparse_tensors`, once it is published on its own.

## **Drawbacks**

| Risk | If it happens |
|---|---|
| No owners for the new project | A is worse than code in core. Stop at E + D + F, and freeze new backends. |
| Core does not add the three stable-ABI pieces | Fall back to G. Or move kernels as above, at its cost per call. |
| A broken package breaks `import torch` | The package's CI runs `import torch` on every platform; a release blocker. |
| torch depends on a package that depends on torch | pytorch CI builds one more wheel from a pin, on every platform. The release order above is untested, and so is conda. |
| Hidden sparse paths in core (259 files mention sparse) | Found only by the 122 core sparse tests run without the package. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` turns sparse kernels off too | Sparse gets an import-time hook of its own. |

## **Alternatives**

| | Option | What moves | Who reviews sparse PRs | Why it loses, as the destination |
|---|---|---|---|---|
| 0 | As today | nothing | emeritus sparse and core maintainers | no owner for sparse; review falls on emeritus and core maintainers |
| **A** | Separate project | kernels, autograd, Python; core keeps layouts, keys, TensorImpls | the new repo's owners | **proposed** |
| B | Core keeps the five layouts; other backends as plugins | MPS and later backends | core, for CPU/CUDA | CPU/CUDA code and its PRs stay in core with no owner |
| C | The functorch path | Python, into an in-repo package | core | review and CI stay in core; cheap as A's first step |
| D | Sparse tests in their own config | nothing | core | saves only CI; needed by A anyway |
| E | Maintenance mode in core | nothing | core, with a rule to decline parity PRs | bug fixes still need core reviewers; right for now |
| F | Semi-structured to TorchAO | 2:4 ops and classes | TorchAO | covers only 2:4 |
| G | The torch-xpu-ops model | kernel sources, to a repo pytorch pins and builds in tree | new repo for kernels; core for registrations and pin bumps | close second; the fallback |

**A against G.**

| | A | G |
|---|---|---|
| Maintenance | Sparse PRs leave core. Coupling is through the aten schemas the package calls (~30 for the two families moved), which change under BC rules. A pin in R2–R3 only. | Registrations stay in core: a new op or backend needs a core PR. Internal ATen changes can break the sparse repo. A pin to bump about weekly: XPU's moved 61 times in 12 months, with 3 reverts.[^xpu] |
| Users | No Python name breaks. Risks: a dependency cycle; a broken package breaks `import torch`. | No change. |
| Effort | The three stable-ABI pieces first. | Files move; a pin and a CMake block ([CMakeLists.txt:1266-1336][xpucmake]). |
| CI | Sparse leaves pytorch's jobs. | Sparse is built in every job. |

G wins on users and effort; A on maintenance. G couples core to sparse for good, and
its weekly pin needs an owner sparse lacks. XPU's model works because Intel staffs it.

## **Prior Art**

- **functorch** moved the other way: a separate repo merged into pytorch. `functorch`
  is now an alias over `torch._functorch` ([functorch/\_\_init\_\_.py:6-7][ft]), the
  route R1 uses.
- **torchaudio**: a maintenance phase, and a stable-ABI wheel that works with later
  torch releases.
- **TorchAO** builds stable-ABI modules with `TORCH_TARGET_VERSION`.
- **triton**: a pinned dependency of torch, built by pytorch CI.
- **torch-xpu-ops**: option G.
- **`torch/_vendor/`**: vendored pure-Python packages ([README.md][vendor]); too small
  a precedent for a module with C++ behind it.
- **`torch.backends` entry points**, [pytorch#122468](https://github.com/pytorch/pytorch/issues/122468),
  made for out-of-tree devices ([torch/\_\_init\_\_.py:3531-3546][autoload]); R2 loads
  the kernels this way.
- **RFC-0050**: cross-repository CI for out-of-tree backends.

## **How we teach this**

- Users keep `torch.sparse`. Nothing changes in their code.
- `sparse.md` gets the maintenance policy: bug fixes; new ops and backends only with
  user demand and an owner who commits.
- The package's docs stay at the same URL.
- The name `sparse_tensors` says what the package holds, and keeps it apart from
  TorchAO's sparsity (pruning). `torchsparse` reads better, but
  `mit-han-lab/torchsparse` uses that import name; `torch_sparse` is
  `rusty1s/pytorch_sparse`, used by PyG.

## **Unresolved questions**

Questions 1–4 and 6–8 are for malfet and amjames; 5 for amjames; 9 for the XPU
maintainers.

1. Is A, through C, the direction you'd back, given that torch keeps
   depending on the package?
2. Who would own the new repo? Without owners we stop at E + D + F.
3. Will core commit to the agreement on new backends?
4. Will pytorch CI build the package from a commit pin during R2–R3, as for
   triton, on every platform torch ships?
5. Will TorchAO take the remaining prototype 2:4 ops and classes (F)?
6. Is four releases the right pace, or should R2 and R3 be one?
7. Will core add the three stable-ABI pieces in R1? Who reviews them?
8. If not, would core take G for sparse, as for XPU? Who would bump the pin
   weekly?
9. The XPU maintainers (EikanWang, gujinghui, guangyey, etaf; [merge_rules.yaml:552-563][xpumr]):
   under A, 156 `SparseXPU` and 128 `SparseCsrXPU` registrations of generic kernels
   move to the package. Would you test them there, or take them into torch-xpu-ops?

[^prs]: GitHub search, 2026-11-16: `repo:pytorch/pytorch is:pr created:2025-11-16..2026-11-16 label:"module: sparse"`,
    and the same with `label:"release notes: sparse"`; against 891 random PRs of the
    window with neither label. Human: not a bot, not the author.
[^issues]: GitHub search, 2026-11-16: `repo:pytorch/pytorch created:2025-11-16..2026-11-16`,
    `is:issue` or `is:pr`, with and without the sparse labels.
[^size]: Raw lines at the commit above of `aten/src/ATen/native/sparse/**` and ATen's
    other `Sparse*` files, against all C++/CUDA in ATen.
[^cs]: GitHub code search, 2026-11-03, `language:Python`: `"torch.sparse_coo_tensor"`,
    `"SparseAdam"`, `"Embedding(" "sparse=True"`. Approximate; counts copies.
[^ci]: 99 landed PRs touching no sparse file, 2026-08-05 to 2026-11-02: job times
    from the GitHub Actions API, per-file test times (`test-times.json`, 2026-11-02)
    over the test files target determination chose.
[^xpu]: GitHub API, 2026-11-16: commits to `third_party/xpu.txt`, 2025-11-16 to 2026-11-16.

## Resolution
We decided to do it. X% of the engineering team actively approved of this change.

### Level of Support
Choose one of the following:
* 1: Overwhelming positive feedback.
* 2: Positive feedback.
* 3: Majority Acceptance, with conflicting Feedback.
* 4: Acceptance, with Little Feedback.
* 5: Unclear Resolution.
* 6: RFC Rejected.
* 7: RFC Rejected, with Conflicting Feedback.


#### Additional Context
Some people were in favor of it, but some people didn’t want it for project X.


### Next Steps
Will implement it. 


#### Tracking issue
<github issue URL>


#### Exceptions
Not implementing on project X now. Will revisit the decision in 1 year.

[poi]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/community/persons_of_interest.md?plain=1#L150-L156
[mr]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.github/merge_rules.yaml#L159-L180
[xpumr]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.github/merge_rules.yaml#L552-L563
[beta]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/sparse.md?plain=1#L13-L15
[semi]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/sparse.md?plain=1#L72
[masked]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/masked.md?plain=1#L18-L19
[layout]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/headeronly/core/Layout.h#L11-L21
[dk]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/c10/core/DispatchKey.h#L211-L213
[acc]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/autograd/functions/accumulate_grad.h#L188-L239
[red]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/distributed/c10d/reducer.cpp#L91-L162
[rebuild]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_utils.py#L369
[cslt]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/backends.md?plain=1#L286-L301
[preload]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/__init__.py#L402
[ops]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/distributed/c10d/Ops.cpp#L795-L801
[sa]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/optim/__init__.py#L31
[fsa]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/optim/functional.py#L71
[cudasparse]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/cuda/sparse.py#L1
[triton]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.github/workflows/build-triton-wheel.yml
[bpe]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.ci/pytorch/binary_populate_env.sh#L73-L87
[pyproj]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/pyproject.toml#L188
[inductor]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_inductor/kernel/mm.py#L203-L205
[be1]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/c10/core/Backend.h#L44-L45
[be2]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/c10/core/Backend.h#L102
[be3]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/c10/core/Backend.h#L184
[dkcpp]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/c10/core/DispatchKey.cpp#L353
[c10layout]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/c10/core/Layout.h#L15
[tt]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/utils/tensor_types.cpp#L39
[tn]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/utils/tensor_new.cpp#L566-L581
[tgm]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torchgen/model.py#L309-L310
[tgg]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torchgen/gen.py#L3054
[magma]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/cuda/linalg/MagmaUtils.h#L35
[shim]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/shim_common.cpp#L77-L186
[xpucmake]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/caffe2/CMakeLists.txt#L1266-L1336
[ft]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/functorch/__init__.py#L6-L7
[autoload]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/__init__.py#L3531-L3546
[vendor]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_vendor/README.md
