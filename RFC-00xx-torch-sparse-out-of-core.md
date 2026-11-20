> **Pre-plan for discussion, not yet an RFC for comment.** It proposes a direction
> and lists the questions to settle before the full RFC is written.

# Moving torch.sparse out of pytorch core

**Authors:**
* @pearu


All pytorch line numbers below are at commit
[`a67a74d3e7`](https://github.com/pytorch/pytorch/tree/a67a74d3e71bf2891b0d02b126fe7390090e1f1c)
(2026-11-02). Option letters (A–G) are defined under [Alternatives](#alternatives).

## **Summary**

- Core no longer works on torch.sparse. Its five maintainers are emeritus, and new
  sparse work lands only with user demand and an owner who commits.
- Proposal: sparse becomes a separate project with owners and governance of its own
  (A), reached through an in-repo package first (C). At once: a maintenance policy (E)
  and a sparse test config (D). Alongside: semi-structured (2:4) sparsity moves to
  TorchAO (F). If too few people commit to own the project, sparse stays in core under
  E + D + F.
- User stories: at least 1,798 public projects call torch.sparse, 525 of them in
  scientific computing, mostly with COO and CSR. The demand not met is MPS, solvers,
  autograd and missing ops. The stories set the new project's priorities, and it keeps
  them current from R0.
- The asks of core: owners for the new repo; four stable-ABI pieces; building the
  package from a commit pin in CI, as for triton; small PRs that let sparse support a
  new backend. Fallback, if core does not add the stable-ABI pieces: the
  torch-xpu-ops model (G).
- No Python name breaks: torch depends on the new package, so `import torch` brings
  `torch.sparse` as today. Four releases (R0–R3, about eight months), plus R4 for the
  last ATen copies.
- The technical goals of sparse (new backends, parity, a stable-ABI package) can be
  pursued with or without this proposal. It decides who owns and reviews that work.
  How the move works is in the appendix, [Technical feasibility](#appendix-technical-feasibility).

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
- The ownership question blocks reviews. [pytorch#201801](https://github.com/pytorch/pytorch/pull/201801) (open
  since 2026-10-23, +53/−6) adds an unsafe CSR/BSR constructor on MPS, a first step
  toward sparse CSR on MPS for a downstream library. A core reviewer asked who would
  maintain MPS sparse parity; the sparse reviewer could not commit to it.[^201801]
  Whether such a PR lands depends on who owns sparse backends, which is what this
  proposal decides.
- Sparse is small: 1.5 % of issues and 0.4 % of PRs[^issues]; ~27 k lines of C++/CUDA
  in ATen, 3.6 % of it.[^size]

**Users next.** The API is beta ([sparse.md:13-15][beta]); semi-structured and
`torch.masked` are prototype ([sparse.md:72][semi], [masked.md:18-19][masked]). It is
used: PyG, DGL, TorchAO and `torch_sparse` call it. Public Python files that mention
`torch.sparse_coo_tensor`: 29,504; `SparseAdam`: 24,512; `Embedding(` with
`sparse=True`: 11,968.[^cs] At least 1,798 public projects call it outside their tests,
1,052 of them maintained ([User stories](#user-stories-r0)). So sparse cannot be
removed; it can move behind the same names.

The aim is to keep this work, which users depend on, maintained by owners of its own.

**CI last.** A PR that touches no sparse file spends ~1.9 of its ~119 runner hours of
test jobs on sparse tests, 1.6 %.[^ci] Saving it is a side effect, not the reason.

## **Proposed Implementation**

### Core no longer works on sparse

This is already the practice, and it has been stated in public: "we have avoided adding
sparse tensor coverage for MPS because we are not actively working on
torch.sparse".[^112849] This proposal writes it down as a policy (E), from R0, in
`sparse.md` and CONTRIBUTING:
- bug fixes are reviewed as today;
- new ops and new backends land only with user demand and an owner who commits to
  maintain them;
- new features should preferably land in the new project, not in core.

An early proposal, still a draft ([pytorch/rfcs#6][rfc3], 2020), aimed at a sparse
tensor that is a drop-in replacement for a dense one in every operation. A core reviewer
knew of no actively developed project that uses compressed layouts on MPS; a user
answered with one in the same thread.[^201801] The user stories below find 52 projects
whose sparse code names MPS, and three that route their sparse code around MPS, one of
them "until sparse works on MPS".

### Governance, and the path to community maintenance

Before the move starts (R0), the new project needs named owners who commit to review
and release it, and a rule for who breaks ties on design decisions (question 2).
Enough interest means enough owners to review each other's PRs, so that no sparse PR
waits on core, and user stories behind the project's priorities.

If that is not met at R0, nothing moves: sparse stays in core under E + D + F, and the
2:4 code still moves to TorchAO. That is a sound end state, not a failure. The
question can be reopened when owners come forward.

**The design document (R0).** The new repo starts with one. Invariant checks stay off
by default for performance, yet crashes on invalid tensors keep being filed as bugs, and
PRs keep adding checks that already exist.[^inv] The design document settles this once.
It says why the checks are off; that a bug report starts with a reproducer that still
fails with checks enabled (`torch.sparse.check_sparse_tensor_invariants`); and that AI
agents must read it before they file issues or PRs.

### User stories (R0)

Public evidence shows who uses torch.sparse and what for. It was collected from code on
GitHub, every PyPI package changed in the last 24 months, every conda-forge package
that depends on pytorch, papers, forums and pytorch issues. Method, tables and limits:
[Appendix: User stories in detail](#appendix-user-stories-in-detail).

- **How many.** At least 1,798 public projects call torch.sparse outside their tests.
  1,052 of them had activity in the last 12 months, and 575 are published on PyPI or
  conda-forge. Of the PyPI packages that declare torch, 2.5 % call it in their own code
  (516 of 20,661).
- **Who.** 1,101 projects are in deep learning and 525 in scientific computing; the
  scientific ones are more often maintained (79 % against 49 %). 576 belong to
  organizations and 1,132 to personal accounts. 9 of the 10 projects with a hundred or
  more PyPI dependents are organizations'.
- **Layouts and devices.** COO is used by 1,203 projects and CSR by 443; CSC, BSR, BSC
  and 2:4 by 32 to 97 each. Most code follows its input's device (1,267 projects); 727
  name CUDA, 52 MPS, 15 ROCm and 11 XPU. Inferred from the code, 625 projects run sparse
  matmul in CUDA or device-generic code, which on CUDA goes through cuSPARSE, and 32 use
  2:4 through cuSPARSELt or CUTLASS.
- **The strongest stories,** by use, maintenance and dependents, each read by hand:
  - deep learning: sentence-transformers (sparse encoders), torchmetrics, peft, PyG
    (sparse adjacency), DeepSpeed (sparse gradients), pytorch3d (mesh Laplacians);
  - scientific computing: POT (optimal transport), linear_operator (Gaussian
    processes), cellpose (cell segmentation), pyfixest (econometrics), cvxpylayers
    (differentiable optimisation), pypose (bundle adjustment with CSR and BSR).

  In many of them sparse is one optional component, and a few use it only to build or
  ship data.
- **Demand not met.**
  - MPS: suite2p, instanseg and omicverse route their sparse code around it, and pypose's bundle
    adjustment asks for CSR and BSR on MPS ([pytorch#201801][bae]).
  - Solvers: `spsolve` runs only in CUDA builds with cuDSS, and on XPU
    ([native_functions.yaml:14698-14702][spsolve],
    [SparseCsrTensorMath.cu:717-721][spsolve-cudss]); torch-sla ships its own
    differentiable solvers.
  - Forum questions and issues name missing ops (18 questions), autograd through
    sparse inputs (12), performance (9), solvers (5), fp16 and bf16, and MPS.
    [pytorch#10043][usecases], open since 2018, collects use cases, such as row
    slicing of CSR tensors on GPU in single-cell biology ([comment][singlecell]).
- **Signs of removal.** In 2026, TorchAO removed its sparsity package "due to lack of
  widespread usage of this feature" ([pytorch/ao#4728][ao4728]); torch keeps the
  semi-structured kernels it called and the BSR layout and Triton kernel that TorchAO's
  own copy came from, and TorchAO keeps a float8 2:4 format in its quantization. This
  bears on F, which stays an open point (question 5). xformers removed its 2:4 code and
  points users to torch.sparse. NeMo and PhysicsNeMo dropped their sparse paths in
  clean-ups.
- **Against linalg and nested tensors.** In equal-size code-search samples, 959 projects
  call sparse and 1,903 call `torch.linalg`; in a sample half that size, 169 call nested
  tensors. Sparse has 1.4× linalg's ATen code and about half its issues and PRs. Sparse
  pays off only where data are sparse enough: published measurements find gains at very
  high sparsity, with block or 2:4 structure and the right layout, and none at moderate
  sparsity. So a user base smaller than linalg's, and concentrated where data are
  sparse, is expected.
- **Limits.** Code search is sampled and sees only public default branches, so the
  counts are lower bounds; private users show only through issues, forums and papers.
  Field and device come from rules, except for the 50 projects read by hand.

The new project keeps the stories current: it recounts them at R0 and at each release
with the same method, and its design document ranks the priorities, and the backends
it adds, from them.

### The asks of core

1. **A new repo with owners**, `pytorch/sparsetorch`, under the governance above
   (question 2).
2. **Four stable-ABI pieces in R1**: `Scalar`, accessors and a setter for a sparse
   tensor's parts, the current cuSPARSE handle, and typed exceptions
   ([A.4](#a4-what-core-adds), question 7). Without them, the fallback is G
   (question 8).
3. **A CI build of the package from a commit pin** during R2–R3, as for triton
   ([A.1](#a1-how-the-package-reaches-torch), question 4).
4. **New backends.** A backend needs a small core PR with no kernel code in it
   ([A.5](#a5-new-backends-the-core-pieces)). The understanding to ask for (question 3):
   1. Core accepts PRs whose only purpose is to let the sparse project support a backend.
   2. The device's maintainers review them, with a sparse owner as co-reviewer.
   3. The sparse project decides which backends to add, by its own bar: user demand and
      an owner who commits.
5. **AOTInductor's C shims of moved ops call the dispatcher** (R2, a torchgen change),
   so that compiled models keep loading and reach the package
   ([A.2](#a2-how-torch-calls-the-package-and-the-hard-cases)). So far this concerns
   only `_to_sparse`.

### What stays in core

Layout values and dispatch keys; `SparseTensorImpl` and `SparseCsrTensorImpl`, with
the 29 entries that build, read or copy a sparse tensor; autograd formulas for sparse
ops; generic gradient accumulation; DDP bucketing of sparse gradients; the
serialization name; `torch.backends.cusparselt`; `torch.masked`; the cuSPARSE handle
pool; the c10d sparse allreduce. Reasons and sources: [A.3](#a3-what-stays-in-core-and-why).

Everything else moves: the kernels (~26 k lines), `torch/sparse/*` and `SparseAdam`.

### The migration

pytorch releases about every two months. R0 is the first release after the RFC is
accepted. The new repo is `pytorch/sparsetorch`, package `sparsetorch`.

**Design rule.** Move code as close to verbatim as possible, with its existing bugs,
issues and limitations. Keep the existing tests passing. Make each step small enough
for a human reviewer to follow. A bug found during the move gets its own PR, after the
move.

| Release | What changes | `torch.sparse` imports as |
|---|---|---|
| **R0** | E: the policy in `sparse.md` and CONTRIBUTING. D: a `sparse` test config. Governance is decided (question 2); without owners, the plan stops here, at E + D + F. User stories are recounted. The new repo gets named owners and a design document. F starts; the semi-structured API warns "moving to TorchAO". | today |
| **R1** | C: `torch/sparse/*`, `SparseAdam` and `sparse_adam` move into the in-repo package `sparsetorch`, laid out as the new repo. | alias |
| **R2** | Kernels move, one family per PR. A family is an op with all its variants (functional, in-place, `out=`). Each PR registers one family from the package over ATen's kernels and bumps the package's pin. ATen's copy stays, frozen, behind a switch. It is deleted only after a release in which the package's kernel shipped and passed the sparse tests. The package builds on the stable ABI and loads through a `torch.backends` entry point. F done. | dependency of torch, pinned `==`, built by pytorch CI |
| **R3** | The rest, the 14 MPS-only kernels among them. ATen copies of R2's families deleted. Sparse test files leave pytorch CI. | dependency, `>=` |
| **R4** | ATen copies of R3's families deleted. | dependency |

Users installing the package themselves is not proposed; that is a later decision.

**The alias (R1 on).** `torch.sparse`, `torch.optim.SparseAdam` and
`torch.optim.functional.sparse_adam` re-export the package; names that stay in core need
none ([A.1](#a1-how-the-package-reaches-torch)).

**Why four releases.** No Python name breaks, so the pace is set by the work, not by
deprecation: R1 is mechanical, R2–R3 carry the kernels.

**Every release is a safe place to stop, and to reconsider the plan.** Until a family's
ATen copy is deleted, the switch can bring ATen's copies back for any release, for
example when the package isn't ready. If plans change, whether to continue, change
course or move code back is decided then, under the governance of question 2, starting
from a working torch.

### Who could do the work

These people have a record in the code that moves. They are suggested candidates; none
has agreed to take part yet:
- pearu: sparse layouts and kernels ([pytorch#129645](https://github.com/pytorch/pytorch/pull/129645)), and stable-ABI work: [pytorch#167923](https://github.com/pytorch/pytorch/pull/167923)
  and the port of torchaudio to the stable ABI ([pytorch/audio#4137](https://github.com/pytorch/audio/pull/4137), [pytorch/audio#4142](https://github.com/pytorch/audio/pull/4142)).
- amjames: sparse compressed kernels, and a sparse code owner ([pytorch#104062](https://github.com/pytorch/pytorch/pull/104062)).
- nikitaved: sparse COO and compressed kernels on CPU and CUDA ([pytorch#72710](https://github.com/pytorch/pytorch/pull/72710),
  [pytorch#202778](https://github.com/pytorch/pytorch/pull/202778)).
- alexsamardzic: 2:4 semi-structured sparsity and CUTLASS kernels, in pytorch and
  TorchAO ([pytorch#123473](https://github.com/pytorch/pytorch/pull/123473), [pytorch/ao#1940](https://github.com/pytorch/ao/pull/1940)).
- Isalia20: the MPS sparse kernels ([pytorch#167013](https://github.com/pytorch/pytorch/pull/167013), [pytorch#169125](https://github.com/pytorch/pytorch/pull/169125), [pytorch#169368](https://github.com/pytorch/pytorch/pull/169368)).

Help also comes from the community. In the last 12 months, 67 other people opened sparse
PRs, and 71 opened or commented on sparse issues.[^community]

## **Metrics**

- Sparse PRs and reviews reaching core reviewers: should drop to the part kept in core.
- User stories behind the priorities: every op or backend added after R0 cites at least
  one actively maintained public project that needs it. Stories are recounted at each
  release, by layout and device, with the method of [B.1](#b1-method), against this
  RFC's count: 1,798 projects, 1,052 maintained.
- Sparse share of pytorch CI time: from 1.6 % towards zero after R3.
- No regressions in the 122 sparse tests in core test files, run with and without the
  package.
- Downloads of `sparsetorch`, once it is published on its own.

## **Drawbacks**

| Risk | If it happens |
|---|---|
| No owners for the new project | Nothing moves; sparse stays in core under E + D + F (see Governance). |
| Work stalls or plans change (R2–R3) | Each release is still working. Families not yet deleted from ATen can switch back. The next step is decided under question 2. |
| Core does not add the four stable-ABI pieces | Fall back to G. Or move kernels through the Python route ([A.9](#a9-evidence-kernel-families-moved)), at its cost per call. |

## **Alternatives**

| | Option | What moves | Who reviews sparse PRs | Why it loses, as the destination |
|---|---|---|---|---|
| 0 | As today | nothing | emeritus sparse and core maintainers | no owner for sparse; review falls on emeritus and core maintainers |
| **A** | Separate project | kernels, Python; core keeps layouts, keys, TensorImpls, autograd formulas | the new repo's owners | **proposed** |
| B | Core keeps the five layouts; other backends as plugins | MPS and later backends | core, for CPU/CUDA | CPU/CUDA code and its PRs stay in core with no owner |
| C | The functorch path | Python, into an in-repo package | core | review and CI stay in core; cheap as A's first step |
| D | Sparse tests in their own config | nothing | core | saves only CI; needed by A anyway |
| E | Maintenance mode in core | nothing | core, with a rule to decline parity PRs | bug fixes still need core reviewers; right for now |
| F | Semi-structured to TorchAO | 2:4 ops and classes | TorchAO | covers only 2:4 |
| G | The torch-xpu-ops model | kernel sources, to a repo pytorch pins and builds in tree | new repo for kernels; core for registrations and pin bumps | close second; the fallback |

A against G, in detail: [A.7](#a7-a-against-g).

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
- The package's docs stay at the same URL.
- The name `sparsetorch` follows `functorch`: a package for torch only. It is free on
  PyPI and conda-forge, and distinct from `torchsparse` (the import name of
  `mit-han-lab/torchsparse`) and `torch_sparse` (`rusty1s/pytorch_sparse`, used by
  PyG).

## **Unresolved questions**

The first readers are malfet and amjames; the Ask lines name who else can answer each
question.

1. Is A, through C, the direction you'd back, given that torch keeps
   depending on the package?\
   Ask: albanD, malfet
2. Governance of the new repo: who owns it, and who breaks ties on design decisions?
   Decided at R0. Without owners, sparse stays in core under E + D + F.\
   Ask: albanD, malfet, amjames, eqy, nikitaved
3. Will core commit to the agreement on new backends?\
   Ask: malfet, EikanWang
4. Will pytorch CI build the package from a commit pin during R2–R3, as for
   triton, on every platform torch ships?\
   Ask: atalman, malfet
5. Will TorchAO take the remaining prototype 2:4 ops and classes (F)?\
   Ask: jerryzh168, vkuzo, amjames
6. Is four releases the right pace, or should R2 and R3 be one?\
   Ask: atalman, malfet
7. Will core add the four stable-ABI pieces ([A.4](#a4-what-core-adds)) in R1? Who reviews them?\
   Ask: janeyx99, albanD; for the cuSPARSE handle, also eqy, nikitaved
8. If not, would core take G for sparse, as for XPU? Who would bump the pin
   weekly?\
   Ask: malfet, EikanWang
9. The XPU maintainers ([merge_rules.yaml:552-563][xpumr]): under A, the package
   registers the generic `SparseXPU` and `SparseCsrXPU` kernels
   ([A.6](#a6-xpu-and-mps)). Would you test them there, or take them into
   torch-xpu-ops?\
   Ask: EikanWang, guangyey

## Appendix: Technical feasibility

Details that matter when implementing the plan, not when deciding its direction.

### A.1 How the package reaches torch

From R2, pytorch CI builds it from a commit pin, as for triton
([build-triton-wheel.yml][triton], [binary_populate_env.sh:73-87][bpe]), so registering
a family from the package and bumping the pin is one PR. At release, the package is
built on torch N−1's stable ABI and released first; torch N pins it. With
`SPARSETORCH_USE_ATEN=1` (name to be decided) it registers nothing, and ATen's copies
serve. The package's CI runs the sparse tests twice, with its kernels and with the
switch, against the current torch, so the way back stays tested while ATen's copies
exist.

**The alias (R1 on)** re-exports, where each is documented today: `torch.sparse`;
`torch.optim.SparseAdam` ([optim/\_\_init\_\_.py:31][sa]); and
`torch.optim.functional.sparse_adam` ([functional.py:71][fsa]). Names that stay need
none: the `Tensor` methods, `torch.sparse_*_tensor`, `Embedding(sparse=True)` and
`torch.cuda.sparse` ([cuda/sparse.py:1][cudasparse]).

### A.2 How torch calls the package, and the hard cases

**How it works.**
- Core keeps every operator schema and its autograd formula. A sparse call goes through
  the dispatcher to the kernel registered for its key, and the package registers those
  kernels through the stable ABI ([OperatorEntry.cpp:397-479][dispatch-order]).
- Built that way, the package imports only libtorch's C functions, which carry the ABI
  version ([shim.h:16-37][shim-c]), and no C++ symbols. Where a kernel needs a
  `Scalar` or must raise a typed error such as `IndexError`, the stable ABI has no
  way yet. That kernel uses a small C++ bridge until it does.
- Core keeps the 29 sparse entries that build, read or copy a sparse tensor, so torch
  without the package can still load, print and move one.

| Hard case | How it moves |
|---|---|
| c10d sparse allreduce ([Ops.cpp:795-801][ops]) | Stays in core. It needs `ProcessGroup` and the Gloo backend, which the stable ABI doesn't reach. |
| Autograd | Formulas belong to the op and stay with the schemas; sparse work inside them moves as kernels; masked-semantics ops keep their meaning. |
| Package absent after a deletion | Each deleted kernel leaves a generated stub in ATen that says "install `sparsetorch`". A backend fallback would not do: composite kernels run ahead of it (37 ops, plus 53 `structured_delegate` ops on the compressed layouts), it doesn't cover dense keys, and it would claim ops that never moved. The package overrides the stubs, so the dispatcher's override warning must be silent for them. |
| Dense-to-sparse conversions ([native_functions.yaml:7585][nf-tosparse]) | One family. The package registers it under the dense keys too, and its stubs cover the dense keys. |
| MPS kernels | Ported to the stable ABI's Metal shim ([shim_mps.h:20-101][shim-mps]). This route is untested. ATen's copy stays behind the switch until a macOS runner tests the port. |
| MKL and cuSPARSE | cuSPARSE through the stable ABI (R1). For MKL, the package's owners choose between the shim and their own kernels. |
| Meta kernels | Registered with each family. 150 of 153 are the CPU kernel itself. |
| Inductor's 2:4 lowering ([mm.py:1128-1174][mm-lower]) | Stays until TorchAO has its own op (F), then goes with the `aten` op. |
| AOTInductor's C shim for `_to_sparse` ([c_shim_cpu.h:47][aoti-shim]) | It calls the CPU and CUDA kernels directly ([gen_aoti_c_shim.py:458-482][aoti-gen]), not through the dispatcher. After deletion it would reach the stub even with the package installed, and torchgen refuses to delete it outright, since compiled models call it by name. The shim keeps its name and signature, so compiled models still load; its body becomes a dispatcher call (a torchgen change, R2). |
| Compressed `mm` on CUDA calling triton ([SparseBlasImpl.cpp:241-255][bsr-triton]) | Moves with the compressed `mm` family (R3). The package registers the kernel from Python, and that kernel chooses between triton and C++. `torch.cuda`'s registration of the triton ops leaves core. |

Replacing the masked-semantics ops with `torch.masked` is a later decision under
pytorch's deprecation policy, not part of the move.

**The compiler stack.** `torch.compile` refuses sparse tensors today, at the input
([builder.py:3449-3460][dynamo-refuse]). This proposal doesn't change that, and
doesn't close the door: there is interest in compiling sparse code.[^compile]
Support can be added later by the new project, together with the compiler owners.
Compiled graphs would reach the package's kernels through the dispatcher, as eager
code does. Core would need dynamo to accept sparse inputs, and fake and meta support
for the ops, which the package registers per family. `torch.export` already
captures sparse ops ([config.py:522-526][dynamo-export]); its graphs, like
TorchScript archives, name aten ops and layouts that stay in core, so they keep
loading and run through the package. Fake tensors, AOTAutograd, FX,
functionalization and vmap check the layout only, or build sparse fakes from the
core constructors. They need no hook. One Python fake impl (`_spdiags`) stays with
its schema, as the autograd formulas do.

**Backward compatibility.** Each release is backward compatible with the previous
one. With the package, five families moved and deleted from ATen gave the same result
as torch's main branch, per test, gradients included. Until a family's ATen copy is
deleted, the switch restores it. After that, a missing package gives an error, never a
wrong result (see [A.9](#a9-evidence-kernel-families-moved)).

### A.3 What stays in core, and why

| Item | Why it stays |
|---|---|
| Layout values `sparse_coo`, `sparse_csr/csc/bsr/bsc` | part of the stable ABI's enum ([Layout.h:11-21][layout]) |
| Dispatch keys `Sparse`, `SparseCsr` | outside code cannot add a functionality key ([DispatchKey.h:211-213][dk]) |
| `SparseTensorImpl`, `SparseCsrTensorImpl`, with the 29 entries that build, read or copy a sparse tensor | core autograd reads them |
| Generic gradient accumulation | picks in-place or out-of-place add by layout ([accumulate_grad.h:188-239][acc]) |
| DDP bucketing of sparse gradients | set in the reducer ([reducer.cpp:91-162][red]) |
| The serialization name `torch._utils._rebuild_sparse_tensor` | saved files name it ([_utils.py:369][rebuild]) |
| `torch.backends.cusparselt` | documented without a prototype label ([backends.md:286-301][cslt]); cuSPARSELt stays a torch wheel dependency ([torch/\_\_init\_\_.py:402][preload]) |
| `torch.masked` | prototype, mostly strided |
| The cuSPARSE handle pool | dense MAGMA code uses it ([MagmaUtils.h:35][magma]); exposed through the stable ABI in R1 |
| Autograd formulas for sparse ops | they belong to the op, with its schema (see [A.2](#a2-how-torch-calls-the-package-and-the-hard-cases)) |
| The c10d sparse allreduce | it needs `ProcessGroup` and the Gloo backend ([Ops.cpp:795-801][ops]) |

### A.4 What core adds

| Release | Core adds |
|---|---|
| R0 | the `sparse` test config; a marker for the 122 sparse tests in core test files |
| R1 | `sparsetorch` in the wheel's `packages`, as `functorch` ([pyproject.toml:188][pyproj]); a merge rule and CODEOWNERS. **In the stable ABI: (1) `Scalar` as a kernel argument and for calling ops (in progress: [pytorch#203903](https://github.com/pytorch/pytorch/pull/203903), a draft); (2) accessors and an in-place setter for a sparse tensor's indices, values and nnz; (3) the current cuSPARSE handle; (4) typed exceptions: a kernel's `IndexError` reaches Python as `RuntimeError` (the message already does, [pytorch#180135](https://github.com/pytorch/pytorch/pull/180135)).** Any stable-ABI extension with a `Scalar` argument or a vendor library handle needs the same. |
| R2 | a CI job that builds the package from a pin, and the dependency in the wheel requirements; a stub, generated for each deleted sparse kernel, that says "install `sparsetorch`", and no override warning when the package replaces a stub ([OperatorEntry.cpp:213-225][override-warn]); AOTInductor C shims that call the dispatcher for moved ops, with their headers unchanged ([gen_aoti_c_shim.py:458-482][aoti-gen]); a way to skip sparse OpInfo samples without it; Inductor's 2:4 kernel choice through a registration ([mm.py:203-205][inductor]) |
| R3 | an extension point for `Adagrad`'s sparse step |

### A.5 New backends: the core pieces

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
with no kernel code in it.

The understanding core is asked for is in [The asks of core](#the-asks-of-core).

### A.6 XPU and MPS

156 of 173 `SparseXPU`, 128 of 139 `SparseCsrXPU`, 147 of 161 `SparseMPS` and 13 of 13
`SparseCsrMPS` entries in `native_functions.yaml` name the same generic kernel as
CPU/CUDA. The package registers each moved family for every key that names it today.
pytorch's XPU and MPS jobs test it during R2–R3; after that, the package needs XPU and
macOS runners, or those maintainers test it.

The sparse share of CI time is small partly because sparse is barely tested on backends:
on MPS, only COO, and only in trunk jobs, never in a PR's default checks.[^mps] Full MPS
coverage would cost about 0.45 h of Apple-silicon runner time per run (about a fifth of
today's MPS test run, 1.8–2.7 h), on a pool that pytorch CI keeps off ordinary PRs.
Under this proposal, the package's own CI decides which backends to cover, and pays for
it.

### A.7 A against G

| | A | G |
|---|---|---|
| Maintenance | Sparse PRs leave core. Coupling is through the aten schemas the package calls (~30 for the two families moved), which change under BC rules. A pin in R2–R3 only. | Registrations stay in core: a new op or backend needs a core PR. Internal ATen changes can break the sparse repo. A pin to bump about weekly: XPU's moved 61 times in 12 months, with 3 reverts.[^xpu] |
| Users | No Python name breaks. Risks: a dependency cycle; a broken package breaks `import torch`. | No change. |
| Effort | The four stable-ABI pieces first. | Files move; a pin and a CMake block ([CMakeLists.txt:1266-1336][xpucmake]). |
| CI | Sparse leaves pytorch's jobs. | Sparse is built in every job. |

G wins on users and effort; A on maintenance. G couples core to sparse for good, and
its weekly pin needs an owner sparse lacks. XPU's model works because Intel staffs it.

### A.8 Technical risks

| Risk | If it happens |
|---|---|
| A broken package breaks `import torch` | The package's CI runs `import torch` on every platform; a release blocker. |
| torch depends on a package that depends on torch | pytorch CI builds one more wheel from a pin, on every platform. The release order above is untested, and so is conda. |
| Hidden sparse paths in core (259 files mention sparse) | Found only by the 122 core sparse tests run without the package. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` turns sparse kernels off too | Sparse gets an import-time hook of its own. |

### A.9 Evidence: kernel families moved

COO `mul` and `addmm`, CPU and CUDA, were removed from ATen and registered from a
stable-ABI package (target 2.13) that `import torch` loads. Timed on these two families
only; diffs and results in [`RFC-00xx-assets/`](RFC-00xx-assets/README.md).

- [`pytorch-out-of-tree.diff`](RFC-00xx-assets/pytorch-out-of-tree.diff) deletes 8 lines
  of `native_functions.yaml`, nothing else. The PR that later deletes a family's ATen
  copy looks like this, plus deleting the sources. Removing `addmm.out` alone left
  `torch.addmm` in core: the family is the unit.
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

A third experiment kept ATen's `mul` kernels and registered the package's over them
([`sparse-ext-override.diff`](RFC-00xx-assets/sparse-ext-override.diff)). Results were
identical with the package, with a switch back to ATen, and without the package.
Removing a Python registration at run time restored ATen's kernel.

A fourth deleted five families from ATen, CPU and CUDA: `sin`, `abs`, `log1p`,
`_sparse_softmax` and `_to_sparse_csr`, the last under dense keys too. The package
registered them ([`pytorch-delete.diff`](RFC-00xx-assets/pytorch-delete.diff),
[`sparse-ext-delete.diff`](RFC-00xx-assets/sparse-ext-delete.diff)). With the package,
2,171 targeted tests gave the same result as main, per test, and gradcheck passed
through core's unchanged formulas. Without it, every failure was an error, none a wrong
result. `_sparse_softmax` needed a C++ bridge: a stable kernel's `IndexError` reaches
Python as `RuntimeError`, the fourth piece. A backend fallback missed composite and
dense-key kernels, hence the stubs.

A fifth checked the compiler stack on the fourth's build
([`pytorch-aoti.diff`](RFC-00xx-assets/pytorch-aoti.diff),
[`sparse-ext-aoti.diff`](RFC-00xx-assets/sparse-ext-aoti.diff)). With the package,
`test/export/test_sparse.py` (207 tests) and the dynamo sparse tests (5) gave the same
result as main, per test; without it, the 52 export tests that reach `abs` or
`_to_sparse_csr` raised an error. A program over `abs` on COO, saved by `torch.export`
on main, loaded on the modified torch and ran only with the package
([`export-save.py`](RFC-00xx-assets/export-save.py),
[`export-load.py`](RFC-00xx-assets/export-load.py)). `_to_sparse`'s dense kernel then
became a stub, since torchgen refuses to delete it outright. AOTInductor models
compiled on main and on the modified torch hit the stub even with the package loaded,
because the C shim calls the CPU kernel directly. Generating the shim's body as a
dispatcher call, headers unchanged, made both reach the package
([`aoti-compile.py`](RFC-00xx-assets/aoti-compile.py),
[`aoti-run.py`](RFC-00xx-assets/aoti-run.py)). CUDA was checked at the C shim, not
through a compiled CUDA model: triton was not installed.

## Appendix: User stories in detail

The evidence behind [User stories](#user-stories-r0), counted in November 2026. The full
report, with the 50 highest-ranked projects read by hand, is
[`user-stories.md`](RFC-00xx-assets/user-stories.md). The comparison with other
features is [`usage-baseline.md`](RFC-00xx-assets/usage-baseline.md), and the
measurements of when sparse pays off are
[`when-sparse-pays-off.md`](RFC-00xx-assets/when-sparse-pays-off.md).

### B.1 Method

- **Sources.**
  - GitHub code search, ten queries, sampled; and the dependents of PyG, DGL, TorchAO
    and torch_sparse.
  - A census of PyPI: of the 474,629 projects changed in the last 24 months, the 20,661
    whose newest wheel declares torch were downloaded and scanned, all but 10.
  - All 752 conda-forge packages that depend on pytorch.
  - 28 papers read in full; 53 forum questions and 22 pytorch issues; Hugging Face;
    ecosystem lists, among them the PyTorch landscape.
- **What counts.** A call to the sparse API on a project's code path. Comments,
  docstrings, tests and examples don't count. These are dropped:
  - pytorch itself and its copies, vendored files, and copies of other projects
    (matched by file content and package tree; the upstream is kept);
  - repositories with fewer than 10 commits, or with no packaging and no paper link.

  Projects whose latest release dropped their sparse path are past users, not users.
  Hugging Face models and Spaces (27) and projects that use only torch.masked (43) are
  set apart.
- **By rule.** The field (deep learning, scientific, unclassified) comes from keywords.
  The devices are those named in the files that call sparse. "Maintained" means
  activity in the last 12 months.
- **By hand.** The 50 highest-ranked projects were read against their code; 66 in all,
  since removals let others in. Of the 66, 6 were not users, 2 were past users, 7 were
  doubtful (kept at low confidence) and 1 was a duplicate.
- **Ranking.** A project's weight is confidence (1, ½ or ¼) × activity (1 if
  maintained, ½ if not) × (1 + log10(1 + PyPI dependents)). Ranked by weight or by
  count, no layout × device cell moves by more than a third of the list.

### B.2 Tables

| Field | Projects | Maintained |
|---|---|---|
| deep learning | 1,101 | 538 |
| scientific computing | 525 | 415 |
| unclassified | 172 | 99 |
| all | 1,798 | 1,052 |

| GitHub owner | Projects | Maintained | ≥ 1 PyPI dependent | ≥ 100 |
|---|---|---|---|---|
| personal account | 1,132 | 613 | 42 | 1 |
| organization | 576 | 362 | 86 | 9 |
| PyPI package, no repository found | 85 | 73 | 14 | 0 |
| account no longer resolves | 5 | 4 | 0 | 0 |

**Layout × device:** projects, with their weight in parentheses. A project counts once
per cell it uses. "cpu" means CPU named or no device named; "generic" means code that
follows its input's device. The thin cells are partly testing, typing and inspection
tools that list every layout and device.

| Layout | cpu | generic | cuda | rocm | mps | xpu |
|---|---|---|---|---|---|---|
| COO | 514 (409) | 907 (731) | 475 (369) | 11 (12) | 42 (47) | 9 (9) |
| CSR | 197 (183) | 338 (323) | 214 (199) | 11 (11) | 23 (26) | 6 (6) |
| CSC | 44 (40) | 72 (77) | 38 (38) | 5 (5) | 13 (16) | 4 (4) |
| BSR | 20 (23) | 35 (35) | 29 (28) | 6 (8) | 9 (8) | 4 (4) |
| BSC | 18 (20) | 24 (26) | 18 (17) | 5 (5) | 9 (8) | 4 (4) |
| 2:4 | 10 (7) | 22 (16) | 21 (17) | 4 (6) | 3 (2) | 4 (4) |

| Op | Projects | Weight | Scientific |
|---|---|---|---|
| `sparse_coo_tensor` | 1,048 | 818 | 343 |
| `torch.sparse.mm` | 743 | 532 | 217 |
| `torch.sparse.sum` | 342 | 195 | 74 |
| `sparse_csr_tensor` | 268 | 261 | 129 |
| `to_sparse_csr` | 228 | 216 | 93 |
| `.to_sparse()` | 191 | 148 | 65 |
| `to_sparse_coo` | 138 | 127 | 55 |
| `SparseAdam` | 100 | 69 | 11 |
| `torch.sparse.softmax` | 72 | 35 | 12 |
| `check_sparse_tensor_invariants` | 58 | 65 | 31 |

**Beside other features,** by the same method. Issues, PRs and reviews cover the 12
months to 2026-11-16. The projects are counted in code-search samples of equal size for
sparse and linalg, and half that size for nested tensors and DTensor.

| | sparse | linalg | nested | DTensor |
|---|---|---|---|---|
| ATen C++/CUDA lines | 27,542 | 19,947 | 7,803 | 51 |
| Listed maintainers | 5, all emeritus | 1 active, 4 emeritus | 6, all emeritus | 9 active, 11 emeritus |
| Issues opened | 117 | 258 | 24 | 155 |
| PRs opened | 119 | 222 | 9 | 720 |
| … reviewed by a listed maintainer | 42 % | 19 % | 2 of 9 | 31 % |
| Distinct human reviewers | 36 | 44 | 7 | 99 |
| Projects calling it, in the sample | 959 | 1,903 | 169 | 186 |
| conda-forge packages calling it, before the copy and hand-check steps | 57 | 193 | 8 | 11 |

linalg's ATen count includes dense matmul. DTensor's maintainers are those of
`torch.distributed`, and most of its use comes through training frameworks, which
direct calls don't show.

### B.3 What the code shows

- **Small parts of large projects.** In many top projects sparse is one optional
  component: a model adapter, a loss, an experimental path.
- **A data format, not compute.** Some use sparse tensors only to build or ship data
  and densify at once: CSR as a wire encoding, COO as a scatter idiom.
- **MPS is avoided where it would be needed.** suite2p falls back to CPU "until sparse
  works on MPS", instanseg moves to CPU, and omicverse goes to MLX on Apple hardware.
- **`spsolve` is hard to rely on.** torch-sla probes for it before use. A Stack Overflow
  question on an RTX 40 card ([79227337](https://stackoverflow.com/questions/79227337))
  shows the cuDSS requirement in practice.
- **Requests.** Of the 75 issue and forum records, 13 name a project and 5 of these
  are requests. On MPS, they ask for CSR (3), COO (3) and BSR (1); on 2:4, for batched
  matrices, Conv2d and training.
- **Removals in 2026.** TorchAO's `sparsity/` package, at HEAD; its latest release,
  v0.18.0, still has it. NeMo's sparse loss, with "older and non-maintained models".
  PhysicsNeMo's regridding plugin, in a dependency clean-up. xformers' cuSPARSELt 2:4
  code, in favour of torch.sparse.

### B.4 When sparse pays off

Ten of the thirteen papers that compare with dense report gains, and three report none.
Of the ten, four measure torch.sparse, five the authors' own kernels, and one a
comparison that includes torch.sparse BSR.

| Source | What was measured | Result |
|---|---|---|
| BSR Triton kernels in torch ([pytorch#110396](https://github.com/pytorch/pytorch/pull/110396), [#111760](https://github.com/pytorch/pytorch/pull/111760), [#114757](https://github.com/pytorch/pytorch/pull/114757)) | BSR matmul against strided matmul, and tuning, on A100 and other GPUs | above a break-even sparsity that depends on block size, dtype and GPU, BSR beats strided; tuning lowered the break-even; `nn.linear` with float32 BSR weights became 62–72 % faster on average |
| [Bundle Adjustment in the Eager Mode](https://arxiv.org/abs/2409.12190) | torch.sparse, a bundle-adjustment Jacobian at under 0.01 % density | 5.2 TB dense against 125 MB sparse |
| [Learning Discrete Riemannian Metrics for Physical Fields](https://arxiv.org/abs/2608.14556) | torch.sparse, per step | 10 ms sparse against 700 ms dense |
| [Beyond Matryoshka](https://arxiv.org/abs/2503.01776) | torch.sparse, retrieval over 1M items | 2.1× faster than dense embeddings above 80 % sparsity |
| [TorchAO](https://arxiv.org/abs/2507.16099) | torch.sparse 2:4, ViT | up to 1.3× faster than dense |
| [Fast Inference with Kronecker-Sparse Matrices](https://arxiv.org/abs/2405.15013) | structure-aware implementations, BSR through `torch.sparse.mm` among them, A100 | they beat the generic dense and sparse baselines; the gain grows with matrix size |
| [Longer Attention Span](https://arxiv.org/abs/2502.01659) | the authors' CUDA kernels, attention masks | CSR 7.8–31.6× faster than dense attention; COO hundreds of times slower |
| [SpQR](https://arxiv.org/abs/2306.03078) | PyTorch sparse matmul in LLM inference | "not faster than 16-bit inference" |
| [Is 2:4 Sparsity Still Worth It?](https://arxiv.org/abs/2610.28947) | 2:4 against dense FP8 and BF16, three GPUs | FP16 2:4 slower than dense FP8; BF16 2:4 no faster than dense BF16; stacked on FP8, 1.3–1.7× on prefill-shaped layers |
| [Operation-Level Performance Benchmarking of GNNs](https://arxiv.org/abs/2207.09955) | native PyTorch against PyG ops | native ops as or more competitive at low to moderate sparsity |
| [Forum post](https://discuss.pytorch.org/t/198836) | a COO linear layer, 2048 × 2048, 90 % sparse, CPU | training about 10× slower than dense |

Gains come with very sparse data, with structure (blocks, Kronecker factors, 2:4) and
with the right layout. They vanish at moderate sparsity, and where format conversion or
memory traffic dominates.

### B.5 Limits

- **Sampling.** Code search is sampled and sees only public default branches. The
  PyPI census reads only the newest wheel's metadata: 12,900 sdist-only projects were
  not judged, and a package that reaches torch only through another library, without
  declaring torch, is missed. Private and closed-source users show only through issues,
  forums and papers.
- **Copies.** On PyPI, copies are matched by path and package name, not by content.
  More may remain among the PyPI-only packages; their number is not known.
- **Rules.** Outside the 50 read by hand, field and device come from rules. The field
  rule is wrong for an estimated 11–17 % of projects (95 % interval 3–39 %). A paper
  link doesn't prove that a repository is the paper's code: in a sample of 10 small
  paper repositories, 3 were not users.
- **What a count means.** A use is a call on the code path; it doesn't say how much a
  project depends on sparse. "Dependents" are packages that depend on a project, not
  packages that use its sparse path: of the 1,146 repositories that call PyG's sparse
  path, only 167 are certain to produce torch.sparse tensors.
- **Not fully read.** Semantic Scholar's full-text search (it needs an API key),
  OpenAlex's later result pages, and Hugging Face code beyond search terms and the top
  downloads.

[^prs]: GitHub search, 2026-11-16: `repo:pytorch/pytorch is:pr created:2025-11-16..2026-11-16 label:"module: sparse"`,
    and the same with `label:"release notes: sparse"`; against 891 random PRs of the
    window with neither label. Human: not a bot, not the author.
[^201801]: https://github.com/pytorch/pytorch/pull/201801#issuecomment-6453111398,
    https://github.com/pytorch/pytorch/pull/201801#issuecomment-6453318175 and
    https://github.com/pytorch/pytorch/pull/201801#issuecomment-6465922172.
[^112849]: https://github.com/pytorch/pytorch/issues/112849#issuecomment-5267130814 (2026-08-12).
[^inv]: Crashes on tensors that fail the checks, filed as bugs: [pytorch#182662](https://github.com/pytorch/pytorch/issues/182662) and
    [pytorch#190045](https://github.com/pytorch/pytorch/issues/190045) (closed as not planned), [pytorch#193824](https://github.com/pytorch/pytorch/issues/193824) (open). PRs adding
    checks that exist but are off: [pytorch#163535](https://github.com/pytorch/pytorch/pull/163535) (landed, then reverted) and
    [pytorch#169807](https://github.com/pytorch/pytorch/pull/169807) (stopped); see [pytorch#169954](https://github.com/pytorch/pytorch/issues/169954). Since [pytorch#171695](https://github.com/pytorch/pytorch/pull/171695),
    torch warns once that the checks are implicitly off ([Context.cpp:945][ctx]).
[^issues]: GitHub search, 2026-11-16: `repo:pytorch/pytorch created:2025-11-16..2026-11-16`,
    `is:issue` or `is:pr`, with and without the sparse labels.
[^size]: Raw lines at the commit above of `aten/src/ATen/native/sparse/**` and ATen's
    other `Sparse*` files, against all C++/CUDA in ATen.
[^cs]: GitHub code search, 2026-11-03, `language:Python`: `"torch.sparse_coo_tensor"`,
    `"SparseAdam"`, `"Embedding(" "sparse=True"`. Approximate; counts copies.
[^ci]: 99 landed PRs touching no sparse file, 2026-08-05 to 2026-11-02: job times
    from the GitHub Actions API, per-file test times (`test-times.json`, 2026-11-02)
    over the test files target determination chose.
[^mps]: At the commit above: no macOS job in `pull.yml`. MPS runs on trunk (after merge,
    or on a PR labeled `ciflow/trunk`) ([trunk.yml:292-302][trunkmac]) or with a
    `ciflow/mps` label ([mac-mps.yml:4-8][macmps]),
    which runs a fixed test list without the sparse files ([run_test.py:1994-2013][mpslist]).
    `test_sparse.py` allows MPS for COO only ([test_sparse.py:5920][tsmps]). The estimate
    scales CUDA's device time for the two sparse test files, 2026-11-02, to MPS.
[^community]: GitHub, issues and PRs labeled `module: sparse` or `release notes: sparse`,
    created 2025-11-16 to 2026-11-16: distinct authors and commenters. Bots, the people named
    above and the core maintainers are excluded.
[^xpu]: GitHub API, 2026-11-16: commits to `third_party/xpu.txt`, 2025-11-16 to 2026-11-16.
[^compile]: [pytorch#113232](https://github.com/pytorch/pytorch/issues/113232) (open since 2023-11):
    Inductor support for GNN workloads with sparse tensors.
    [pytorch#192093](https://github.com/pytorch/pytorch/pull/192093) and
    [pytorch#192092](https://github.com/pytorch/pytorch/pull/192092) (open PRs, 2026-08):
    trace sparse compressed tensor creation in dynamo, and run sparse ops as Inductor
    fallback kernels.

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
[ctx]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/Context.cpp#L945
[shim]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/shim_common.cpp#L77-L186
[xpucmake]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/caffe2/CMakeLists.txt#L1266-L1336
[ft]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/functorch/__init__.py#L6-L7
[autoload]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/__init__.py#L3531-L3546
[vendor]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_vendor/README.md
[trunkmac]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.github/workflows/trunk.yml#L292-L302
[macmps]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.github/workflows/mac-mps.yml#L4-L8
[mpslist]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/test/run_test.py#L1994-L2013
[tsmps]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/test/test_sparse.py#L5920
[dispatch-order]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/core/dispatch/OperatorEntry.cpp#L397-L479
[shim-c]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/stable/c/shim.h#L16-L37
[nf-tosparse]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/native_functions.yaml#L7585
[shim-mps]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/inductor/aoti_torch/c/shim_mps.h#L20-L101
[mm-lower]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_inductor/kernel/mm.py#L1128-L1174
[override-warn]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/core/dispatch/OperatorEntry.cpp#L213-L225
[rfc3]: https://github.com/pytorch/rfcs/pull/6
[usecases]: https://github.com/pytorch/pytorch/issues/10043
[bae]: https://github.com/pytorch/pytorch/pull/201801#issuecomment-6453318175
[spsolve]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/native_functions.yaml#L14698-L14702
[spsolve-cudss]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/cuda/SparseCsrTensorMath.cu#L717-L721
[ao4728]: https://github.com/pytorch/ao/pull/4728
[singlecell]: https://github.com/pytorch/pytorch/issues/10043#issuecomment-966518421
[aoti-shim]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h#L47
[bsr-triton]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseBlasImpl.cpp#L241-L255
[dynamo-refuse]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_dynamo/variables/builder.py#L3449-L3460
[dynamo-export]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_dynamo/config.py#L522-L526
[aoti-gen]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torchgen/gen_aoti_c_shim.py#L458-L482
