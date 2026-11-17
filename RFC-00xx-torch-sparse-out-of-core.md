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
- User stories, collected at R0, set the new project's priorities.
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
`sparse=True`: 11,968.[^cs] So sparse cannot be removed; it can move behind the same
names.

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
answered with one in the same thread.[^201801] The user stories below test the question
with evidence.

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

Before any code moves, the project collects user stories from public evidence: who
uses each sparse layout, cuSPARSE and cuSPARSELt, and what for, in scientific
computing as well as deep learning. The sources are:
- projects that depend on torch and call the sparse constructors;
- PyPI packages;
- papers;
- pytorch issues, among them [pytorch#10043][usecases], "Sparse tensor use cases",
  open since 2018.

Each story records the project, its field, the layouts, devices and ops it uses, and
whether it needs cuSPARSE or cuSPARSELt. Two are already public:
- bundle adjustment with CSR and BSR tensors, which needs them on MPS ([pytorch#201801][bae]);
- row slicing of CSR tensors on GPU, in single-cell biology ([pytorch#10043][singlecell]).

The design document ranks the project's priorities, and the backends it adds, from
these stories.

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
| **R0** | E: the policy in `sparse.md` and CONTRIBUTING. D: a `sparse` test config. Governance is decided (question 2); without owners, the plan stops here, at E + D + F. User stories are collected. The new repo gets named owners and a design document. F starts; the semi-structured API warns "moving to TorchAO". | today |
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
- User stories behind the priorities: every op or backend added after R0 cites at
  least one actively maintained public project that needs it. Stories are counted at
  R0 and at each release, by layout and device.
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
([builder.py:3449-3460][dynamo-refuse]), and this doesn't change. `torch.export`
captures sparse ops ([config.py:522-526][dynamo-export]); its graphs, like TorchScript
archives, name aten ops and layouts that stay in core, so they keep loading and run
through the package. Fake tensors, AOTAutograd, FX, functionalization and vmap check
the layout only, or build sparse fakes from the core constructors. They need no hook.
One Python fake impl (`_spdiags`) stays with its schema, as the autograd formulas do.

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
[singlecell]: https://github.com/pytorch/pytorch/issues/10043#issuecomment-966518421
[aoti-shim]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h#L47
[bsr-triton]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseBlasImpl.cpp#L241-L255
[dynamo-refuse]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_dynamo/variables/builder.py#L3449-L3460
[dynamo-export]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_dynamo/config.py#L522-L526
[aoti-gen]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torchgen/gen_aoti_c_shim.py#L458-L482
