> **Pre-plan for discussion, not yet an RFC for comment.** It proposes a direction
> and lists the questions to settle before the full RFC is written.

# Moving torch.sparse out of pytorch core

**Authors:**
* @pearu

All pytorch line numbers below are at commit
[`a67a74d3e7`](https://github.com/pytorch/pytorch/tree/a67a74d3e71bf2891b0d02b126fe7390090e1f1c)
(2026-11-02).

## **Summary**

torch.sparse has users but no owner. At least 1,798 public projects call it; its five
maintainers are emeritus, and core no longer develops it. When a sparse PR arrives,
core reviewers ask who needs it and who will maintain it, and nobody can answer.

This RFC proposes two steps. The second is taken only if owners commit.

1. **Now, in core:** write today's practice down as a policy: a new op or backend
   lands only with user demand and an owner who commits. Give the sparse tests a CI
   config of their own.
2. **With owners:** move torch.sparse, 2:4 sparsity included, to a project of its own,
   `sparsetorch`, which torch depends on. `import torch` brings `torch.sparse` as
   today. No Python name breaks.

- **Who drives it.** One owner has committed, the author. One committed owner is enough
  to start; more owners reduce the risk. Without an owner, sparse stays in core under
  step 1, a sound end state.
- **Where it goes.** Development for users who can be named, not parity with dense
  tensors. The bar of step 1 stays; the project applies it, and core takes part where a
  decision means notable changes to core, as a new backend does.
- **What it costs.** The project writes the PRs and runs its own CI. Core creates the
  repo, hosts one CI build job during the move, and approves the core-side lines of
  about 100 deletion PRs. It reviews a few small core changes, four additions to the
  stable ABI among them. If those four can't land, the fallback is a kernel repo that
  pytorch builds in tree, as for XPU. Distributions that package torch build one more
  package.
- **How long.** Four pytorch releases, about eight months, and one more to clean up.
  Every release is a safe place to stop.
- **Why not less.** Moving only new backends out of tree leaves ~26 k lines of CPU and
  CUDA kernels in core with no owner.

How the move works is in the appendix,
[Technical feasibility](#appendix-technical-feasibility).

## **Motivation**

**The problem is ownership, not use.**
- All five "Sparse (torch.sparse)" maintainers are emeritus
  ([persons_of_interest.md:150-156][poi]), and only four of them can approve under the
  "Sparse" merge rule ([merge_rules.yaml:159-180][mr]). It has been said in public:
  "we are not actively working on torch.sparse".[^112849]
- They still review. Of 119 sparse PRs in 12 months, 50 (42 %) were reviewed by an
  emeritus sparse maintainer.[^prs] The rest of the review falls on core maintainers.
- The cost is not delay, nor volume: sparse is 1.5 % of issues and 0.4 % of
  PRs.[^issues] Sparse PRs get a first review as fast as other PRs, and may take a
  little longer to land (median 6 against 4 days).[^prs] The cost is who reviews, and
  that nobody owns the decisions.
- Filling a hole is now cheap. With an LLM, a missing op (on MPS, or on CUDA where CPU
  has it) is easy to find and fill. Core reviewers then ask who needs the op, and
  there is no owner to answer or to maintain it afterwards.
- The same question blocks wanted work.
  [pytorch#201801](https://github.com/pytorch/pytorch/pull/201801) (open since
  2026-10-23) adds the CSR/BSR constructor on MPS, a first step toward CSR on MPS for a
  downstream library. A core reviewer asked who would maintain MPS
  sparse parity; the sparse reviewer could not commit to it.[^201801]

**It cannot be removed, and it is worth keeping.** At least 1,798 public projects call
torch.sparse outside their tests; 1,052 of them are maintained, and 575 are published
on PyPI or conda-forge
([Appendix: User stories in detail](#appendix-user-stories-in-detail)). They use it for
graph adjacency (PyG), sparse gradients (DeepSpeed), sparse encoders
(sentence-transformers), mesh Laplacians (pytorch3d), optimal transport (POT), Gaussian
processes (linear_operator) and bundle adjustment (pypose). The 525 scientific projects
are more often maintained than the deep-learning ones (79 % against 49 %). They mostly
build their matrices with SciPy and hand them to torch.sparse for the GPU and for
autograd ([B.7](#b7-scientific-users-and-scipy)). So it can move, but only behind the
same names.

Nothing replaces it. Of five sparse array libraries compared (torch.sparse, SciPy,
PyData `sparse`, CuPy and Finch), it is the only one with GPU support, autograd, and
batched and hybrid layouts together ([B.8](#b8-torchsparse-beside-other-sparse-libraries)).
The PyTorch libraries with layouts of their own last released in 2023 and 2024
([B.3](#b3-what-the-code-shows)). xformers removed its 2:4 code and points users to
torch.sparse, and torch-sla (2026) builds differentiable solvers around it.

**The aim.** Give torch.sparse owners of its own, who can answer "who needs this?" from
user evidence, and who develop it for those users, not for parity with dense tensors.

**What users ask for**, in the order used throughout:
- **MPS.** pypose's bundle adjustment asks for CSR and BSR on MPS
  ([pytorch#201801][bae]); suite2p, instanseg and omicverse route their sparse code
  around it.
- **Solvers.** `spsolve` runs only in CUDA builds with cuDSS, and on XPU
  ([native_functions.yaml:14698-14702][spsolve],
  [SparseCsrTensorMath.cu:717-721][spsolve-cudss]); torch-sla ships its own
  differentiable solvers.
- **Autograd through sparse inputs, and missing ops.** These are what forum questions
  and issues name most: missing ops 18 times, autograd 12. The Sparse Array
  Consortium's benchmark suite works around the ops torch.sparse lacks (`einsum`,
  `where`, logical ops).[^consortium]
- **SciPy interop.** 64 % of the scientific projects scanned also use scipy.sparse, and
  201 use both libraries in one file, mostly to convert between them
  ([B.7](#b7-scientific-users-and-scipy)).

Few users report. The maintainers of about 4 % of the projects have taken part in
pytorch's sparse issues, and none of the 22 workarounds found in code links one
([B.9](#b9-where-users-voices-go)). So the issue tracker undercounts what users miss.

**The limits.**
- The API is beta ([sparse.md:13-15][beta]); 2:4 and `torch.masked` are prototype
  ([sparse.md:72][semi], [masked.md:18-19][masked]).
- Most use is basic: the constructors, `mm` and `sum`.
- It is 2.5 % of the PyPI packages that declare torch (516 of 20,661), and reach is
  concentrated: 10 projects have a hundred or more PyPI dependents.
- Some users left in 2026: TorchAO, NeMo and PhysicsNeMo
  ([B.3](#b3-what-the-code-shows)).
- Gains depend on sparsity, structure and layout ([B.4](#b4-when-sparse-pays-off)).

**Why not new maintainers in core?** If core supported sparse maintenance, there would
be little need to move it. Today no one is obligated to maintain torch.sparse, yet
sparse PRs keep coming, and not all of them are generated noise. Part of what they
change lies outside the code sparse maintainers own: registering a sparse kernel changes
`native_functions.yaml`, which the "Sparse" merge rule doesn't cover (12 of the 119 PRs,
8 of them for MPS or XPU).[^prfiles] For such changes it is hard to find a core reviewer
with the interest or the expertise to approve them. Owners with a package of their own
resolve much of this: they review and approve, decide what to add, and test the backends
they choose. The move's first step gives them that for the Python code, still inside
pytorch/pytorch. Once kernels move, a kernel registers from the package, and
`native_functions.yaml` doesn't change.

**What core and users gain.** Core: sparse PRs reviewed and approved by owners of their
own. Users: an owner who can say yes. A new op or backend lands only with an owner, and
the sparse project is that owner.

**CI is not the reason.** A PR that touches no sparse file spends ~1.9 of its ~119
runner hours of test jobs on sparse tests, 1.6 %.[^ci] Saving it is a side effect.

## **Proposed Implementation**

**Terms.**
- *The sparse project*: the owners of torch.sparse and their repo,
  `pytorch/sparsetorch`.
- *The package*: `sparsetorch`, which torch depends on; inside the torch wheel at R1,
  a wheel of its own from R2.
- *Options A–G*, from [Alternatives](#alternatives): the proposal is A, reached through
  C, with D and E at once; G is the fallback.
- *R0–R4*: R0 is the first pytorch release after this RFC is accepted; R1–R4 follow,
  about two months apart.
- *A family*: an op with all its variants (functional, in-place, `out=`), the unit that
  moves.
- *The switch*: an environment variable that brings ATen's copies back
  ([A.1](#a1-how-the-package-reaches-torch)).
- *A commit pin*: the package commit that pytorch CI builds, as for `triton`.
- *The stable ABI*: pytorch's versioned C interface that lets an extension built for one
  torch release run with later ones.

### Owners and governance

The RFC can be accepted for the maintenance policy and the test config alone. The
move, R1 on, starts only with an ownership commitment: named people who commit to own
the sparse project, review its PRs and drive its direction, and a rule for who breaks
ties on design decisions (question 2). The bar was stated in public before this RFC:
"I would like to see that there are individuals who are interested in doing the work,
and maintaining it long term."[^112849]

**Committed:** pearu, the author of this RFC. The commitment covers review, fixes,
maintenance and releases: fully while an organization supports the owner's time, and
otherwise in the owner's own time, at lower availability. The record:
- sparse layouts and kernels: pinned memory for all five layouts
  ([pytorch#129645](https://github.com/pytorch/pytorch/pull/129645)), int8 BSR Triton
  kernels ([pytorch#133855](https://github.com/pytorch/pytorch/pull/133855));
- the stable ABI: [pytorch#167923](https://github.com/pytorch/pytorch/pull/167923), and
  the port of torchaudio to it ([pytorch/audio#4137](https://github.com/pytorch/audio/pull/4137),
  [pytorch/audio#4142](https://github.com/pytorch/audio/pull/4142));
- two earlier sparse RFCs: a roadmap ([pytorch/rfcs#6](https://github.com/pytorch/rfcs/pull/6))
  and a fill-value property ([pytorch/rfcs#8](https://github.com/pytorch/rfcs/pull/8)),
  both still open.

One committed owner is enough to start. Technical review (correctness, coverage,
regression tests) can largely be done with AI agents, as in pytorch's own PR lifecycle,
where an automated review comes before the human one
([CONTRIBUTING.md:124-204][prlife]). Part of the project's agent review is to split a
PR's review into technical and architectural parts, which shows when a human needs to
step in. Human review is needed for architecture: whether a new op belongs, whether a
change fits the design, changes to the design, and decisions that break compatibility.
With a single owner, core reviewers or the community give that review, and users who ask
for a feature review the PRs that add it. Inside pytorch/pytorch, an owner named in the
sparse merge rule can have technical PRs approved by pytorch's automated reviewer when
the rule covers every file the PR changes ([CONTRIBUTING.md:672-686][greenlight]).

**When core takes part.** Core takes part in a design decision only when the project has
too few owners, or when the decision means notable changes to core, as a new backend
does.

More owners reduce the risk, and can review each other's PRs. Comparable projects
suggest that a project lasts when more than one organization backs it, when it has
written governance, and when its CI doesn't depend on a single funder.[^lasts] Seeking
more owners is one of this RFC's goals.

**Known to the community, with the experience to own and drive the direction.** They
have a record in the code that moves; being listed here is not a commitment:
- amjames: sparse compressed kernels, and a sparse code owner ([pytorch#104062](https://github.com/pytorch/pytorch/pull/104062)).
- nikitaved: sparse COO and compressed kernels on CPU and CUDA ([pytorch#72710](https://github.com/pytorch/pytorch/pull/72710),
  [pytorch#202778](https://github.com/pytorch/pytorch/pull/202778)).
- alexsamardzic: 2:4 semi-structured sparsity and CUTLASS kernels, in pytorch and
  TorchAO ([pytorch#123473](https://github.com/pytorch/pytorch/pull/123473), [pytorch/ao#1940](https://github.com/pytorch/ao/pull/1940)).
- Isalia20: the MPS sparse kernels ([pytorch#167013](https://github.com/pytorch/pytorch/pull/167013), [pytorch#169125](https://github.com/pytorch/pytorch/pull/169125), [pytorch#169368](https://github.com/pytorch/pytorch/pull/169368)).

Help also comes from the community. In the last 12 months, 67 other people opened sparse
PRs, and 71 opened or commented on sparse issues.[^community]

**Periods of minimal effort.** The aim is a sustainable project with active owners.
There may still be periods when no owner can give more than minimal effort. The project
then keeps a minimal-effort owner: an owner who stays on or, if none does, core
(question 2). Core is the natural candidate, since torch depends on the package. Such a
period keeps the package healthy until community interest or a new owner brings it back
to active development. How it works is left to when this RFC is implemented, under
core's AI policy ([AI_POLICY.md:29-32][aipol]). With an LLM, a hole in torch.sparse is
cheap to fill ([Motivation](#motivation)); the idea is to make sorting cheap too. AI
agents help reviewers sort sparse issues and PRs into important bug fixes, updates to
follow torch, and feature requests or changes of direction. The last wait for an owner
who commits, as the maintenance policy asks ([The maintenance
policy](#the-maintenance-policy)). For example, agents could prepare simple, localized
fixes that a reviewer answers for, and leave existing tests unchanged. With no fixes
landing, a stable-ABI wheel keeps working with later torch releases, as torchaudio's
does ([Prior Art](#prior-art)).

If no owner has committed by R0, nothing moves. That is a sound end state, not a
failure, and the question can be reopened when owners come forward.

### Vision

torch.sparse stays PyTorch's sparse tensor library, owned and developed by the sparse
project for users who can be named ([Motivation](#motivation)).

**Scope.** The sparse project owns what sparse tensors mean and the code behind it: the
layouts' invariants and semantics, the documentation, the CPU, CUDA and MPS kernels that
move from core, and a test suite that is device-generic and shared. Projects for other
devices, today torch-xpu-ops, own their kernels and register them through a documented
interface. This is how torch-xpu-ops already works, except that the interface is not
written down ([A.6](#a6-xpu-and-mps)). A backend is supported when CI runs the shared
suite on it.

**Schemas.** torch's general-purpose ops aim at the Array API (many of torch's
machine-learning ops are outside it), and sparse tensors aim at the semantics of strided
ones. So core keeps the schemas of general-purpose ops, with their autograd formulas,
and the sparse project registers kernels for them. It owns the schemas of
sparse-specific ops only: new sparse-only ops, defined outside core, and later the
existing ones (`torch.sparse.*`, the sparse factories, `to_sparse_*`). That is a
direction after the move, not part of R0–R3.

**Design decisions,** for the design document the new repo starts with (R0):
- Unspecified elements are zero. The ops that treat unspecified elements as missing,
  not as zero (such as `torch.sparse.softmax` and `torch.sparse.log_softmax`), keep
  that meaning.
- No implicit conversion. An op without a kernel for a layout raises; it does not
  convert to dense or to another layout. Conversion is expensive, and it hides the
  missing fast kernel.
- Invariant checks stay off by default, for performance. Yet crashes on invalid tensors
  keep being filed as bugs, and PRs keep adding checks that exist.[^inv] So a bug report
  starts with a reproducer that still fails with the checks on
  (`torch.sparse.check_sparse_tensor_invariants`), and AI agents read the design
  document before they file issues or PRs.

**New features** start as opt-in experiments, each for a maintained project that asks
for it. They live in a namespace of their own, with no compatibility promise until they
move to the stable namespace; the mechanism is not fixed yet. Use decides whether an
experiment stays.

**Priorities** come from what users ask for ([Motivation](#motivation)): MPS, solvers,
autograd through sparse inputs, missing ops (39 Array API functions fit zero semantics
and are missing, [B.8](#b8-torchsparse-beside-other-sparse-libraries)), and SciPy
interop. Backends are added by the project's bar: a project that asks, and an owner. A
new backend needs changes to core's keys and code generation, small but notable
([A.5](#a5-new-backends-the-core-pieces)), so core takes part in that decision (ask 4).

### The maintenance policy

Core no longer develops torch.sparse; this is already the practice
([Motivation](#motivation)). This proposal writes it down as the maintenance policy
(E), from R0, in `sparse.md` and CONTRIBUTING:
- bug fixes are reviewed as today;
- new ops and new backends land only with user demand and an owner who commits to
  maintain them;
- new features belong in the sparse project, once it exists, not in core.

From R0, the sparse tests also run in a CI config of their own, `sparse` (D).

### The migration

**Design rule.** Move code as close to verbatim as possible, with its existing bugs,
issues and limitations. Keep the existing tests passing. Make each step small enough
for a human reviewer to follow. A bug found during the move gets its own PR, after the
move.

| Release | What changes | How the package ships |
|---|---|---|
| **R0** | E: the policy in `sparse.md` and CONTRIBUTING. D: a `sparse` test config. Owners and governance are decided (question 2); without an owner, the plan stops here. User stories are recounted. The new repo gets its owners and a design document. | — |
| **R1** | C: `torch/sparse/*`, `SparseAdam` and `sparse_adam` move into the in-repo package `sparsetorch`, laid out as the new repo. Their names in torch re-export it ([A.1](#a1-how-the-package-reaches-torch)). | inside the torch wheel |
| **R2** | The package moves to the new repo. Kernels move, one family per PR: the package's kernels take over from ATen's. ATen's copy stays, frozen, behind the switch. It is deleted only after a release in which the package's kernel shipped and passed the sparse tests. The package builds on the stable ABI and loads through a `torch.backends` entry point. | its own wheel, required `==`, built by pytorch CI from a commit pin |
| **R3** | The rest, the 14 MPS-only kernels among them. ATen copies of R2's families deleted. Sparse test files leave pytorch CI. | its own wheel, required `>=` |
| **R4** | ATen copies of R3's families deleted. | the same |

**The size.** About 100 families move in R2–R3: 89 with sparse dispatch keys, and 11
with sparse kernels under dense keys, the 2:4 ops among them.[^families] Each takes
three PRs: one in the new repo, and two in pytorch, a commit-pin bump and, a release
later, the deletion of its ATen copy. Sparse owners approve the sparse-only files, the
commit pin among them. A core reviewer approves a deletion's changes to shared files: a
few dispatch lines each and, for 18 families, the removal of a kernel's source
([A.4](#a4-what-core-adds)).

**Why four releases, and a fifth to clean up.** No Python name breaks, so the pace is
set by the work, not by deprecation: R1 is mechanical, R2–R3 carry the kernels.

**Every release is a safe place to stop, and to reconsider the plan.** Until a family's
ATen copy is deleted, the switch can bring ATen's copies back for any release, for
example when the package isn't ready. If plans change, whether to continue, change
course or move code back is decided then, under the project's governance, starting from
a working torch.

### What stays in core

Core keeps what defines a sparse tensor and what other core code reads: the layouts and
dispatch keys, the two tensor classes with the 29 operators that build, read or copy a
sparse tensor, and the schemas of today's ops with their autograd formulas. The full
list, with reasons: [A.3](#a3-what-stays-in-core-and-why).

Everything else moves: the kernels (~26 k of the ~27 k lines[^size]), `torch/sparse/*`
and `SparseAdam`.

### The asks of core

The sparse project writes every core PR this plan needs. What it asks of core is
review, and only what the project cannot do itself:
1. **A new repo**, `pytorch/sparsetorch`, for the owners, under the governance above
   (question 2).
2. **Review of four stable-ABI pieces in R1.** The pieces: `Scalar`, accessors and a
   setter for a sparse tensor's parts, the current cuSPARSE handle, and typed exceptions
   ([A.4](#a4-what-core-adds), question 6). Without them, the fallback is G (question
   7).
3. **A CI job that builds the package from a commit pin** during R2–R3, as for triton
   ([A.1](#a1-how-the-package-reaches-torch), question 4). The sparse project writes
   and maintains it; core hosts it.
4. **Small core PRs for new backends**, with no kernel code in them
   ([A.5](#a5-new-backends-the-core-pieces)). Core accepts such PRs when their only
   purpose is to let the sparse project support a backend (question 3). One of them
   gives the generic sparse kernels to any backend key, which B and G need too.

Beyond these four, the asks are review of the smaller core changes listed in
[A.4](#a4-what-core-adds), and of the core-side part of each deletion PR ([The
migration](#the-migration)). Nothing is asked of TorchAO (A.4).

## **Metrics**

- Sparse PRs and reviews reaching core reviewers: should drop to the part kept in core.
- User stories behind the priorities: every op added to the stable namespace after R0,
  and every new backend, cites at least one actively maintained public project that
  needs it. Stories are recounted at each release, by layout and device, with the method
  of [B.1](#b1-method), against this RFC's count: 1,798 projects, 1,052 maintained.
- Sparse share of pytorch CI time: from 1.6 % towards zero after R3.
- The 122 sparse tests in core's other test files (autograd, serialization,
  distributed, …): with the package, today's results; without it, a test that reaches
  a moved kernel is skipped, and the rest pass.

## **Drawbacks**

| Risk or cost | If it happens |
|---|---|
| No owner for the sparse project | Nothing moves ([Owners and governance](#owners-and-governance)). |
| Owners leave after the move | As today: sparse falls back to maintenance mode, now in the package torch depends on. The minimal-effort owner is an owner who stays on, or core if it agrees ([Owners and governance](#owners-and-governance), question 2). The code moved verbatim, so the move can be reversed the same way. |
| Work stalls or plans change (R2–R3) | Each release still works; families not yet deleted from ATen can switch back. |
| torch depends on a package that depends on torch | pytorch CI builds one more wheel from a commit pin, on every platform. The release order of [A.1](#a1-how-the-package-reaches-torch) is untested, and so is conda. |
| A broken package breaks `import torch` | The package's CI runs `import torch` on every platform; a release blocker. |
| The four stable-ABI pieces don't land | Fall back to G. Or move kernels through the Python route ([A.9](#a9-evidence-kernel-families-moved)), at its cost per call. |
| Kernels get slower | `addmm` costs the same on CUDA at nnz 10⁶. `mul` was 1.3–4.6× slower: an extra copy, a dispatcher call per metadata read and, on CUDA, a detour around a kernel the experiment left in core. Accessors and a setter, piece (2), are expected to remove most of it, and that kernel moves with `mul`; not measured yet ([A.9](#a9-evidence-kernel-families-moved)). |
| Distributions that package torch must package `sparsetorch` too, from R2 | It builds against torch with the same compilers and CUDA libraries. Until R2, nothing changes for them. |

Not a breaking change for Python users: saved files, exported programs and compiled
models keep loading ([A.2](#a2-how-torch-calls-the-package-and-the-hard-cases)). A C++
program without Python reaches the moved kernels through the same `at::` functions once
it loads the package's library; a C++ namespace and an agreement on linking can follow
if C++ users need them.

## **Alternatives**

| | Option | What moves | Who reviews sparse PRs | Role in this proposal |
|---|---|---|---|---|
| 0 | As today | nothing | emeritus sparse and core maintainers | rejected: no owner |
| 0+ | New sparse maintainers in core | nothing | the new maintainers, and core for shared files | rejected: every new op and backend stays core's to test and keep |
| **A** | Separate project | kernels and Python ([What stays in core](#what-stays-in-core)) | the sparse project's owners | **the destination** |
| B | Core keeps the five layouts; other backends as plugins | the 14 MPS-only kernels, and later backends | core, for CPU/CUDA | rejected: ~26 k lines of CPU/CUDA code and its PRs stay in core with no owner |
| C | The functorch path | Python, into an in-repo package | core | the first step (R1) |
| D | Sparse tests in their own config | nothing | core | at once (R0) |
| E | The maintenance policy | nothing | core | at once (R0); the end state if no owner commits |
| F | Semi-structured to TorchAO | 2:4 ops and classes | TorchAO | rejected: TorchAO removed its sparsity package for lack of use ([pytorch/ao#4728][ao4728]); 2:4 moves with A |
| G | The torch-xpu-ops model | kernel sources, to a repo pytorch pins and builds in tree | the new repo for kernels; core for registrations; the sparse project bumps the commit pin | the fallback: every new op or backend still needs a core PR, and the commit pin a bump about weekly; no packaging change ([A.7](#a7-a-against-g)) |

**Why not B alone.** B settles the question for new hardware only; solvers, missing ops
and SciPy interop are not hardware. B's one core change
([A.5](#a5-new-backends-the-core-pieces)) is asked for here too. A new backend means
notable changes to core, so core takes part in discussing and reviewing it ([Owners and
governance](#owners-and-governance)); once core approves it, the core changes it needs
can land (ask 4).

## **Prior Art**

- **functorch** moved the other way: a separate repo merged into pytorch. `functorch`
  is now an alias over `torch._functorch` ([functorch/\_\_init\_\_.py:6-7][ft]), the
  route R1 uses.
- **torchaudio**: a maintenance phase, and a stable-ABI wheel that works with later
  torch releases.
- **TorchAO** builds stable-ABI modules with `TORCH_TARGET_VERSION`.
- **triton**: a pinned dependency of torch, built by pytorch CI.
- **torch-xpu-ops**: option G.
- **`torch.backends` entry points**, [pytorch#122468](https://github.com/pytorch/pytorch/issues/122468),
  made for out-of-tree devices ([torch/\_\_init\_\_.py:3531-3546][autoload]); R2 loads
  the kernels this way.
- **RFC-0050**: cross-repository CI for out-of-tree backends.

## **How we teach this**

- Users keep `torch.sparse`. Nothing changes in their code.
- The torch.sparse docs stay at the same URL.
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
2. Governance of the sparse project: who owns it, and who breaks ties on design
   decisions? While there is a single owner, will core reviewers take part in
   architecture review ([Owners and governance](#owners-and-governance))? Can pytorch's
   automated reviewer run on the new repo too? If no owner stays on, would core be the
   minimal-effort owner, under its own AI policy? Decided at R0.\
   Ask: albanD, malfet
3. Will core accept the small core PRs that a new sparse backend needs (ask 4)?\
   Ask: malfet, EikanWang
4. Will pytorch CI host a job, written and maintained by the sparse project, that builds
   the package from a commit pin during R2–R3, on every platform torch ships?\
   Ask: atalman, malfet
5. Is four releases the right pace, or should R2 and R3 be one?\
   Ask: malfet
6. Will core review the four stable-ABI pieces ([A.4](#a4-what-core-adds)) in R1,
   written by the sparse project?\
   Ask: janeyx99; for the cuSPARSE handle, also eqy
7. If not, would core take G for sparse, as for XPU, with the sparse project bumping the
   commit pin?\
   Ask: malfet, EikanWang
8. The XPU maintainers ([merge_rules.yaml:552-563][xpumr]): under A, would you run the
   package's shared test suite on XPU, and register XPU's 28 sparse kernels and 4
   dispatch stubs through the package's documented interface ([A.6](#a6-xpu-and-mps)),
   each project bumping its own commit pin in pytorch?\
   Ask: EikanWang, guangyey
9. Where does the package's CI run after R3, and who funds its GPU and macOS runners?
   Sparse tests are a small share of CI, so pytorch's runners, on demand rather than on
   every PR, may do. MPS needs an owner with a particular interest in it.\
   Ask: atalman, malfet

**Out of scope, for later:** making the package an optional install; replacing the
masked-semantics ops with `torch.masked`; `torch.compile` for sparse; a cross-library
sparse standard, which the Array API leaves out
([data_interchange.rst:88-98][aa-sparse],
[array-api#840](https://github.com/data-apis/array-api/issues/840)) and which must
settle what unspecified elements are: zero, or a fill value
([pytorch/rfcs#8](https://github.com/pytorch/rfcs/pull/8)); torch.sparse could
contribute its layout model and this question to it, with the Sparse Array
Consortium.[^consortium]

## Appendix: Technical feasibility

Details that matter when implementing the plan, not when deciding its direction.

### A.1 How the package reaches torch

From R2, pytorch CI builds it from a commit pin, as for triton
([build-triton-wheel.yml][triton], [binary_populate_env.sh:73-87][bpe]). A family's
kernel lands in the new repo, and reaches torch in one pytorch PR, the commit-pin bump.
At release, the package is built on torch N−1's stable ABI and released first; torch N
requires it, `==`. With `SPARSETORCH_USE_ATEN=1` (name to be decided) it registers
nothing, and ATen's copies serve. The package's CI runs the sparse tests twice, with its
kernels and with the switch, against the current torch, so the way back stays tested
while ATen's copies exist.

**The alias (R1 on)** re-exports, where each is documented today: `torch.sparse`;
`torch.optim.SparseAdam` ([optim/\_\_init\_\_.py:31][sa]); and
`torch.optim.functional.sparse_adam` ([functional.py:71][fsa]). Names that stay need
none: the `Tensor` methods, `torch.sparse_*_tensor`, `Embedding(sparse=True)` and
`torch.cuda.sparse` ([cuda/sparse.py:1][cudasparse]).

### A.2 How torch calls the package, and the hard cases

**How it works.**
- Core keeps the schemas of today's ops and their autograd formulas
  ([Vision](#vision)). A sparse call goes through
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
| Package absent after a deletion | Each deleted kernel leaves a placeholder kernel in ATen, generated (the experiments call it a stub), that says "install `sparsetorch`". A backend fallback would not do: composite kernels run ahead of it (37 ops, plus 53 `structured_delegate` ops on the compressed layouts), it doesn't cover dense keys, and it would claim ops that never moved. The package overrides the placeholders, so the dispatcher's override warning must be silent for them. |
| Dense-to-sparse conversions ([native_functions.yaml:7585][nf-tosparse]) | One family. The package registers it under the dense keys too, and its placeholders cover the dense keys. |
| MPS kernels | Ported to the stable ABI's Metal shim ([shim_mps.h:20-101][shim-mps]). This route is untested. ATen's copy stays behind the switch until a macOS runner tests the port. |
| MKL and cuSPARSE | cuSPARSE through the stable ABI (R1). For MKL, the package's owners choose between torch exposing its MKL sparse calls through the stable ABI and their own CPU kernels. |
| Meta kernels | Registered with each family. 150 of 153 are the CPU kernel itself. |
| Inductor's 2:4 lowering ([mm.py:1128-1174][mm-lower]) | Stays in core. It lowers to the `aten` 2:4 ops, whose schemas stay; their kernels move with the 2:4 families, and the kernel choice goes through a registration ([A.4](#a4-what-core-adds)). |
| AOTInductor's C shim for `_to_sparse` ([c_shim_cpu.h:47][aoti-shim]) | It calls the CPU and CUDA kernels directly ([gen_aoti_c_shim.py:458-482][aoti-gen]), not through the dispatcher. After deletion it would reach the placeholder even with the package installed, and torchgen refuses to delete it outright, since compiled models call it by name. The shim keeps its name and signature, so compiled models still load; its body becomes a dispatcher call (a torchgen change, R2). |
| A C++ program without Python | It calls the same `at::` functions. The dispatcher reaches the package's kernels once the program loads the package's library, whose registrations run as it loads (`STABLE_TORCH_LIBRARY_IMPL` in [`sparse-ext.diff`](RFC-00xx-assets/sparse-ext.diff)). Not tested. A C++ namespace and an agreement on linking can follow if C++ users need them. |
| Compressed `mm` on CUDA calling triton ([SparseBlasImpl.cpp:241-255][bsr-triton]) | Moves with the compressed `mm` family (R3). The package registers the kernel from Python, and that kernel chooses between triton and C++. `torch.cuda`'s registration of the triton ops leaves core. |

Replacing the masked-semantics ops with `torch.masked` is a later decision under
pytorch's deprecation policy, not part of the move.

**The compiler stack.** `torch.compile` refuses sparse tensors today, at the input
([builder.py:3449-3460][dynamo-refuse]). This proposal doesn't change that, and
doesn't close the door: there is interest in compiling sparse code.[^compile]
Support can be added later by the sparse project, together with the compiler owners.
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
| `SparseTensorImpl`, `SparseCsrTensorImpl`, with the 29 operators that build, read or copy a sparse tensor | core autograd reads them |
| Generic gradient accumulation | picks in-place or out-of-place add by layout ([accumulate_grad.h:188-239][acc]) |
| DDP bucketing of sparse gradients | set in the reducer ([reducer.cpp:91-162][red]) |
| The serialization name `torch._utils._rebuild_sparse_tensor` | saved files name it ([_utils.py:369][rebuild]) |
| `torch.backends.cusparselt` | documented without a prototype label ([backends.md:286-301][cslt]); cuSPARSELt stays a torch wheel dependency ([torch/\_\_init\_\_.py:402][preload]) |
| `torch.masked` | prototype, mostly strided |
| The cuSPARSE handle pool | dense MAGMA code uses it ([MagmaUtils.h:35][magma]); exposed through the stable ABI in R1 |
| Schemas of general-purpose ops, with their autograd formulas | torch's general-purpose ops aim at the Array API (many of torch's machine-learning ops are outside it), and sparse tensors aim at the semantics of strided ones, so the op is core's; the sparse project registers kernels for it. Schemas of sparse-specific ops (`torch.sparse.*`, the sparse factories, `to_sparse_*`) may move to the sparse project after the move ([Vision](#vision)) |
| The c10d sparse allreduce | it needs `ProcessGroup` and the Gloo backend ([Ops.cpp:795-801][ops]) |

### A.4 What core adds

| Release | Core adds |
|---|---|
| R0 | the `sparse` test config; a marker for the 122 sparse tests in core test files |
| R1 | `sparsetorch` in the wheel's `packages`, as `functorch` ([pyproject.toml:188][pyproj]); a merge rule and CODEOWNERS for the package. The rule covers the sparse-only files, the commit-pin file among them, as the XPU rule covers `third_party/xpu.txt` ([merge_rules.yaml:552-563][xpumr]), and sparse owners approve changes to them. A deletion PR also changes files the rule doesn't cover, so a core reviewer approves those parts: `native_functions.yaml`, which core reviews anyway; files shared with dense code, where 18 of the 100 families have a kernel[^families] (`TensorShape.cpp`, `TensorConversions.cpp`, `ReduceOps.cpp` and four more); and `build_variables.bzl`, which names the sparse sources one by one ([build_variables.bzl:1574-1587][bv]). **In the stable ABI: (1) `Scalar` as a kernel argument and for calling ops (in progress: [pytorch#203903](https://github.com/pytorch/pytorch/pull/203903), a draft); (2) accessors and an in-place setter for a sparse tensor's indices, values and nnz; (3) the current cuSPARSE handle; (4) typed exceptions: today a stable kernel's `IndexError` reaches Python as `RuntimeError`, though its message arrives ([pytorch#180135](https://github.com/pytorch/pytorch/pull/180135)).** Any stable-ABI extension with a `Scalar` argument or a vendor library handle needs the same. |
| R2 | a CI job that builds the package from a commit pin, and the dependency in the wheel requirements; a placeholder kernel, generated for each deleted sparse kernel, that says "install `sparsetorch`", and no override warning when the package replaces a placeholder ([OperatorEntry.cpp:213-225][override-warn]); AOTInductor C shims that call the dispatcher for moved ops, with their headers unchanged ([gen_aoti_c_shim.py:458-482][aoti-gen]); a way to skip sparse OpInfo samples without it; Inductor's 2:4 kernel choice through a registration ([mm.py:203-205][inductor]) |
| R3 | an extension point through which the package provides `Adagrad`'s step for sparse gradients, sparse code now in core's `torch/optim/adagrad.py` |

The sparse project writes these changes; core reviews them.

Nothing is asked of TorchAO. Its float8 2:4 formats run its own CUTLASS kernel or,
on ROCm, `torch._cslt_compress` and `torch._cslt_sparse_mm`
([float8_sparse_2x4_1d_data_1d_metadata_tensor.py:204-215][ao-cslt]), whose
schemas stay in core, so they keep working through the package.

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

**Generic kernels for any backend.** Most sparse kernels are device-generic: 147 of the
161 `SparseMPS` entries and 284 of the 312 XPU entries name the CPU/CUDA kernel, each by a
hand-written line. A device outside core (`PrivateUse1`) can construct COO and CSR tensors
([tensor_new.cpp:558-575][tn-pu1], [SparseCsrTensor.cpp:371-372][csr-pu1]), but no sparse
kernel is registered for its keys. A torchgen change that registers the generic kernels for
every sparse key serves A, B and G alike. The generic code also calls 4 sparse dispatch
stubs, the per-device hooks of [SparseStubs.h:14-23][stubs]: XPU fills all 4 and MPS fills 3, through core's
internal headers, so the stubs become a documented entry point too
([`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §5-6).

Core is asked to accept these PRs: ask 4 of [The asks of core](#the-asks-of-core).

### A.6 XPU and MPS

Most XPU and MPS entries name the same generic kernel as CPU/CUDA
([A.5](#a5-new-backends-the-core-pieces)); all 13 `SparseCsrMPS` entries do. The package
registers each moved family for every key that names it today. pytorch's XPU and MPS
jobs test it during R2–R3; after that, the package needs XPU and macOS runners, or
those maintainers test it (question 9).

The sparse share of CI time is small partly because sparse is barely tested on backends
other than CPU and CUDA:
on MPS, only COO, and only in trunk jobs, never in a PR's default checks.[^mps] Full MPS
coverage would cost about 0.45 h of Apple-silicon runner time per run (about a fifth of
today's MPS test run, 1.8–2.7 h), on a pool that pytorch CI keeps off ordinary PRs.
Under this proposal, the package's own CI decides which backends to cover; where it
runs, and who funds its runners, is question 9.

**Cooperation with torch-xpu-ops.** It holds 28 sparse kernels of its own (COO and CSR
matmul, softmax, coalesce, CSR reductions, `_spsolve`) and fills all 4 sparse dispatch
stubs; the other XPU entries run core's generic kernels. It runs core's sparse test classes
on XPU by importing them ([`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §5).
Under A, the two projects share:
- one device-generic test suite in the package, which torch-xpu-ops runs on XPU with its
  own skip list;
- the registration interface of [Vision](#vision), so that XPU's kernels and stubs
  register against the package without core's internal headers;
- a rule that each project bumps its own commit pin in pytorch.

### A.7 A against G

| | A | G |
|---|---|---|
| Maintenance | Sparse PRs leave core. Coupling is through the aten schemas the package calls (~30 for the two families moved), which change under BC rules. A commit pin in CI during R2–R3 only. | Registrations stay in core: a new op or backend needs a core PR. Internal ATen changes can break the sparse repo. A commit pin to bump about weekly: XPU's moved 61 times in 12 months, with 3 reverts.[^xpu] |
| Users | No Python name breaks. Risks: a dependency cycle; a broken package breaks `import torch`. | No change. |
| Effort | The four stable-ABI pieces first. | Files move; a pin and a CMake block ([CMakeLists.txt:1266-1336][xpucmake]). |
| CI | Sparse leaves pytorch's jobs. | Sparse is built in every job. |

G wins on users and effort; A on maintenance. G couples core to sparse for good, and its
weekly pin needs a sparse owner to bump it. XPU's model works because it has dedicated
maintainers.

### A.8 Technical risks

The risks a reader decides on are in [Drawbacks](#drawbacks); these are the technical
ones.

| Risk | If it happens |
|---|---|
| Hidden sparse paths in core (259 files mention sparse) | Found only by the 122 core sparse tests run without the package. |
| `TORCH_DEVICE_BACKEND_AUTOLOAD=0` turns sparse kernels off too | Sparse gets an import-time hook of its own. |

### A.9 Evidence: kernel families moved

COO `mul` and `addmm`, CPU and CUDA, were removed from ATen and registered from a
stable-ABI package, `sparse-ext` (target 2.13), which stands in for `sparsetorch` and
which `import torch` loads. Timed on these two families
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
- `addmm` costs the same on CUDA at nnz 10⁶ (1.00×); `mul` is 1.3–4.6× slower. `mul`
  copies indices and values once more for lack of a setter, reads each piece of
  metadata through a dispatcher call, and on CUDA goes around the intersection kernel,
  a dispatch stub the experiment left in core. Accessors and a setter, piece (2), are
  expected to remove most of the gap; that is not measured. In a full move the
  intersection kernel moves with `mul`. `addmm_cuda.cu` keeps a handle pool of its own
  for lack of piece (3).

An override experiment kept ATen's `mul` kernels and registered the package's over them
([`sparse-ext-override.diff`](RFC-00xx-assets/sparse-ext-override.diff)). Results were
identical with the package, with a switch back to ATen, and without the package.
Removing a Python registration at run time restored ATen's kernel.

A deletion experiment deleted five families from ATen, CPU and CUDA: `sin`, `abs`, `log1p`,
`_sparse_softmax` and `_to_sparse_csr`, the last under dense keys too. The package
registered them ([`pytorch-delete.diff`](RFC-00xx-assets/pytorch-delete.diff),
[`sparse-ext-delete.diff`](RFC-00xx-assets/sparse-ext-delete.diff)). With the package,
2,171 targeted tests gave the same result as main, per test, and gradcheck passed
through core's unchanged formulas. Without it, every failure was an error, none a wrong
result. `_sparse_softmax` needed a C++ bridge for lack of the fourth piece, typed
exceptions: a stable kernel's `IndexError` reaches Python as `RuntimeError`. A backend
fallback missed composite and dense-key kernels, hence the placeholders.

A compiler-stack experiment ran on the deletion experiment's build
([`pytorch-aoti.diff`](RFC-00xx-assets/pytorch-aoti.diff),
[`sparse-ext-aoti.diff`](RFC-00xx-assets/sparse-ext-aoti.diff)). With the package,
`test/export/test_sparse.py` (207 tests) and the dynamo sparse tests (5) gave the same
result as main, per test; without it, the 52 export tests that reach `abs` or
`_to_sparse_csr` raised an error. A program over `abs` on COO, saved by `torch.export`
on main, loaded on the modified torch and ran only with the package
([`export-save.py`](RFC-00xx-assets/export-save.py),
[`export-load.py`](RFC-00xx-assets/export-load.py)). `_to_sparse`'s dense kernel then
became a placeholder, since torchgen refuses to delete it outright. AOTInductor models
compiled on main and on the modified torch hit the placeholder even with the package loaded,
because the C shim calls the CPU kernel directly. Generating the shim's body as a
dispatcher call, headers unchanged, made both reach the package
([`aoti-compile.py`](RFC-00xx-assets/aoti-compile.py),
[`aoti-run.py`](RFC-00xx-assets/aoti-run.py)). CUDA was checked at the C shim, not
through a compiled CUDA model: triton was not installed.

## Appendix: User stories in detail

The evidence behind the [Motivation](#motivation), counted in November 2026. The full
report, with the 50 highest-ranked projects read by hand, is
[`user-stories.md`](RFC-00xx-assets/user-stories.md). The comparison with other
features is [`usage-baseline.md`](RFC-00xx-assets/usage-baseline.md), and the
measurements of when sparse pays off are
[`when-sparse-pays-off.md`](RFC-00xx-assets/when-sparse-pays-off.md). Backends, the
scientific users, users' voices, other sparse libraries, XPU and MPS are in
[`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md).

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

COO is used by 1,203 projects and CSR by 443; CSC, BSR, BSC and 2:4 by 32 to 97 each.
Most code follows its input's device (1,267 projects); 727 name CUDA, 52 MPS, 15 ROCm
and 11 XPU.

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

Sparse pays off only where data are sparse enough ([B.4](#b4-when-sparse-pays-off)), so a
user base smaller than linalg's, and concentrated where data are sparse, is expected.

### B.3 What the code shows

- **Small parts of large projects.** In many top projects sparse is one optional
  component: a model adapter, a loss, an experimental path.
- **A data format, not compute.** Some use sparse tensors only to build or ship data
  and densify at once: CSR as a wire encoding, COO as a scatter idiom.
- **MPS is avoided where it would be needed.** suite2p falls back to CPU "until sparse
  works on MPS", instanseg moves to CPU, and omicverse goes to MLX on Apple hardware. 52
  projects' sparse code names MPS. A core reviewer knew of no actively developed project
  that uses compressed layouts on MPS; a user answered with one in the same
  thread.[^201801]
- **An old aim, and old requests.** An early proposal, still a draft
  ([pytorch/rfcs#6][rfc3], 2020), aimed at a sparse tensor that is a drop-in replacement
  for a dense one in every operation. [pytorch#10043][usecases], open since 2018, collects
  use cases, such as row slicing of CSR tensors on GPU in single-cell biology
  ([comment][singlecell]).
- **Libraries with layouts of their own.** The PyTorch libraries that brought sparse
  layouts of their own, torch_sparse and DGL, last released in 2023 and 2024. The
  libraries still maintained are narrow (solvers and gradients), have no autograd
  (SciPy, CuPy, PyData `sparse`), or are experimental and NVIDIA-only (nvmath-python's
  sparse module).
- **`spsolve` is hard to rely on.** torch-sla probes for it before use. A Stack Overflow
  question on an RTX 40 card ([79227337](https://stackoverflow.com/questions/79227337))
  shows the cuDSS requirement in practice.
- **Requests.** Of the 75 issue and forum records, 13 name a project and 5 of these
  are requests. On MPS, they ask for CSR (3), COO (3) and BSR (1); on 2:4, for batched
  matrices, Conv2d and training.
- **Removals in 2026.** TorchAO's `sparsity/` package, at HEAD, "due to lack of
  widespread usage of this feature" ([pytorch/ao#4728][ao4728]); its latest release,
  v0.18.0, still has it. torch keeps the semi-structured kernels it called and the BSR
  layout and Triton kernel that TorchAO's own copy came from, and TorchAO keeps float8
  2:4 formats in its quantization; so 2:4 moves with the sparse project, not to TorchAO (F).
  NeMo's sparse loss, with "older and non-maintained models". PhysicsNeMo's regridding
  plugin, in a dependency clean-up. xformers' cuSPARSELt 2:4 code, in favour of
  torch.sparse.

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

### B.6 Backends: cuSPARSE and cuSPARSELt

- **cuSPARSE is broad.** 764 projects call a sparse matmul (`torch.sparse.mm`,
  `torch.sparse.addmm`, `sampled_addmm`, `smm`, `hspmm` or `sspaddmm`). Between 239 (the op in a file
  that names CUDA) and 627 (any sign of CUDA, or device-generic code) run it through
  cuSPARSE; both are estimates.
- **cuSPARSELt is narrow.** 32 projects use 2:4: 13 call cuSPARSELt explicitly, and 19
  use `to_sparse_semi_structured`, which picks it by default; 8 of the 32 force CUTLASS.
  29 of the 32 are in deep learning or unclassified.
- **The cost to pytorch,** 2023-11 to 2026-11: 72 issues and PRs on sparse code that uses
  the two libraries, 57 on building and packaging them (CUDA and cuSPARSELt version bumps,
  wheel preloads), and 110 commits to the 17 files that hold their code, 40 of them to
  cuSPARSELt's.

Details: [`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §1.

### B.7 Scientific users and SciPy

Of the 489 scientific projects whose code was scanned, 312 (64 %) also use scipy.sparse,
against 47 % in a sample of 145 deep-learning projects. 201 use both libraries in one
file, mostly to convert between them, and 112 call `scipy.sparse.linalg`: `eigsh` (50),
`spsolve` (45), `cg` (28), `splu` (25). So in science torch.sparse is mostly the GPU and
torch side of matrices that SciPy builds and solves, not a replacement for SciPy.
Details: [`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §2.

### B.8 torch.sparse beside other sparse libraries

The 146 Array API 2025.12 functions that take an array, each called once on a small
sparse input ([`cross-library-probe.py`](RFC-00xx-assets/cross-library-probe.py)):

| | torch, any layout | SciPy COO | PyData `sparse` COO | CuPy CSR |
|---|---|---|---|---|
| works | 53 | 55 | 108 | 32 |
| densifies | 0 | 13 | 3 | 1 |
| fails | 93 | 78 | 35 | 113 |

- **No implicit conversion.** torch densified nothing: where it has no kernel, it raises.
  SciPy densified in 13 cases. So torch's failures are its gaps, not hidden conversions.
- **The gap to PyData:** 61 functions work in PyData but in no torch layout. 20 of them
  don't map zero to zero, and torch refuses them by design; 39 fit zero semantics and are
  missing; 2 depend on their arguments.
- **What only torch has:** GPU support and autograd, together with batched and hybrid
  layouts.
  It has no iterative solvers, which SciPy and CuPy have, and its API is beta.
- **Finch:** its Python package (0.7.0, "Under construction") failed every call on sparse
  input, and is not counted.

Details, with each library's layouts, semantics and status:
[`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §4.

### B.9 Where users' voices go

The maintainers (the owner and the top 3 contributors) of 69 of 1,702 census projects,
4.1 %, have authored or commented on a pytorch issue or PR labeled `module: sparse`; of
the 36 projects with 10 or more PyPI dependents, 6 have. In the 634 projects whose code
was scanned, 22 comments in 20 projects record a torch.sparse gap worked around locally
(MPS, half-precision matmul, CSR operations, `spdiags` on CUDA, solves on HIP), and none
links a pytorch issue. Matching is by exact GitHub login, so these are lower bounds.
Details: [`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §3.

[^prs]: GitHub search, 2026-11-16: `repo:pytorch/pytorch is:pr created:2025-11-16..2026-11-16 label:"module: sparse"`,
    and the same with `label:"release notes: sparse"`; against 891 random PRs of the
    window with neither label. Human: not a bot, not the author. Time to land: the 95 %
    interval of the difference in medians is −0.1 to +6.6 days.
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
[^prfiles]: The files of the 119 sparse PRs counted for the reviews, from the GitHub API, 2026-11-21,
    against the patterns of the "Sparse" merge rule ([merge_rules.yaml:159-180][mr]).
    The 12 that change `native_functions.yaml`: 5 for MPS, 3 for XPU, and 4 others.
[^families]: `native_functions.yaml` at the commit above, grouped by op name with the
    overload and the in-place `_` dropped. The 29 operators that stay in core are not
    counted; backward ops count as families of their own (5). Of the 100, 18 have a
    kernel defined outside the files that the "Sparse" merge rule covers
    ([merge_rules.yaml:159-180][mr]), in seven files shared with dense code:
    `TensorShape.cpp`, `TensorConversions.cpp`, `ReduceOps.cpp`, `BinaryOps.cpp`,
    `TensorFactories.cpp`, `cuda/Indexing.cu` and
    `cuda/ValidateCompressedIndicesKernel.cu`.
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
[^consortium]: [NSF award 2612044](https://www.nsf.gov/awardsearch/showAward?AWD_ID=2612044),
    "POSE: Phase I: Accelerating the Sparse Array Programming Ecosystem", 2026-08 to 2027-07.
    The [Sparse Array Consortium](https://github.com/SparseArrayConsortium) lists Finch,
    Binsparse, Array API sparse extensions and the Sparse Array Programming Suite, whose
    PyTorch wrapper works around the ops torch.sparse lacks
    ([saps_pytorch.py:196-197](https://github.com/SparseArrayProgrammingSuite/SparseArrayProgrammingSuite/blob/8c62415a87305e23db24f12996f34db73dc74d8a/frameworks/saps_pytorch.py#L196-L197),
    [444-446](https://github.com/SparseArrayProgrammingSuite/SparseArrayProgrammingSuite/blob/8c62415a87305e23db24f12996f34db73dc74d8a/frameworks/saps_pytorch.py#L444-L446)).
[^lasts]: Commits per year on GitHub, 2019–2026
    ([`sparse-landscape.md`](RFC-00xx-assets/sparse-landscape.md) §8): DGL, MinkowskiEngine,
    MegaBlocks and stk, each led by one organization and without written governance,
    fell to near zero, and nobody took over. SciPy, vLLM and DeepSpeed, with written
    governance and several backing organizations, kept going.

## Resolution

To be filled in after the discussion.

[poi]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/community/persons_of_interest.md?plain=1#L150-L156
[mr]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/.github/merge_rules.yaml#L159-L180
[prlife]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/CONTRIBUTING.md?plain=1#L124-L204
[greenlight]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/CONTRIBUTING.md?plain=1#L672-L686
[aipol]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/AI_POLICY.md?plain=1#L29-L32
[bv]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/build_variables.bzl#L1574-L1587
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
[ao-cslt]: https://github.com/pytorch/ao/blob/a791dcb26432500f99d40271c81ee8e2753cf99a/torchao/quantization/quantize_/workflows/float8/float8_sparse_2x4_1d_data_1d_metadata_tensor.py#L204-L215
[singlecell]: https://github.com/pytorch/pytorch/issues/10043#issuecomment-966518421
[aoti-shim]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/inductor/aoti_torch/generated/c_shim_cpu.h#L47
[bsr-triton]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseBlasImpl.cpp#L241-L255
[dynamo-refuse]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_dynamo/variables/builder.py#L3449-L3460
[dynamo-export]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/_dynamo/config.py#L522-L526
[aoti-gen]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torchgen/gen_aoti_c_shim.py#L458-L482
[aa-sparse]: https://github.com/data-apis/array-api/blob/ff497ed83220372eb689a4fb0878c831470f93bc/spec/2025.12/design_topics/data_interchange.rst?plain=1#L88-L98
[tn-pu1]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/utils/tensor_new.cpp#L558-L575
[csr-pu1]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseCsrTensor.cpp#L371-L372
[stubs]: https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseStubs.h#L14-L23
