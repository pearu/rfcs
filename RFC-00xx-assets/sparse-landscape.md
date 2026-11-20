# torch.sparse in its landscape: backends, scientific users, users' voices, other libraries, XPU and MPS, comparable projects

Counted in November 2026. pytorch is at
[`a67a74d3e7`](https://github.com/pytorch/pytorch/tree/a67a74d3e71bf2891b0d02b126fe7390090e1f1c).
torch-xpu-ops is at the commit pytorch pins there,
[`8c125b2e`](https://github.com/intel/torch-xpu-ops/tree/8c125b2e5dcd10571b49567cb7f6819fa7dc42b6).
"The census" is the 1,798 projects of [`user-stories.md`](user-stories.md).

All counts of projects that use something are inferred from their code, not measured at run
time.

## 1. Backends: cuSPARSE is broad, cuSPARSELt is narrow

**cuSPARSELt.** 32 of the 1,798 projects use 2:4 sparsity.
- 13 call cuSPARSELt explicitly (`torch._cslt_*`, `torch.backends.cusparselt`).
- 19 use `to_sparse_semi_structured` or `SparseSemiStructuredTensor`. These pick cuSPARSELt by
  default; CUTLASS is used only if `_FORCE_CUTLASS` is set
  ([semi_structured.py:63, 372-376](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/sparse/semi_structured.py#L372-L376)).
- 8 of the 32 set `_FORCE_CUTLASS = True` in a file the census cites, so they choose CUTLASS.
  This is a lower bound.
- 29 of the 32 are in deep learning or unclassified, and 3 are in scientific computing.

**cuSPARSE.**
- **Sparse matmul:** 764 projects call one of `torch.sparse.mm`, `torch.sparse.addmm`,
  `sampled_addmm`, `hspmm`, `smm` or `sspaddmm`.
  - On CUDA these run cuSPARSE, except `sspaddmm`/`smm`, which aren't implemented there
    ([SparseMM.cu:17-19](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/cuda/SparseMM.cu#L17-L19)).
  - Of the 764, 239 have the op in a file that also names `cuda`, 87 more name cuda elsewhere in
    the project, and 301 more run it in device-generic code.
- **Estimate:** between 239 and 627 projects run cuSPARSE. Both ends are estimates: the patterns
  miss the matrix-multiply operator and `torch.mm` on sparse inputs.
- **Not cuSPARSE:** conversions such as `to_sparse_csr` and reductions run torch's own kernels on
  CUDA.

**What the libraries cost pytorch, three years (2023-11-20 to 2026-11-20).**
- **Issues and PRs:** GitHub search finds 2,205 pytorch issues and PRs that mention cuSPARSE or
  cuSPARSELt in the title or body.
  - 2,040 mention them only in an environment dump (`nvidia-cusparse-cu12` and similar lines).
  - The other 165, read by title, fall into three classes:

| Class | PRs | Issues |
|---|---|---|
| sparse code: kernels, ops, tests, docs | 47 (33 cuSPARSELt) | 25 (13 cuSPARSELt) |
| building or packaging the libraries | 41 (36 cuSPARSELt) | 16 (10 cuSPARSELt) |
| incidental: a build log, or a refactor that touches the file | 14 | 22 |

- **Sparse-code PRs:** 28 authors. 12 of the 47 are by one contributor (semi-structured
  sparsity), and 12 are ROCm or hipSPARSE(Lt) work.
- **Sparse-code issues:** of the 24 written by people, 9 are by authors with no association to
  pytorch, 8 by collaborators and 7 by contributors.
  - The 2026 ones include three CUDA CSR matmul correctness bugs
    ([pytorch#202128](https://github.com/pytorch/pytorch/issues/202128),
    [#202134](https://github.com/pytorch/pytorch/issues/202134),
    [#204102](https://github.com/pytorch/pytorch/issues/204102)).
- **Packaging PRs:** CUDA version bumps, cuSPARSELt version bumps (0.5.2 to 0.8.1) and wheel
  preload fixes; 26 of 41 are by collaborators.
- **Commits:** 110 commits touched the 17 source files that hold pytorch's cuSPARSE and
  cuSPARSELt code (13 of them reverts). 40 of these touched the 4 cuSPARSELt files.
  - The 17 files are `aten/src/ATen/cuda/CUDASparse*`, `CuSparseHandlePool.cpp`,
    `aten/src/ATen/native/sparse/cuda/{SparseBlas,SparseBlasImpl,SparseBlasLegacy,SparseCUDABlas}*`,
    `SparseCUDATensorMath.cu`, `SparseMatMul.cu`, `cuSPARSELtOps.{cpp,h}`,
    `torch/csrc/cuda/shared/cusparselt.cpp` and `torch/backends/cusparselt/__init__.py`.

## 2. Scientific users and SciPy

**What was scanned.** The full Python source of 489 of the 525 scientific projects in the census,
at the commit the census cites. The other 36 have no GitHub repository (26) or failed to download
(10). For comparison, a seeded sample of 150 deep-learning projects was scanned (145 succeeded).
Comments and docstrings were blanked before matching.

| | scientific (489) | deep-learning sample (145) |
|---|---|---|
| also use scipy.sparse | 312 (64 %) | 68 (47 %) |
| cupyx.scipy.sparse | 24 | 1 |
| PyData `sparse` | 7 | 0 |
| call `scipy.sparse.linalg` | 112 | 13 |
| torch.sparse and scipy.sparse in the same file | 201 | 33 |
| torch.sparse code names a GPU device / device-generic only | 202 / 185 | 50 / 55 |

- **The `scipy.sparse.linalg` calls** in the scientific projects: `eigsh` 50 projects, `spsolve`
  45, `cg` 28, `splu` 25, `eigs` 17, `svds` 15, `gmres` 15, and fewer of the rest.
- **Hand check of 6 random files** with both libraries: 5 convert between them, and 1 has a
  project's own `to_sparse` method.
- **Reading:** in science, torch.sparse is the GPU and torch side of matrices that scipy.sparse
  builds or solves, not a replacement for scipy.sparse.

## 3. Where users' voices go

**Participation in pytorch's tracker.**
- **pytorch side:** 1,262 pytorch issues and PRs labeled `module: sparse` (all years), with
  their authors, commenters and reviewers. That is 828 people.
- **Census side:** each project's owner (if a person) and its top 3 contributors by commits,
  for the 1,702 projects where these are known.
- **Result:**
  - 69 projects (4.1 %) have someone who authored or commented on such an issue or PR.
  - 35 have someone who opened such an issue.
  - Among the 36 projects with 10 or more PyPI dependents, 6 do.
- **Limit:** matching is by exact GitHub login, so these are lower bounds.

**Gaps worked around in code.**
- **Code comments:** in the 634 scanned repositories (§2), 22 comments in 20 projects record a
  torch.sparse gap that the project works around itself. The gaps:
  - sparse on MPS (4 projects);
  - half or bfloat16 sparse matmul (3);
  - CSR addition, stacking, slicing and concatenation;
  - `spdiags` on CUDA;
  - sparse solve on HIP;
  - tensors over INT_MAX elements;
  - CSR backward in an old release.
- **Links:** none of the 22 comments links a pytorch issue within six lines.
- **Issues:** in a seeded sample of 100 census projects (98 searched), 9 issues in 3 projects
  record a torch.sparse limitation; one of them links a pytorch issue.

**Reading:** most users don't report torch.sparse gaps to pytorch. They densify, move to CPU,
convert layouts or write their own kernel, so the tracker understates the demand.

## 4. torch.sparse beside other sparse libraries

**Method:**
- **Function list:** the 146 functions of the Array API 2025.12 that take an array: 121 in the
  main namespace (14 more, such as `arange` and `finfo`, take none) and 25 in the linalg extension
  ([array_api_stubs/_2025_12](https://github.com/data-apis/array-api/tree/ff497ed83220372eb689a4fb0878c831470f93bc/src/array_api_stubs/_2025_12)).
- **The probe:** each function is called once on a 4×4 float64 sparse array, in each library.
  [`cross-library-probe.py`](cross-library-probe.py) does this.
- **Classes:**
  - **works:** the call returns the library's sparse type, or a scalar;
  - **densifies:** it returns a dense array;
  - **fails:** it raises, or the library has no such function.
- **Versions:** torch at the commit above, CPU and CUDA; SciPy 1.18.1; PyData `sparse` 0.19.3;
  CuPy 14.2.0 on an RTX 5060 Ti; NumPy 2.3.5.
- **Finch:** finch-tensor 0.7.0 (its README says "Under construction") raised on every operation
  with a sparse input, so it is assessed only by which names it defines.

**Coverage** (works / densifies / fails):

| | torch COO | torch CSR | torch CSR, CUDA | SciPy COO | SciPy CSR | PyData COO | CuPy CSR |
|---|---|---|---|---|---|---|---|
| element-wise (67) | 29/0/38 | 25/0/42 | 25/0/42 | 25/1/41 | 25/1/41 | 67/0/0 | 15/0/52 |
| manipulation (14) | 7/0/7 | 2/0/12 | 2/0/12 | 8/4/2 | 6/4/4 | 14/0/0 | 4/0/10 |
| statistical (9) | 1/0/8 | 1/0/8 | 1/0/8 | 5/2/2 | 5/2/2 | 7/0/2 | 4/0/5 |
| searching, sorting, set (13) | 0/0/13 | 0/0/13 | 0/0/13 | 3/6/4 | 3/6/4 | 4/3/6 | 3/1/9 |
| other main namespace (18) | 8/0/10 | 7/0/11 | 8/0/10 | 11/0/7 | 10/0/8 | 16/0/2 | 5/0/13 |
| linalg extension (25) | 6/0/19 | 1/0/24 | 3/0/22 | 3/0/22 | 3/0/22 | 0/0/25 | 1/0/24 |
| **total (146)** | **51/0/95** | **36/0/110** | **39/0/107** | **55/13/78** | **52/13/81** | **108/3/35** | **32/1/113** |

- **torch, any layout and device:** 53 of 146 work.
  - The CPU build measured has no MKL, so CSR-times-CSR matmul fails on CPU here. It works on CUDA, and
    on MKL builds.
- **Finch** defines 127 of the 135 main-namespace names and 23 of the 25 linalg names.
- **No implicit conversion in torch:** torch.sparse densified nothing. Where it has no kernel, it
  raises. SciPy densified in 13 cases, mostly through NumPy.
- **The gap to PyData:** 61 functions work in PyData COO but in no torch layout.
  - 20 of them don't map zero to zero (`exp`, `cos`, `log`, `divide`, `equal`, …). torch refuses
    these by design.
  - 2 depend on their arguments (`clip`, `where`).
  - The other 39 fit zero semantics and are missing implementations:
    - 17 element-wise (the binary bitwise and logical functions, `maximum`, `minimum`, `hypot`,
      …);
    - 12 structural (`reshape`, `broadcast_to`, `flip`, `roll`, `tile`, `tril`, `triu`, …);
    - 10 reductions, sorts and searches (`max`, `min`, `mean`, `argmax`, `sort`, …).
- **The other way:** 6 functions work in torch but not in PyData, all in the linalg extension.
  PyData has no `linalg` namespace.

**Layouts, semantics and status:**

| | torch.sparse | scipy.sparse | PyData `sparse` | cupyx.scipy.sparse | Finch 0.7.0 |
|---|---|---|---|---|---|
| Formats | COO; CSR, CSC, BSR, BSC; 2:4 | COO, CSR, CSC, BSR, DIA, DOK, LIL | COO, GCXS, DOK | COO, CSR, CSC, DIA | levels per dimension |
| n-D | COO n-D; compressed 2-D with batch dimensions | COO n-D; CSR and DOK up to 2-D | n-D | 2-D | n-D |
| Batch and hybrid (dense) dimensions | yes | no | no | no | by construction |
| Unspecified elements | zero; non-zero-preserving ops refused; −inf in `torch.sparse.softmax`; masked in `torch.masked` | zero | a fill value, any value (`exp` gives fill value 1) | zero | a fill value |
| Devices | CPU, CUDA, ROCm; partial MPS, XPU | CPU | CPU | CUDA, ROCm | CPU |
| Autograd | yes | no | no | no | no |
| Status | beta ([sparse.md:13-15](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/docs/source/sparse.md?plain=1#L13-L15)); no sparse-specific deprecation policy | stable, with a written deprecation policy | pre-1.0; declares Array API 2025.12 and runs its test suite in CI | stable; mirrors SciPy | "Under construction" |

**Sparse linear algebra.**

| | torch.sparse | scipy.sparse | cupyx.scipy.sparse | PyData `sparse` |
|---|---|---|---|---|
| Direct solvers | `spsolve` (CUDA with cuDSS, XPU); CSR `triangular_solve` | 6 | 5 | none |
| Iterative solvers | none | 12 | 7 | none |
| Eigen and SVD | `lobpcg` (forward only for sparse), `svd_lowrank` | `eigs`, `eigsh`, `lobpcg`, `svds` | `eigsh`, `lobpcg`, `svds` | none |

**Reading:**
- torch.sparse leads on batched and hybrid compressed layouts (Finch's levels can express such
  layouts, but its Python package doesn't run sparse operations yet), on GPU support with
  autograd, and on having explicit semantics for unspecified elements.
- It trails on coverage of the standard's functions (more than half of PyData's coverage is
  missing, mostly unimplemented), on solvers, and on API stability.
- A cross-library standard has to settle what torch.sparse decided one way: zero semantics with
  some operations undefined, or a fill value.

## 5. torch-xpu-ops and sparse

- **Registrations:** core's `native_functions.yaml` has 173 `SparseXPU` and 139 `SparseCsrXPU`
  entries over 196 functions. 284 of these 312 name the same generic kernel as CPU/CUDA.
- **XPU's own kernels:** the other 28 name kernels that torch-xpu-ops defines: COO and CSR matmul,
  softmax, coalesce, CSR reductions, `sampled_addmm`, `triangular_solve` and `_spsolve`, using SYCL
  and oneMKL sparse BLAS
  ([src/ATen/native/sparse/xpu](https://github.com/intel/torch-xpu-ops/tree/8c125b2e5dcd10571b49567cb7f6819fa7dc42b6/src/ATen/native/sparse/xpu)).
- **Stubs:** torch-xpu-ops fills all 4 of core's sparse dispatch stubs
  ([SparseStubs.h:14-23](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseStubs.h#L14-L23);
  [SparseBinaryOpIntersection.cpp:16-23](https://github.com/intel/torch-xpu-ops/blob/8c125b2e5dcd10571b49567cb7f6819fa7dc42b6/src/ATen/native/sparse/xpu/SparseBinaryOpIntersection.cpp#L16-L23),
  [SparseTensor.cpp:22](https://github.com/intel/torch-xpu-ops/blob/8c125b2e5dcd10571b49567cb7f6819fa7dc42b6/src/ATen/native/sparse/xpu/SparseTensor.cpp#L22)).
  It includes 9 of core's internal sparse headers.
- **Tests:** torch-xpu-ops imports core's `test_sparse.py` classes and runs them on XPU, with 11
  skipped cases
  ([test_sparse_xpu.py](https://github.com/intel/torch-xpu-ops/blob/8c125b2e5dcd10571b49567cb7f6819fa7dc42b6/test/xpu/test_sparse_xpu.py)).
  At this commit it runs a fork of `test_sparse_csr.py`; a later PR runs the upstream file instead.
- **Keeping in step:** 56 torch-xpu-ops issues have "sparse" in the title (34 of them from 2026).
  Some port a pytorch sparse fix to XPU by hand ("XPU alignment"), and others record test
  collection broken by a refactor of core's tests.
- **Reading:** the split of "one project owns the API, layouts, reference kernels and shared
  tests; device projects own optimized kernels" is already how XPU works. What is missing is a
  written registration interface. Today it is entries in core's yaml, internal stubs and internal
  headers, all compiled in core's tree.

## 6. Option B, measured

- **What moves:** MPS has 161 `SparseMPS` and 13 `SparseCsrMPS` entries. 147 and 13 of them
  name the generic CPU/CUDA kernel. So B moves only the 14 MPS-only kernels: 3 `.mm` files and
  3 `.metal` shaders, 2,512 lines.
  - The MPS files also fill 3 of the 4 sparse stubs.
- **Compressed layouts on MPS can't be constructed at this commit.** The constructor maps CPU,
  CUDA, XPU, Meta and `PrivateUse1` to a key, but not MPS
  ([SparseCsrTensor.cpp:357-376](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseCsrTensor.cpp#L357-L376)).
  [pytorch#196205](https://github.com/pytorch/pytorch/pull/196205) would enable them, and is
  open.
- **Out-of-tree devices:** a device outside core (`PrivateUse1`) can construct COO and CSR
  tensors
  ([tensor_new.cpp:558-575](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/torch/csrc/utils/tensor_new.cpp#L558-L575),
  [SparseCsrTensor.cpp:371-372](https://github.com/pytorch/pytorch/blob/a67a74d3e71bf2891b0d02b126fe7390090e1f1c/aten/src/ATen/native/sparse/SparseCsrTensor.cpp#L371-L372)).
  But `native_functions.yaml` registers no kernel for `SparsePrivateUse1` or
  `SparseCsrPrivateUse1`.
  - So a new backend gets none of the generic kernels that MPS and XPU get through hand-written
    entries.
  - A route that registers the generic kernels for any sparse key would serve B, A and G alike.
- **What stays in core without an owner:** about 26 k lines of CPU/CUDA kernels, the cuSPARSE and
  cuSPARSELt upkeep of §1, and the CPU/CUDA parity PRs.

## 7. The Sparse Array Programming Suite

- **What it is:** the Sparse Array Consortium's benchmark suite
  ([SparseArrayProgrammingSuite](https://github.com/SparseArrayProgrammingSuite/SparseArrayProgrammingSuite/tree/8c62415a87305e23db24f12996f34db73dc74d8a)).
  It runs 11 entries: NumPy, SciPy, PyData `sparse`, CuPy, GraphBLAS, Finch, Galley, Scorch,
  "smart", and PyTorch on CPU and on GPU (`competition.config.json`).
- **The references:** reference solutions come from the frameworks of its default environment,
  NumPy, SciPy, PyData `sparse`, "smart", "tagger" and GraphBLAS (`CONTRIBUTING.md`, "Testing
  Other Frameworks"). PyTorch runs in its own environment, as a competitor.
- **The PyTorch adapter:** it handles COO, CSR and CSC. It densifies where torch.sparse lacks an
  op, and leaves `einsum` to the suite's generic evaluator.
- **One more entry on torch:** Scorch runs as a drop-in torch namespace on torch's COO and CSR
  tensors.

## 8. Comparable projects: what kept them going

Commits per calendar year on each repository's default branch, from the GitHub commits API
(2026 is 1 January to 20 November). Governance means a written governance document in the
repository.

| Project | Led by | Governance | Commits |
|---|---|---|---|
| [DGL](https://github.com/dmlc/dgl) | one organization | none | 608 (2024), 7 (2025), 0 (2026); about 110 a month until 2024-08, near zero from 2024-10 |
| [MinkowskiEngine](https://github.com/NVIDIA/MinkowskiEngine) | one organization | none | 210 (2020), 8 (2022), 0 (2024–2026); "will not be maintained" ([#624](https://github.com/NVIDIA/MinkowskiEngine/issues/624), 2025) |
| [MegaBlocks](https://github.com/databricks/megablocks) | one organization | none | 232 (2023), 28 (2024), 8 (2025); archived in 2026 |
| [stk](https://github.com/stanford-futuredata/stk) | one research group | none | 62 (2023), 0 (2025–2026) |
| [SciPy](https://github.com/scipy/scipy) | several institutions | [governance.rst](https://github.com/scipy/scipy/blob/73d740ca203cd4ce087a1a818fb2db576646eacc/doc/source/dev/governance.rst) | 1,980–2,979 a year, 2019–2026 |
| [vLLM](https://github.com/vllm-project/vllm) | several organizations ([sponsors](https://github.com/vllm-project/vllm/blob/fc0119425c8bef73cbeadc265d70777103b3e940/docs/community/sponsors.md)) | [process.md](https://github.com/vllm-project/vllm/blob/77c887b4d68c49a8b15e0aee38cb45a5b4e8c0fc/docs/governance/process.md) | 3,354 (2024), 8,634 (2025), 13,382 (2026) |
| [DeepSpeed](https://github.com/deepspeedai/DeepSpeed) | several organizations, after a change of host in 2025 | [GOVERNANCE.md](https://github.com/deepspeedai/DeepSpeed/blob/25eaac4f2e4df516ce138f74ca5a1a0464cb21de/GOVERNANCE.md) | 615 (2024), 385 (2025), 677 (2026) |

**Reading** (inferred): when the one leading organization's effort stopped, commits fell to near
zero within about a year, and nobody took over. No funder's departure is documented. Projects
with written governance and several backing organizations kept going.
