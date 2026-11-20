# Who uses torch.sparse (2026-11-20)

**What this is.** A count of the public projects that call torch.sparse, with the layouts, devices and
ops they use, a ranking of layout × device cells, and the 50 highest-ranked projects read by hand.

The comparison with linalg, nested tensors and DTensor is in `usage-baseline.md`. The public evidence on when
sparse layouts pay off is in `when-sparse-pays-off.md`.

**Sources.**
- **S1:** GitHub code search.
- **S2:** the dependents of the seed libraries.
- **S3:** PyPI projects active in the last 24 months.
- **S4:** every conda-forge package that depends on pytorch.
- **S5:** pytorch issues.
- **S6:** papers.
- **S7:** Q&A.
- **S8:** Hugging Face.
- **S9:** ecosystem lists, including the PyTorch landscape.

**Records.** The per-project records are not published with this report, because many of them name personal
GitHub accounts. The 50 highest-ranked projects are listed below.

## How to read the numbers

- **Partly sampled, so partly lower bounds.**
  - The code-search part of every count is a lower bound: ten queries were sampled, and code search covers default branches only.
  - The PyPI and conda-forge parts are a census, not a sample.
  - Code misses private and closed-source users.
- **Filters,** the same for every source. These are dropped or set apart:
  - pytorch itself and its copies, vendored files and copies of other projects;
  - toy and tutorial repositories, and tests and examples;
  - comments and docstrings, and out-of-tree backends.
- **Past users are not counted:** projects whose latest release no longer has their sparse path.
- **By rule, unless read by hand.**
  - **Field:** a keyword rule, wrong for an estimated 11–17 % of the projects it classifies.
  - **Device:** the devices named in the sparse-calling files.
  - **Read by hand:** the 50 highest-ranked projects; see "The hand check".
- **A use is a call on the code path.**
  - It doesn't say how much a project depends on sparse: several top projects use it in one optional component.
  - "Reach" counts packages that depend on a project, not packages that use its sparse path.

## Headline

**1,798 public projects call torch.sparse outside their tests.** This is a lower bound in its code-search
part (see above).
- **Maintenance:** 1,052 had activity in the last 12 months.
- **Packages:** 575 are published on PyPI or conda-forge.
- **Set apart:** 27 Hugging Face models and Spaces, and 43 projects that use only torch.masked.
- **By source:** 1,418 come from code search and conda-forge. The PyPI census adds 314 packages, papers 47 repositories and ecosystem lists 19.

| | Projects | Maintained |
|---|---|---|
| deep learning | 1,101 | 538 |
| scientific computing | 525 | 415 |
| unclassified | 172 | 99 |

| GitHub owner | Projects | Maintained | ≥ 1 PyPI dependent | ≥ 100 |
|---|---|---|---|---|
| personal account (User) | 1,132 | 613 | 42 | 1 |
| Organization | 576 | 362 | 86 | 9 |
| PyPI package with no repository found | 85 | 73 | 14 | 0 |
| account no longer resolves | 5 | 4 | 0 | 0 |

- **Reach:** 142 projects have at least one PyPI dependent, 37 have ten or more, and 10 have a hundred or more.
- **Owner type** is reported, not filtered on: personal accounts hold widely used projects too, such as torch_sparse and PyTorch-Wavelet-Toolbox.

## The PyPI census (S3)

| Step | Count |
|---|---|
| PyPI projects with any change in the last 24 months (Simple index serial ≥ 26.0M) | 474,629 |
| … of which the newest file is older than 24 months / no files at all (not judged) | 7,050 / 11,507 |
| … whose newest wheel declares `torch` (PEP 658 metadata) | 20,661 (13,729 required, 6,932 as an extra) |
| … or a seed library: torch-geometric 623, torchao 137, dgl 61, torch-sparse 57 | (most also declare torch) |
| Packages downloaded and scanned | 20,712; 10 not scanned (7 over 200 MB, 3 failed downloads) |
| … calling the sparse API outside vendored and bundled files | 558 |
| … after the filters: kept with code-path use / tests only / copies / pytorch copy | 516 / 4 / 37 / 1 |

- **Share:** 516 of the 20,661 packages that declare torch call torch.sparse in their own code, 2.5 %. Of the 13,729 that require torch directly, 382 do, 2.8 %. On conda-forge, 50 of the 752 pytorch dependents are counted users, 6.6 %.
- **Field:** the 314 PyPI-only packages are split almost evenly: 137 scientific, 132 deep learning, 45 unclassified.
- **Copies:**
  - The filters drop copies by the same path rule as S1 (bundled environments, `site-packages/`, `third_party/`).
  - They also drop a top-level package directory that another package owns, and the 11 copies found by reading, which include vendored whisper, sentence-transformers and NeMo code.
  - The scan kept only the hits, not the files, so copies could not be matched by content. More copies may remain among the PyPI-only packages; their number is not known.
- **Limits:**
  - The census reads only the newest wheel's metadata, so 12,900 sdist-only projects were not judged.
  - A package that reaches torch only through another library, and doesn't declare torch, is missed. PyG's own wheel declares no torch requirement; it is covered as a seed.

## Layouts and devices

**Layouts.** These are projects calling each layout outside tests. Layouts come from code-path files, a
project can use several, and the counts are partly from sampled queries.

| Layout | Projects | Maintained | Scientific | of them found only by the census, papers or lists |
|---|---|---|---|---|
| COO | 1,203 | 762 | 378 | 311 |
| CSR | 443 | 364 | 192 | 71 |
| CSC | 97 | 74 | 42 | 17 |
| BSR | 49 | 40 | 14 | 9 |
| BSC | 36 | 31 | 10 | 8 |
| semi-structured (2:4) | 32 | 21 | 3 | 3 |

**Devices.**
- **What counts:** the devices named in the sparse-calling files, by rule. For the top 50 they are read by hand and mapped to the same vocabulary.
- **"Generic"** means code that follows its input's device.

| Device | Projects | Scientific |
|---|---|---|
| generic (follows its input) | 1,267 | 390 |
| CPU named, or no device named | 749 | 237 |
| CUDA | 727 | 206 |
| MPS | 52 | 22 |
| ROCm | 15 | 7 |
| XPU | 11 | 4 |

## The ranking

**What is ranked:** the 1,798 counted projects. Hugging Face and torch.masked-only projects are set apart, as in the headline.

**Layout × device.** Projects, with weights in parentheses. A project counts once per cell it uses.
- **The weight** is confidence (1, ½ or ¼) × activity (1 if maintained, ½ if not) × (1 + log10(1 + PyPI dependents)).
- **Columns:** "cpu" means CPU named or no device named; "generic" means device-generic code.

| Layout | cpu | generic | cuda | rocm | mps | xpu |
|---|---|---|---|---|---|---|
| COO | 514 (409) | 907 (731) | 475 (369) | 11 (12) | 42 (47) | 9 (9) |
| CSR | 197 (183) | 338 (323) | 214 (199) | 11 (11) | 23 (26) | 6 (6) |
| CSC | 44 (40) | 72 (77) | 38 (38) | 5 (5) | 13 (16) | 4 (4) |
| BSR | 20 (23) | 35 (35) | 29 (28) | 6 (8) | 9 (8) | 4 (4) |
| BSC | 18 (20) | 24 (26) | 18 (17) | 5 (5) | 9 (8) | 4 (4) |
| 2:4 | 10 (7) | 22 (16) | 21 (17) | 4 (6) | 3 (2) | 4 (4) |

- **No cell is empty.**
  - BSR, BSC and the accelerator columns are thin. Part of them comes from testing, typing and inspection tools that list every layout and device.
- **Sensitivity:** ranked by weight and by count, no cell's rank moves by more than a third of the 36 cells. The ranking doesn't depend on the weight formula.

**Top ops** (projects, weight, of them scientific):

| op | projects | weight | scientific |
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

**Requested cells** come from issues (S5, 22) and Q&A (S7, 53). The main asks:
- **Unspecified device:** mostly COO and CSR;
- **MPS:** CSR (3, 2 of them naming a project), COO (3) and BSR (1, pypose/bae);
- **2:4:** batched matrices, Conv2d and training.

Of the 75 issue and Q&A records, 13 name a project, 5 of them requests; only those count as demand. Q&A's pain points:
- missing ops 18, autograd 12, performance 9;
- solvers 5, memory 5, bugs 5.

**cuSPARSE and cuSPARSELt dependence** (inferred from the code, not measured):
- **cuSPARSE:** 625 projects run sparse matmul in CUDA or device-generic code; 189 are scientific.
- **cuSPARSELt:** 13 call it explicitly, and 19 more use 2:4 through cuSPARSELt or CUTLASS.

## The top 50, read by hand

The 50 highest-ranked projects were read against their code, at HEAD or the latest released package.
- **How many:** 66 projects were read in all, because removals let others in.
- **Evidence:** each reading records file:line evidence; the field is the record's.

In the table, "(o)" marks an optional path: a flag, a model choice, or a fallback.

| # | Project | Owner | Field | What calls sparse | Layouts | Device, read |
|---|---|---|---|---|---|---|
| 1 | sentence-transformers | org | DL | sparse encoder model, similarity | COO, CSR | CPU, CUDA, MPS |
| 2 | torchmetrics | org | DL | clustering metrics | COO | follows input |
| 3 | peft | org | DL | two adapter types | COO | follows input |
| 4 | vllm | org | DL | guard against malformed sparse inputs (defensive) | — | follows input |
| 5 | PyG | org | DL | sparse adjacency | COO, CSR, CSC | CPU, CUDA, MPS |
| 6 | openai-whisper | org | DL | alignment-heads mask buffer | COO | follows the model |
| 7 | trl | org | DL | one experimental trainer | COO | CPU, CUDA |
| 8 | DeepSpeed | org | DL | sparse gradients | COO | follows the training device |
| 9 | POT | org | sci | optimal transport | COO | none named |
| 10 | linear_operator | org | sci | Gaussian processes | COO, CSR | follows input |
| 11 | cellpose | org | sci | cell segmentation | COO | input's device |
| 12 | pyod | user | uncl. | adjacency in the CoLA graph anomaly detector (o) | COO | follows input (CPU in practice) |
| 13 | torchao | org | DL | 2:4 and BSR weights (o) in v0.18.0, the latest release. At HEAD, `sparsity/` is removed "due to lack of widespread usage" (pytorch/ao#4728); a float8 2:4 path remains through torchao's own CUTLASS ops (CUDA) and on ROCm | BSR, 2:4 | CUDA; HEAD also ROCm |
| 14 | scvi-tools | org | sci | experimental sparse loader (o) | CSR, CSC | follows input |
| 15 | torch_sparse | user | DL | conversions; sparse×sparse matmul via `torch.sparse.mm` | COO, CSR, CSC | follows input |
| 16 | Fireworks SDK | org | DL | CSR as a wire encoding (o) | CSR | CPU |
| 17 | mteb | org | DL | SPLADE-style sparse embeddings (o) | COO | CPU |
| 18 | pyfixest | org | sci | econometrics | COO, CSR, CSC | all; COO on MPS |
| 19 | SAELens | org | DL | TopK sparse activations (o); autocast off: "sparse.mm fails with bf16" | COO | follows input |
| 20 | pytorch3d | org | DL | mesh Laplacians | COO | follows input |
| 21 | qiskit-machine-learning | org | sci | sparse quantum-network outputs and gradients (o) | COO | follows input |
| 22 | nvmath-python | org | sci | interop with all five layouts | all five | CPU and CUDA |
| 23 | coola | user | DL | compares and hashes sparse tensors (o) | all five | follows input |
| 24 | whisper-timestamped | org | DL | alignment-heads mask | COO | follows input |
| 25 | pykeen | org | DL | R-GCN message passing, WL colouring | COO, CSR | follows input |
| 26 | gsplat | org | DL | sparse gradients of visible Gaussians (o) | COO | CUDA |
| 27 | HanLP | user | DL | scatter idiom in SRL decoding | COO | follows input |
| 28 | cvxpylayers | org | sci | differentiable convex optimisation | CSR | GPU and compile; CPU uses scipy |
| 29 | tonic | org | DL | event frames as sparse tensors (o) | COO | CPU |
| 30 | PyTorch-Wavelet-Toolbox | user | sci | wavelet matrices | COO | follows input |
| 31 | relbench | org | DL | CSR as a ragged index list | COO, CSR | CPU |
| 32 | tinker | org | DL | CSR as a wire encoding, automatic when smaller | CSR | CPU |
| 33 | suite2p | org | sci | ROI masks; MPS falls back to CPU "until sparse works on MPS" | COO, CSC | CUDA default |
| 34 | cornac | org | DL | graph propagation in bundle recommenders | COO | CUDA or CPU |
| 35 | lenskit | org | DL | interaction matrices, LightGCN, SparseAdam (o) | COO, CSR | GPU by default |
| 36 | instanseg | org | sci | instance masks; MPS moved to CPU | COO | follows input |
| 37 | dcoupler | user | sci | hydrological grid remapping (o) | COO | CPU |
| 38 | pytorch-optimizer | user | DL | MADGRAD's sparse-gradient path (o) | COO | follows parameters |
| 39 | deepinv | org | sci | sparse histogram operator (o) | COO | follows input |
| 40 | SYMFLUENCE | org | sci | river-network adjacency for GNN routing | COO | CUDA if available |
| 41 | PyPOTS | user | DL | builds a banded matrix, densifies at once | COO | follows input |
| 42 | omicverse | user | sci | single-cell PCA, spectral init (o) | COO, CSR | CUDA; MPS goes to MLX |
| 43 | neuropythy | user | sci | cortical-mesh summation matrices | COO | CPU default |
| 44 | sglang | org | DL | BGE-M3 sparse embedding output (o) | COO | GPU |
| 45 | SQcircuit | org | sci | quantum-circuit operators (o) | COO | CPU |
| 46 | PhiML | org | sci | sparse matrices throughout its torch backend | all five | follows input |
| 47 | torch-harmonics | org | sci | DISCO filter basis, densified | COO | follows input |
| 48 | pypose | org | sci | BSR/CSR matmul, sparse Jacobians in an opt-in sparse LM (o; needs `bae`); MPS wanted (pytorch#201801) | COO, CSR, BSR | follows input |
| 49 | torch-sla | org | sci | differentiable sparse solvers. Its `torch.sparse.spsolve` backend probes first; its authors write that the op "is only registered for the MPS backend in mainline PyTorch", but pytorch main registers `_spsolve` for CUDA (in cuDSS builds, not ROCm) and XPU, not for MPS or CPU (`native_functions.yaml:14698-14702`, at 2026-11-02) | COO, CSR | follows input; CUDA, ROCm, CPU |
| 50 | anemoi-graphs | org | sci | graph edge weights exported as a sparse matrix (o) | COO | follows input |

**What the hand check changed**
- **Past users (2), set apart as xformers was:**
  - **NeMo:** its k2 MAP loss went in "Clean ASR older and non-maintained models" (2026-03-23), and is absent from nemo-toolkit 3.0.0.
  - **PhysicsNeMo:** its regridding plugin went in a dependency clean-up (2026-02-14); it is absent from v2.0.0 onwards.
- **Not users (6), removed from the counts:**
  - tensordict and helion only read `sparse_grad` to reject it;
  - fairseq2 lists sparse factories in a dtype hook;
  - apache-beam's `to_sparse()` is TensorFlow's;
  - executorch and xinference hold copies of pytorch and whisper code.
- **Doubtful (7), kept at low confidence:** skorch, deepchem, cdlib, zeta, deepxde, atomworks and pyzag. In each, the sparse path is public but unused, or vestigial.
- **A duplicate:** torch-sla's two repositories, a personal one and its organization's, are counted once.
- **Devices:** the rule's devices were replaced by what the code does, and mapped to the same vocabulary.

**What the code shows beyond counts.**
- **Small parts of large projects.** In many top projects sparse is one optional component: a model adapter, a loss, an experimental path.
- **A data format, not compute.** Several use sparse tensors only to build or ship data, densifying at once: CSR as a wire encoding in tinker and Fireworks, COO as a scatter idiom in HanLP and PyPOTS.
- **MPS is avoided where it would be needed:** suite2p, instanseg and omicverse route around it.
- **`spsolve` is hard to rely on.** torch-sla probes for `torch.sparse.spsolve` before using it. Its authors write that the op runs only on MPS, but pytorch main registers `_spsolve` for CUDA (in cuDSS builds, not ROCm) and XPU, not for MPS or CPU (`native_functions.yaml:14698-14702`, at 2026-11-02). A forum post (S7) on an RTX 40 card shows the cuDSS requirement in practice.
- **Some sparse paths were removed in 2026:**
  - **torchao** removed its `sparsity/` package at HEAD "due to lack of widespread usage of this feature" (pytorch/ao#4728). The latest release, v0.18.0, still has it.
  - **NeMo** removed its sparse loss with "older and non-maintained models".
  - **PhysicsNeMo**'s removal came with a dependency clean-up and no stated reason.
  - **xformers** removed its cuSPARSELt 2:4 code in 2026-09, pointing users to torch.sparse.

## Scientific stories, by name

Scientific projects are 525 of the 1,798. The hand-checked and highest-weight ones, by field:
- **Optimisation and statistics:** POT, linear_operator (Gaussian processes), cvxpylayers, pyfixest (econometrics).
- **Biology and medicine:** cellpose, instanseg, suite2p, scvi-tools, omicverse, spateo, nichecompass, scimilarity, BulkFormer, QHap (genomics).
- **Physics, weather and Earth systems:** anemoi, graph_weather, torch-harmonics, dcoupler and SYMFLUENCE (hydrology), neuraloperator, deepxde.
- **Numerical methods:** PhiML, torch-sla, TensorGalerkin (FEM), nvmath-python, PyTorch-Wavelet-Toolbox, deepinv.
- **Quantum:** qiskit-machine-learning, SQcircuit.
- **Robotics and neuroscience:** pypose and pypose/bae, neuropythy, norse.

## Papers, Q&A and issues

These sources say why sparse is used and what is missing.
- **Papers (S6):** 28 read in full.
  - **Why sparse:** speed 18, memory 11, the data's structure 10.
  - **Results:** 10 report gains against dense, 3 report none. Of the 10, 4 measure torch.sparse, 5 measure the authors' own kernels, and 1 compares implementations that include torch.sparse BSR.
  - **Where gains appear:** with very sparse or structured data and the right layout. Examples:
    - a bundle-adjustment Jacobian is 5.2 TB dense against 125 MB sparse;
    - attention with CSR, on the authors' own kernels, is 7.8–31.6× faster than dense, while COO is hundreds of times slower.
  - **Where they don't:** they vanish at moderate sparsity, and where format conversion or memory traffic dominates. SpQR: PyTorch's sparse matmul "is not faster than 16-bit inference".
  - The details are in `when-sparse-pays-off.md`.
  - **Checked against their sources:** the paper quotes against the arXiv full text, and the forum posts against their pages, by a second reader. Every source cited there and here has been checked.
- **Q&A (S7)** and **issues (S5)** name the missing pieces:
  - ops;
  - autograd through sparse inputs;
  - sparse solvers;
  - fp16 and bf16 support;
  - MPS.

## Filters and drop rates

- **S1 code search,** 47,345 hits:
  - vendored or torch source 22,563;
  - pytorch itself and trees 10,479;
  - not confirmed 2,970;
  - copies 2,919;
  - toy names 1,673;
  - kept 6,741.

  The kept hits are in 4,779 repositories:
  - 1,466 hold only copies;
  - 1,671 are small or unpackaged (fewer than 10 commits, or no packaging, without a paper link);
  - 1,642 are kept.
- **S3 PyPI:** of 558 packages with hits outside vendored and bundled files, 37 are copies, 1 is a pytorch copy and 4 are tests-only.
- **S4 conda-forge:** 752 scanned; 50 counted users.
- **Hand check:** 66 projects read; 6 not users, 2 past, 7 doubtful, 1 duplicate.

## Limits

- **Sampling and reach.**
  - Code search is sampled, and it covers default branches only.
  - The census misses sdist-only packages and packages that reach torch only indirectly.
  - Private and closed-source users are invisible, except through issues, forums and papers.
- **Copies.** The PyPI copy filter matches paths and package names, not file contents. Beyond the 11 copies found by reading and removed, more may remain among the PyPI-only packages; their number is not known.
- **Rules.**
  - Field and device are by rule outside the top 50. The field error rate is 11–17 % (95 % interval 3–39 %).
  - A paper link doesn't prove that a repository is the paper's own code. In a sample of 10 small paper repositories, 3 were not users.
- **Reach.**
  - "Depends on" is not "uses the sparse path". For PyG, 1,146 repositories call its sparse path; only 167 of them are certain to produce torch.sparse tensors.
- **Sources not fully read.**
  - Semantic Scholar's full-text snippet search, which needs an API key.
  - OpenAlex's later result pages (its keyless budget).
  - Hugging Face code beyond search terms and the top downloads.
