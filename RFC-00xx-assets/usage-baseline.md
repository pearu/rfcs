# torch.sparse beside linalg, nested tensors and DTensor (2026-11-19)

**Purpose.** These are the user-stories statistics for torch.sparse, set beside features that serve as
yardsticks:
- **`torch.linalg`:** the main comparison. It is near-complete against `scipy.linalg`, in maintenance mode, and not a candidate to leave core.
- **Nested tensors:** an alternative layout.
- **DTensor:** a tensor data structure under active development.

Each feature went through the same light pipeline as sparse (method below).

## Side by side

| | sparse | linalg | nested | DTensor |
|---|---|---|---|---|
| **Code** (lines, tracked files) | 59,577 | 36,583 | 22,461 | 82,438 |
| … of which ATen C++/CUDA | 27,542 | 19,947 ¹ | 7,803 | 51 |
| … tests | 14,394 | 12,546 | 9,409 | 44,227 |
| **Maintainers** listed in `persons_of_interest.md` | 5, all emeritus | 1 active, 4 emeritus | 6, all emeritus | 9 active, 11 emeritus ² |
| Merge rule (approvers) | Sparse (4, all emeritus) | Linear Algebra (3: 1 active) | none | Distributed (7) ² |
| **Issues opened**, 12 months ³ | 117 | 258 | 24 | 155 |
| … closed as completed | 72 | 145 | 6 | 81 |
| Open issues, all years | 329 | 351 | 115 | 160 |
| **PRs opened** ³ | 119 | 222 | 9 | 720 |
| … landed | 69 (58 %) | 106 (48 %) | 3 | 352 (49 %) |
| … with a human review | 84 % | 70 % | 7 of 9 | 65 % |
| … reviewed by a listed maintainer of the feature | 42 % | 19 % | 2 of 9 | 31 % |
| Days to first human review, median | 0.7 | 0.9 | 1.4 | 0.9 |
| Days from open to landing, median | 6.1 | 7.9 | 1.5 (n = 3) | 5.9 |
| Distinct human reviewers | 36 | 44 | 7 | 99 |
| **Commits on main** touching its files ⁴ | 256 | 273 | 93 | 890 |
| … at least half of the commit's files | 91 | 111 | 16 | 582 |
| **Code search**: raw hits on the entry points ⁵ | 60,607 | 174,372 | 10,839 | 18,688 |
| … sampled hits | 7,842 | 7,134 | 4,238 | 3,913 |
| … kept after the hit filters and the copy step | 3,309 | 4,688 | 529 | 562 |
| Projects calling it outside tests (in the sample) | 959 | 1,903 | 169 | 186 |
| … maintained (pushed in the last 12 months) | 529 | 1,197 | 121 | 170 |
| … Organization / User accounts | 317 / 642 | 559 / 1,344 | 85 / 84 | 117 / 69 |
| … per 1,000 sampled hits | 122 | 267 | 40 | 48 |
| … scaled to all raw hits (an upper estimate) ⁶ | ~7,400 | ~46,500 | ~430 | ~890 |
| **conda-forge packages** calling it outside tests (of 752 that depend on pytorch) | 57 | 193 | 8 | 11 |

**Notes**
1. linalg's ATen count includes dense matmul in `LinearAlgebra.cpp`, so it is an upper bound.
2. DTensor has no maintainer section of its own; the counts are the Distributed section and merge rule, shared with all of `torch.distributed`.
3. **Window:** 2025-11-16 to 2026-11-16, by label:
   - sparse: `module: sparse`, `release notes: sparse`;
   - linalg: `module: linear algebra`, `release notes: linalg_frontend`;
   - nested: `module: nestedtensor`, `release notes: nested tensor`;
   - DTensor: `module: dtensor`, `release notes: distributed (dtensor)`.

   Labels undercount all four alike.
4. **Commits:** 2025-11-02 to 2026-11-02, on pytorch main, against the exact file list of the size count.
5. **Entry points:**
   - sparse: `sparse_coo_tensor`, `torch.sparse.mm`, `to_sparse_csr`, `sparse_csr_tensor`;
   - linalg: `torch.linalg.solve`, `.eigh`, `.svd`, `.norm`;
   - nested: `torch.nested.nested_tensor`, `.as_nested_tensor`;
   - DTensor: `distribute_tensor`, `DTensor.from_local`.

   Each query was sampled at about 2,000 hits, so sparse and linalg have about twice the sample of nested and DTensor.
6. **The estimate:** projects found × raw hits ÷ sampled hits. It overstates, because a project with many hits is likely found by any sample. Use it only for the order of magnitude.

## Reading the table

- **Size:** sparse is the larger codebase. Its ATen code is 1.4× linalg's, and linalg's count includes dense matmul.
- **Use, measured directly:** linalg has about 3× as many conda-forge packages (193 against 57), about 2× as many projects in equal-size samples (1,903 against 959), and about 6× by the rough scaling.
  - Nested tensors (8 packages) and DTensor (11) are used directly far less than sparse.
  - **DTensor's caveat:** most of its use comes through training frameworks such as FSDP2 and tensor-parallel wrappers, which direct-call counts don't see.
- **Activity:** sparse has about half of linalg's issues and PRs, and its PRs get reviewed and landed at similar speeds (0.7 against 0.9 days to first review; 6.1 against 7.9 days to land).
  - The difference is who reviews: listed sparse maintainers, all emeritus, review 42 % of sparse PRs. Listed linalg maintainers review 19 % of linalg PRs.
  - DTensor is under very active development: 720 PRs and 99 reviewers in the year, with 9 active maintainers, which sparse and nested tensors don't have.
- **Nested tensors** resemble sparse in maintenance (all listed maintainers emeritus, no merge rule), but are smaller and less used.

**How to read the comparison.** Two points frame these numbers; `when-sparse-pays-off.md` has the public sources.
- **Complexity:** sparse is more complex than strided linalg. Its layouts are harder, sparse versions of linalg algorithms are often unavailable, and the more complex data structure costs performance against strided code. So sparse is useful where the data are sparse enough, not everywhere linalg is.
- **Sparsity decides:** where data are sparse by nature, a sparse layout can cut memory greatly and speed things up at high sparsity. The pytorch PRs on the BSR Triton kernels show a break-even sparsity above which BSR beats strided matmul. Papers report gains at very high sparsity, for example 5.2 TB dense against 125 MB sparse for a bundle-adjustment Jacobian. Other papers report no gain at moderate sparsity.

  On this reading, a smaller user base than linalg's is expected, and concentrated where the data are sparse.

## Method

- **Same filters as sparse** (`user-stories.md`): torch itself and its trees, vendored files and toy names are dropped, and each hit's pattern is confirmed; repositories with fewer than 10 commits, or with no packaging and no paper link, are dropped; copies are found by file content and by package tree, and the upstream or the oldest holder is kept. Each kept file is rescanned at its commit with comments and docstrings removed.
- **Patterns:** the entry points of note 5, as calls. A project counts when a non-test file calls the feature.
- **conda-forge:** all 752 pytorch dependents, scanned with the same patterns. A package counts when a non-vendored, non-test file calls the feature.
