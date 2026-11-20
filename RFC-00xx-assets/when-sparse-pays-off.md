# When sparse layouts pay off: the public evidence (2026-11-20)

Sparse layouts help when the data are sparse enough. This file lists public sources that measure that,
with what each one measures.

## BSR Triton kernels in pytorch (`torch/sparse/_triton_ops.py`)

| Source | What it measures | Result as stated |
|---|---|---|
| pytorch#110396, "Add scatter_mm and bsr_scatter_mm operations" (2023-10) | `bsr_scatter_mm` against `bsr_dense_mm`, and both against strided matmul | Figures show the "performance equilibrium point": the BSR sparsity at which the BSR kernel matches strided matmul. `bsr_scatter_mm` is about 2× faster than `bsr_dense_mm` for blocks of 16 and 32 on large tensors (RTX 2060 SUPER), up to 2× for blocks of 16 (A100); `bsr_dense_mm` is up to 20 % faster for blocks ≥ 64. |
| pytorch#111760, "Add NVIDIA A100 optimized meta parameters to bsr_dense_mm" (2023-10) | tuned against default Triton parameters | "speeds up `bsr_dense_mm` about 50 %" and "lowers the performance equilibrium points of BSR tensor sparsity and strided tensor for matmul operations" (A100). |
| pytorch#113553 (2023-11) | tuned parameters for bfloat16 and float32, against default parameters (A100) | bfloat16: average/maximum speed-ups of 12–47 % / 29–75 % by block size (`bsr_scatter_mm` and `bsr_dense_mm`); float32 `bsr_dense_mm`: 2–79 % / 5–84 %. |
| pytorch#114757, "Call triton bsr_dense_mm/bsr_dense_addmm kernels on mm/addmm float32 inputs" (2023-11) | `nn.linear` with BSR weights, float32, A100, before and after | average/maximum speed-up 62–72 % / 76–79 % by block size; "sparsity-speedup graphs" before and after. |
| pytorch#133855, #136088, #132646 (2024-08/09) | int8 BSR kernels; tuning for ViT-H shapes (1280, 5120) | as in the titles; no numbers in the text. |

**What this does and doesn't show.**
- **What the text gives:** the numbers in the PR text compare the BSR kernels with earlier versions of themselves.
- **Where BSR beats strided:** the comparison with strided matmul is in figures, as the sparsity above which BSR wins (the "equilibrium point").
- **What it supports:** "above a break-even sparsity that depends on block size, dtype and GPU, BSR matmul is faster than strided".
- **What it doesn't support:** a single speed-up figure.
- **Memory:** the PRs don't measure memory. The saving follows from the format, since only non-zero blocks are stored, but a measured source is still needed.

## Papers

Read from the arXiv full text. "torch.sparse" means the paper or the code that cites it calls torch.sparse.

**Gains against dense.** In three rows (Compressed Skinning, SparseTransX, Longer Attention Span), the
kernels measured are the authors' own, not torch.sparse: these rows show that sparse layouts can pay off,
not that torch.sparse does.

| Paper | Uses | Result, in the paper's words | Conditions |
|---|---|---|---|
| Bundle Adjustment in the Eager Mode (arXiv 2409.12190) | torch.sparse | a dense Jacobian "requires 5.2 TB of memory in double precision, whereas the actual non-zero gradients require only 125 MB" | BAL Ladybug scene; the system matrix has "less than 0.01% density" |
| Compressed Skinning for Facial Blendshapes (2406.11597) | own CPU engine ("an extra sparse matrix-vector multiplication on the CPU"); torch.sparse only in fitting | memory 81k–87k against 486k–612k (5.6–7.2× less); 2.6–3.5× faster | 40 bones, four face rigs |
| Learning Discrete Riemannian Metrics for Physical Fields (2608.14556) | torch.sparse | "10 ms for RHMP (sparse) versus 700 ms for CW Net (dense hidden-state propagation)" | per step |
| SparseTransX (2502.16949) | own kernels (iSpLib SpMM on CPU, DGL g-SpMM on GPU); torch.sparse.mm one optional engine | "up to 5.3x speedup on the CPU and up to 4.2x speedup on the GPU with a significantly low GPU memory footprint" | knowledge-graph embedding training, against scatter-gather implementations (TorchKGE, DGL-KE) |
| Longer Attention Span (2502.01659) | own CUDA kernels ("custom-written algorithms" with "a PyTorch back-end"); torch.sparse only to build masks | CSR: "speedups over SDP of 9.85x, 31.59x, and 7.81x for the V100, L40, and A100"; COO: "0.002x, 0.003x, and 0.001x" | explicit attention masks; the layout decides: CSR wins, COO loses badly |
| Beyond Matryoshka (2503.01776) | torch.sparse (cited code) | sparse embeddings: "2.1× speedup over dense embeddings when sparsity exceeds 80%" | retrieval, 1M items; the method is named "CSR" (Contrastive Sparse Representation) |
| Fast Inference with Kronecker-Sparse Matrices (2405.15013) | comparison | structure-aware implementations (BSR via `torch.sparse.mm` among them) beat the generic dense and sparse baselines; the gain grows with matrix size | A100, 600 patterns |
| TorchAO (2507.16099) | torch.sparse (2:4) | "up to 1.3x speedup ... on ViT models compared to the non-sparse baseline" | semi-structured 2:4 |
| MACKO (2511.13061) | own kernel | Llama2-7B fp16: 13.59 → 8.87 GB and 66.5 → 98.6 tok/s at 50 % sparsity; 2.67 GB and 255 tok/s at 90 %; none at 20 % | own format, not torch.sparse |
| NxMTransformer (2110.15766) | own kernel | cites cuSPARSELt's "1.3X to 1.6X performance uplift" for BERT matmuls; "smaller than the theoretical 2X ... due to memory bandwidth constraints" | A100 |

**No gain, or a loss, against dense**

| Paper | Result, in the paper's words |
|---|---|
| SpQR (2306.03078) | "standard sparse matrix multiplication in PyTorch is not faster than 16-bit inference"; a custom kernel gives 20–30 % |
| Is 2:4 Sparsity Still Worth It? (2610.28947) | "2:4 with FP16 values runs slower than dense FP8 on all three GPUs"; "With BF16 values, the engine's 2:4 path is no faster than dense BF16"; PyTorch's semi-structured conversion is "several times slower" than a fused kernel. Stacked on FP8, 2:4 "speeds up linear layers by 1.3–1.7×" ("prefill-shaped" layers in §6.1), and a retrained 2:4 model served in FP8 by vLLM reaches "27% higher prefill throughput" (H100) |
| Operation-Level Performance Benchmarking of GNNs (2207.09955) | "memory inefficiency often dominate[s] runtime costs moreso than data sparsity alone"; "native Pytorch operations are often as or more competitive than their Pytorch Geometric equivalents, especially at low to moderate levels of input data sparsity" |

**Against torch.sparse, not dense:**
- Scorch (2405.16883): "1.05–5.78 speedups over PyTorch Sparse on end-to-end tasks", CPU inference.
- Partitioning Unstructured Sparse Tensor Algebra (2604.17198): geometric means of 2.0–9.1× over "PyTorch Sparse's general-purpose strategies", "on sparse tensor expressions where specialized algorithms have not been developed".

**Gaps named by papers.** torch-sla (2601.13994): "its only sparse solve, spsolve, is non-differentiable and
implemented as a CPU-only SuperLU wrapper for square systems. torch.linalg is dense-only." This is the
authors' claim. pytorch main registers `_spsolve` for CUDA (in cuDSS builds, not ROCm) and XPU, not for
CPU or MPS (`native_functions.yaml:14698-14702`, at 2026-11-02).

**Reading.**
- **Where gains appear:** they come with very sparse data (0.01 % density in bundle adjustment, more than 80 % sparsity in retrieval), with structure (blocks, Kronecker factors, 2:4), and with the right layout (CSR, not COO, for attention).
- **Where they don't:** they vanish at moderate sparsity, and where a format conversion or memory traffic dominates.

## Forums
- A COO linear layer at 90 % sparsity (2048 × 2048, CPU) trained about 10× slower than dense (12.6 s against 1.23 s), and a masked version about 50 % slower (discuss.pytorch.org/t/198836); pruning masks give no speed-up because they only mask the weights (/209074). These are the other side of the break-even.
- Fixed since: dense→CSR conversion used far more time and memory than dense→COO until April 2025 (dev-discuss.pytorch.org/t/2915; fixed by pytorch#151372, #151474, #151794). It was a conversion cost, not a sparse-against-dense break-even.
