Clean CUDA container target for this benchmark run:

- You are running inside a disposable CUDA 12.8 development container with
  `nvcc`, Python 3.12, git, build-essential, CMake, Ninja, and `uv` available.
- Build `Megatron-LM/.venv` with Python 3.12.
- Use CUDA-enabled PyTorch `torch==2.10.0` from the `cu128` backend/index.
- Use pinned PyPI Transformer Engine with the single package spec
  `transformer_engine[pytorch]==2.11.0`.
- Target the visible RTX PRO Blackwell proxy GPU: `NVTE_CUDA_ARCHS=120` and
  `TORCH_CUDA_ARCH_LIST=12.0`.
- Use the configured `UV_CACHE_DIR`; do not switch to `~/.cache/uv`.
- Do not install `.[dev]`, `.[te]`, Apex, Mamba/SSM, or causal-conv1d.
- The benchmark measures installation only: do not download datasets, tokenize
  data, or run training.
