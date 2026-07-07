**Task:** Set up the Python environment for the {framework} training framework from scratch.

**Codebase:** You will be working in the `{framework}` codebase. The working branch is
`main`, whose committed state ("setup the workspace") is the clean baseline you start
from. Your job is purely environment setup: build a virtual environment at
`{framework}/.venv` with the dependencies needed for Megatron Core plus CUDA
Transformer Engine. `uv` is installed and is the recommended package manager; Docker is
not available on this host.

**Important scope:**

- Do not download model checkpoints or datasets.
- Do not tokenize data.
- Do not run training scripts.
- Do not install Apex unless you can prove it is required for this smoke test.
- Do not install Mamba/SSM or causal-conv1d kernels unless you can prove they are
  required for this smoke test.
- Do not request elevated permissions. This benchmark runs in non-interactive exec
  mode, so approval prompts cannot be answered.
- Prefer simple shell commands. When a command needs environment variables, use
  `env VAR=value command ...` rather than an interactive shell export sequence.

**Verification workflow:**

1. Build `{framework}/.venv`.
2. Run a CUDA smoke test from the repository workspace that verifies:
   - CUDA is visible to PyTorch.
   - `import megatron.core` succeeds.
   - `import transformer_engine.pytorch` succeeds.
   - `transformer_engine.pytorch.Linear` can run a CUDA forward pass and synchronize.
3. Save the smoke-test output to `artifacts/install-smoke.log`.

**Suggested smoke test:**

```bash
PYTHONPATH=$PWD/{framework} {framework}/.venv/bin/python - <<'PY' | tee artifacts/install-smoke.log
import torch
import megatron.core
from transformer_engine.pytorch import Linear

assert torch.cuda.is_available(), "CUDA is not visible to PyTorch"
layer = Linear(8, 8).cuda()
x = torch.randn(2, 8, device="cuda")
y = layer(x)
torch.cuda.synchronize()
print("install smoke: ok", y.shape)
PY
```

**Deliverables:** A working `{framework}/.venv` and `artifacts/install-smoke.log`
showing `install smoke: ok`. Provide a brief report describing the installation command
sequence and any setup quirks you encountered.

**Start by inspecting the framework source and package metadata, then build the
environment.**
