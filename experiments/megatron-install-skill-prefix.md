Before starting the challenge, read `Megatron-LM/AGENTS.md` if present. Then
read the relevant Megatron-LM skills before forming an install plan:

- `Megatron-LM/skills/mcore-build-and-dependency/SKILL.md`
- `Megatron-LM/skills/mcore-transformer-engine-install/SKILL.md`, if present

Use those skills as mandatory task context. In particular, distinguish
Megatron's CI/container `uv sync` workflow from user/source installs, which use
`uv pip install -e ...` after bootstrapping CUDA-enabled PyTorch with
`uv pip install --no-config`.
