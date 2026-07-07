<h1 align="center">ATE-Bench</h1>

<p align="center">
  <a href="https://blog.mlc.ai/2026/06/01/pithtrain-compact-agent-native-moe-training-system">Blog</a>
  &nbsp;|&nbsp;
  <a href="https://arxiv.org/abs/2605.31463">Paper</a>
</p>

ATE-Bench (Agent-Task Efficiency) measures how efficiently a coding agent, like [Claude Code](https://www.anthropic.com/claude-code), completes real tasks on ML training frameworks. Each attempt materializes a pinned framework on disposable scratch, hands the agent a challenge, and captures the agent's diff, outputs, and session as a permanent, reproducible record.

## Getting Started

From the repo root, with `uv` and an authenticated agent CLI installed:

```bash
python3 challenges/launch.py <framework> <challenge> [--agent claude|codex]
```

- `<framework>` — one of `torchtitan`, `pith-train`, `Megatron-LM`.
- `<challenge>` — the repo-relative path to a challenge directory, e.g.:

```bash
python3 challenges/launch.py torchtitan challenges/new-features/differential-transformer
```

The default agent is Claude Code, matching the original benchmark:

```bash
python3 challenges/launch.py Megatron-LM challenges/operate-and-profile/getting-started --agent claude
```

Codex can be selected with `--agent codex`:

```bash
python3 challenges/launch.py Megatron-LM challenges/operate-and-profile/getting-started --agent codex
```

Use `--model <name>` to override the default model for the selected agent.
Agent JSON events are teed into `artifacts/<agent>-events.jsonl` and copied into
the final snapshot.

For skill/no-skill experiments, use overlays and an instruction prefix. For
example, to inject Megatron-LM's repository guidance and skills into a prepared
workspace before the agent runs:

```bash
python3 challenges/launch.py Megatron-LM challenges/operate-and-profile/getting-started \
  --agent codex \
  --overlay /path/to/Megatron-LM/AGENTS.md:Megatron-LM/AGENTS.md \
  --overlay /path/to/Megatron-LM/skills:Megatron-LM/skills \
  --instruction-prefix-file experiments/megatron-install-skill-prefix.md
```

Use `--skip-agent --keep-workspace` to validate preparation, overlays, and the
generated agent command without spending on a full agent attempt.

For the Megatron-LM bare-metal install comparison, the paired helper runs the
same challenge once without skills and once with the Megatron install-skill
overlay. The helper skips Apex during prepare by default because the prepare
venv is discarded before the agent runs and the Megatron Core/TE challenge path
does not need Apex; pass `--install-apex` only when intentionally testing that
legacy dependency path. The bundled Megatron challenge patches disable
gradient-accumulation fusion so this no-Apex path remains valid.

```bash
experiments/run-megatron-install-comparison.sh \
  --overlay-root /path/to/Megatron-LM \
  --keep-workspace
```

On CUDA 12.8 SM90 hosts, keep the benchmark's default CUDA 13 path intact but
run the comparison with matching prepare-time overrides:

```bash
experiments/run-megatron-install-comparison.sh \
  --overlay-root /path/to/Megatron-LM \
  --torch-backend cu128 \
  --nvte-cuda-archs 90 \
  --torch-cuda-arch 9.0 \
  --keep-workspace
```

Codex attempt summaries can be generated from snapshots with:

```bash
python3 experiments/summarize_codex_events.py \
  snapshots/challenges/operate-and-profile/getting-started
```

Each run clones the framework at its pinned commit into a throwaway sandbox under `workspace/`, runs the agent, and writes the record (patches, artifacts, session transcript) to `snapshots/<challenge>/<uuid>/`. The challenge categories under `challenges/` are `question-and-answer/` (read-only codebase Q&A), `operate-and-profile/` (run, instrument, and profile a workflow), and `new-features/` (integrate a new architecture).

We recommend pointing `workspace/` at a locally-mounted disk rather than NFS. For full isolation, every task installs its own environment and clones the framework from scratch, so this directory takes heavy, repeated I/O — and local disks are much faster than NFS for it. Only the small, permanent records under `snapshots/` need to live on shared storage. You may additionally set `ANTHROPIC_API_KEY` to authenticate the agent.

## Citation

If you find ATE-Bench useful in your research, please consider citing:

```bibtex
@misc{pithtrain2026,
  title={PithTrain: A Compact and Agent-Native MoE Training System},
  author={Ruihang Lai and Hao Kang and Haozhan Tang and Akaash R. Parthasarathy and Zichun Yu and Junru Shao and Todd C. Mowry and Chenyan Xiong and Tianqi Chen},
  year={2026},
  eprint={2605.31463},
  archivePrefix={arXiv},
  primaryClass={cs.LG},
  url={https://arxiv.org/abs/2605.31463},
}
```
