#!/usr/bin/env bash
# Run paired Codex attempts for the Megatron-LM getting-started install task.

set -euo pipefail

CHALLENGE=challenges/operate-and-profile/getting-started
AGENT=codex
MODEL=
OVERLAY_ROOT=
TORCH_BACKEND=
NVTE_CUDA_ARCHS=
TORCH_CUDA_ARCH_LIST=
SKIP_AGENT=0
KEEP_WORKSPACE=0

usage() {
    cat <<'EOF'
usage: experiments/run-megatron-install-comparison.sh --overlay-root /path/to/Megatron-LM [options]

Runs two ATE-Bench attempts:
  1. baseline: pinned Megatron-LM with no added install skills
  2. with-skills: pinned Megatron-LM overlaid with AGENTS.md, skills, and install docs

options:
  --overlay-root PATH   Megatron-LM checkout containing the install skill changes
  --challenge PATH      Challenge directory (default: operate-and-profile/getting-started)
  --agent NAME          Agent backend (default: codex)
  --model NAME          Agent model override
  --torch-backend NAME  Override ATE_TORCH_BACKEND for CUDA host compatibility
  --nvte-cuda-archs X   Override ATE_NVTE_CUDA_ARCHS, e.g. 90
  --torch-cuda-arch X   Override ATE_TORCH_CUDA_ARCH_LIST, e.g. 9.0
  --skip-agent          Prepare/capture only; useful for harness validation
  --keep-workspace      Keep prepared workspace directories
  -h, --help            Show this help
EOF
}

while [[ $# -gt 0 ]]; do
    case "$1" in
        --overlay-root)
            OVERLAY_ROOT=$2
            shift 2
            ;;
        --challenge)
            CHALLENGE=$2
            shift 2
            ;;
        --agent)
            AGENT=$2
            shift 2
            ;;
        --model)
            MODEL=$2
            shift 2
            ;;
        --torch-backend)
            TORCH_BACKEND=$2
            shift 2
            ;;
        --nvte-cuda-archs)
            NVTE_CUDA_ARCHS=$2
            shift 2
            ;;
        --torch-cuda-arch)
            TORCH_CUDA_ARCH_LIST=$2
            shift 2
            ;;
        --skip-agent)
            SKIP_AGENT=1
            shift
            ;;
        --keep-workspace)
            KEEP_WORKSPACE=1
            shift
            ;;
        -h|--help)
            usage
            exit 0
            ;;
        *)
            echo "unknown argument: $1" >&2
            usage >&2
            exit 2
            ;;
    esac
done

if [[ -z "$OVERLAY_ROOT" ]]; then
    echo "--overlay-root is required" >&2
    usage >&2
    exit 2
fi

OVERLAY_ROOT=$(realpath "$OVERLAY_ROOT")
if [[ ! -f "$OVERLAY_ROOT/AGENTS.md" ]]; then
    echo "overlay root is missing AGENTS.md: $OVERLAY_ROOT" >&2
    exit 1
fi
if [[ ! -d "$OVERLAY_ROOT/skills" ]]; then
    echo "overlay root is missing skills/: $OVERLAY_ROOT" >&2
    exit 1
fi

if [[ -n "$TORCH_BACKEND" ]]; then
    export ATE_TORCH_BACKEND=$TORCH_BACKEND
fi
if [[ -n "$NVTE_CUDA_ARCHS" ]]; then
    export ATE_NVTE_CUDA_ARCHS=$NVTE_CUDA_ARCHS
fi
if [[ -n "$TORCH_CUDA_ARCH_LIST" ]]; then
    export ATE_TORCH_CUDA_ARCH_LIST=$TORCH_CUDA_ARCH_LIST
fi

COMMON_ARGS=(Megatron-LM "$CHALLENGE" --agent "$AGENT")
if [[ -n "$MODEL" ]]; then
    COMMON_ARGS+=(--model "$MODEL")
fi
if [[ "$SKIP_AGENT" -eq 1 ]]; then
    COMMON_ARGS+=(--skip-agent)
fi
if [[ "$KEEP_WORKSPACE" -eq 1 ]]; then
    COMMON_ARGS+=(--keep-workspace)
fi

echo "== baseline: no Megatron install skills =="
python3 challenges/launch.py "${COMMON_ARGS[@]}" --run-label baseline

echo "== with-skills: Megatron install skill overlay =="
python3 challenges/launch.py "${COMMON_ARGS[@]}" \
    --run-label with-skills \
    --overlay "$OVERLAY_ROOT/AGENTS.md:Megatron-LM/AGENTS.md" \
    --overlay "$OVERLAY_ROOT/skills:Megatron-LM/skills" \
    --overlay "$OVERLAY_ROOT/docs/get-started/install.md:Megatron-LM/docs/get-started/install.md" \
    --overlay "$OVERLAY_ROOT/README.md:Megatron-LM/README.md" \
    --overlay "$OVERLAY_ROOT/pyproject.toml:Megatron-LM/pyproject.toml" \
    --instruction-prefix-file experiments/megatron-install-skill-prefix.md

SNAPSHOT_ROOT=snapshots/$CHALLENGE
if [[ "$AGENT" == "codex" && -d "$SNAPSHOT_ROOT" ]]; then
    python3 experiments/summarize_codex_events.py "$SNAPSHOT_ROOT"
fi
