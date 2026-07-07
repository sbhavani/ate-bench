#!/usr/bin/env bash
# Run paired Codex attempts for the Megatron-LM install-only task.

set -euo pipefail

CHALLENGE=challenges/operate-and-profile/install-only
AGENT=codex
MODEL=
OVERLAY_ROOT=
AGENT_CONTAINER_IMAGE=
AGENT_CONTAINER_USER=
COMMON_INSTRUCTION_PREFIX_FILE=
TORCH_BACKEND=
NVTE_CUDA_ARCHS=
TORCH_CUDA_ARCH_LIST=
INSTALL_APEX=0
SKIP_AGENT=0
KEEP_WORKSPACE=0
CODEX_BYPASS_APPROVALS=1
CODEX_SANDBOX=workspace-write
STOP_AFTER_SUCCESS_ARTIFACT=1

usage() {
    cat <<'EOF'
usage: experiments/run-megatron-install-comparison.sh --overlay-root /path/to/Megatron-LM [options]

Runs two ATE-Bench attempts:
  1. baseline: pinned Megatron-LM with no added install skills
  2. with-skills: pinned Megatron-LM overlaid with AGENTS.md, skills, and install docs

options:
  --overlay-root PATH   Megatron-LM checkout containing the install skill changes
  --challenge PATH      Challenge directory (default: operate-and-profile/install-only)
  --agent NAME          Agent backend (default: codex)
  --model NAME          Agent model override
  --agent-container-image IMAGE
                        Run the agent inside this Docker image
  --agent-container-user USER
                        User passed to Docker for the agent container
  --common-instruction-prefix-file PATH
                        Prepend shared benchmark instructions to both attempts
  --torch-backend NAME  Override ATE_TORCH_BACKEND for CUDA host compatibility
  --nvte-cuda-archs X   Override ATE_NVTE_CUDA_ARCHS, e.g. 90
  --torch-cuda-arch X   Override ATE_TORCH_CUDA_ARCH_LIST, e.g. 9.0
  --install-apex        Install Apex during prepare; skipped by default
  --no-codex-bypass-approvals
                        Do not pass Codex's bypass flag; useful for managed accounts
  --codex-sandbox MODE  Sandbox passed to Codex (default: workspace-write)
  --no-stop-after-success-artifact
                        Let the agent final-answer instead of stopping once smoke passes
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
        --agent-container-image)
            AGENT_CONTAINER_IMAGE=$2
            shift 2
            ;;
        --agent-container-user)
            AGENT_CONTAINER_USER=$2
            shift 2
            ;;
        --common-instruction-prefix-file)
            COMMON_INSTRUCTION_PREFIX_FILE=$2
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
        --install-apex)
            INSTALL_APEX=1
            shift
            ;;
        --no-codex-bypass-approvals)
            CODEX_BYPASS_APPROVALS=0
            shift
            ;;
        --codex-sandbox)
            CODEX_SANDBOX=$2
            shift 2
            ;;
        --no-stop-after-success-artifact)
            STOP_AFTER_SUCCESS_ARTIFACT=0
            shift
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
export ATE_INSTALL_APEX=$INSTALL_APEX

COMMON_ARGS=(Megatron-LM "$CHALLENGE" --agent "$AGENT")
if [[ -n "$MODEL" ]]; then
    COMMON_ARGS+=(--model "$MODEL")
fi
if [[ -n "$AGENT_CONTAINER_IMAGE" ]]; then
    COMMON_ARGS+=(--agent-container-image "$AGENT_CONTAINER_IMAGE")
fi
if [[ -n "$AGENT_CONTAINER_USER" ]]; then
    COMMON_ARGS+=(--agent-container-user "$AGENT_CONTAINER_USER")
fi
if [[ -n "$COMMON_INSTRUCTION_PREFIX_FILE" ]]; then
    COMMON_ARGS+=(--instruction-prefix-file "$COMMON_INSTRUCTION_PREFIX_FILE")
fi
if [[ "$AGENT" == "codex" && "$CODEX_BYPASS_APPROVALS" -eq 0 ]]; then
    COMMON_ARGS+=(--no-codex-bypass-approvals)
fi
if [[ "$AGENT" == "codex" ]]; then
    COMMON_ARGS+=(--codex-sandbox "$CODEX_SANDBOX")
fi
COMMON_ARGS+=(
    --success-artifact artifacts/install-smoke.log
    --success-artifact-contains "install smoke: ok"
)
if [[ "$STOP_AFTER_SUCCESS_ARTIFACT" -eq 1 ]]; then
    COMMON_ARGS+=(--stop-after-success-artifact)
fi
if [[ "$SKIP_AGENT" -eq 1 ]]; then
    COMMON_ARGS+=(--skip-agent)
fi
if [[ "$KEEP_WORKSPACE" -eq 1 ]]; then
    COMMON_ARGS+=(--keep-workspace)
fi

BASE_UV_CACHE_DIR="${UV_CACHE_DIR:-}"

run_launch() {
    local label=$1
    shift

    if [[ -n "$BASE_UV_CACHE_DIR" ]]; then
        local attempt_cache="$BASE_UV_CACHE_DIR/$label"
        local attempt_pip_cache="$attempt_cache/pip"
        mkdir -p "$attempt_cache"
        mkdir -p "$attempt_pip_cache"
        echo "Using UV_CACHE_DIR=$attempt_cache"
        UV_CACHE_DIR="$attempt_cache" PIP_CACHE_DIR="$attempt_pip_cache" python3 challenges/launch.py "$@"
    else
        python3 challenges/launch.py "$@"
    fi
}

echo "== baseline: no Megatron install skills =="
run_launch baseline "${COMMON_ARGS[@]}" --run-label baseline

echo "== with-skills: Megatron install skill overlay =="
run_launch with-skills "${COMMON_ARGS[@]}" \
    --run-label with-skills \
    --overlay "$OVERLAY_ROOT/AGENTS.md:Megatron-LM/AGENTS.md" \
    --overlay "$OVERLAY_ROOT/skills:Megatron-LM/skills" \
    --overlay "$OVERLAY_ROOT/docs/get-started/install.md:Megatron-LM/docs/get-started/install.md" \
    --overlay "$OVERLAY_ROOT/README.md:Megatron-LM/README.md" \
    --overlay "$OVERLAY_ROOT/pyproject.toml:Megatron-LM/pyproject.toml" \
    --instruction-prefix-file experiments/megatron-install-skill-prefix.md

SNAPSHOT_ROOT=snapshots/$CHALLENGE
if [[ "$AGENT" == "codex" ]]; then
    python3 experiments/summarize_codex_events.py "workspace/$CHALLENGE" "$SNAPSHOT_ROOT"
fi
