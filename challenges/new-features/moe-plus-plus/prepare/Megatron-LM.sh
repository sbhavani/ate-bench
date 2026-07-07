#!/bin/bash
# Prepare the Megatron-LM workspace for new-features/moe-plus-plus.

set -euo pipefail
source challenges/exports.sh

MEGATRON_LM_PATCH=$(realpath challenges/new-features/moe-plus-plus/patches/Megatron-LM.patch)

setup_codebase()
{
    git init Megatron-LM
    git -C Megatron-LM remote add origin $MEGATRON_LM_URL
    git -C Megatron-LM fetch --depth 1 origin $MEGATRON_LM_SHA
    git -C Megatron-LM checkout FETCH_HEAD
    git -C Megatron-LM apply $MEGATRON_LM_PATCH
}

build_environment()
{
    pushd Megatron-LM
    export NVTE_CUDA_ARCHS="${ATE_NVTE_CUDA_ARCHS:-90;100}"
    export TORCH_CUDA_ARCH_LIST="${ATE_TORCH_CUDA_ARCH_LIST:-9.0 10.0}"
    uv sync --only-group build
    uv pip install --python .venv/bin/python --no-config "torch>=2.10.0" --torch-backend="${ATE_TORCH_BACKEND:-cu130}"
    uv pip install --python .venv/bin/python --no-config cmake ninja zstandard
    export CPATH="$(.venv/bin/python -c 'import nvidia,glob,os;b=nvidia.__path__[0];print(os.pathsep.join(glob.glob(os.path.join(b,"*","include"))))')${CPATH:+:$CPATH}"
    export LIBRARY_PATH="$(.venv/bin/python -c 'import nvidia,glob,os;b=nvidia.__path__[0];print(os.pathsep.join(glob.glob(os.path.join(b,"*","lib"))))')${LIBRARY_PATH:+:$LIBRARY_PATH}"
    uv sync --extra dev --extra mlm --no-group test --no-install-package nvidia-resiliency-ext --inexact
    uv pip install --python .venv/bin/python --no-config --no-build-isolation -C="--build-option=--cpp_ext" -C="--build-option=--cuda_ext" "apex @ git+https://github.com/NVIDIA/apex.git"
    popd
}

setup_workspace()
{
    CMDARG=()
    CMDARG+=(mlfoundations/dclm-baseline-1.0)
    CMDARG+=(--repo-type dataset)
    CMDARG+=(--include "global-shard_03_of_10/local-shard_1_of_10/shard_0000000[0-3]_processed.jsonl.zst")
    CMDARG+=(--local-dir datasets/dclm-baseline/rawtxt)
    hf download ${CMDARG[@]}

    CMDARG=()
    CMDARG+=(deepseek-ai/DeepSeek-V2-Lite)
    CMDARG+=(--repo-type model)
    CMDARG+=(--include "tokenizer*")
    CMDARG+=(--local-dir checkpoints/deepseek-v2-lite)
    hf download ${CMDARG[@]}

    RAWTXT=datasets/dclm-baseline/rawtxt
    TOKTXT=datasets/dclm-baseline/toktxt
    MAX_JOBS=$(( $(nproc) / 24 )); if (( MAX_JOBS < 1 )); then MAX_JOBS=1; fi
    mapfile -t FILES < <(find $RAWTXT -name "*.jsonl.zst" | sort)

    PIDS=()
    for FILE in ${FILES[@]}; do
        OUTPUT_PREFIX=${TOKTXT}/${FILE#${RAWTXT}/}; OUTPUT_PREFIX=${OUTPUT_PREFIX%.jsonl.zst}
        mkdir -p $(dirname $OUTPUT_PREFIX)
        if [[ -f ${OUTPUT_PREFIX}_text_document.idx ]]; then continue; fi

        CMDARG=()
        CMDARG+=(--input $FILE)
        CMDARG+=(--output-prefix $OUTPUT_PREFIX)
        CMDARG+=(--tokenizer-type HuggingFaceTokenizer)
        CMDARG+=(--tokenizer-model checkpoints/deepseek-v2-lite)
        CMDARG+=(--append-eod)
        CMDARG+=(--workers 24)
        python3 Megatron-LM/tools/preprocess_data.py ${CMDARG[@]} & PIDS+=($!)

        if (( ${#PIDS[@]} >= MAX_JOBS )); then
            wait -n
            NEW_PIDS=()
            for PID in "${PIDS[@]}"; do
                if kill -0 $PID 2>/dev/null; then NEW_PIDS+=($PID); fi
            done
            PIDS=(${NEW_PIDS[@]})
        fi
    done

    if (( ${#PIDS[@]} > 0 )); then wait ${PIDS[@]}; fi
}

commit_baseline()
{
    git -C Megatron-LM checkout -b main
    git -C Megatron-LM add -A
    git -C Megatron-LM -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"
}

pushd $1
setup_codebase
build_environment
source Megatron-LM/.venv/bin/activate
setup_workspace
deactivate
commit_baseline
popd
