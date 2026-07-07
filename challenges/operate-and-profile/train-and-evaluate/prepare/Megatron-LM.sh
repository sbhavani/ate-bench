#!/bin/bash
# Prepare the Megatron-LM workspace for operate-and-profile/train-and-evaluate.

set -euo pipefail
source challenges/exports.sh

MEGATRON_LM_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/Megatron-LM.patch)
MEGATRON_BRIDGE_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/Megatron-Bridge.patch)
LM_EVALUATION_HARNESS_PATCH=$(realpath challenges/operate-and-profile/train-and-evaluate/patches/lm-evaluation-harness.patch)

setup_codebase()
{
    git init Megatron-LM
    git -C Megatron-LM remote add origin $MEGATRON_LM_URL
    git -C Megatron-LM fetch --depth 1 origin $MEGATRON_LM_SHA
    git -C Megatron-LM checkout FETCH_HEAD
    git -C Megatron-LM apply $MEGATRON_LM_PATCH

    git init Megatron-Bridge
    git -C Megatron-Bridge remote add origin $MEGATRON_BRIDGE_URL
    git -C Megatron-Bridge fetch --depth 1 origin $MEGATRON_BRIDGE_SHA
    git -C Megatron-Bridge checkout FETCH_HEAD
    git -C Megatron-Bridge apply $MEGATRON_BRIDGE_PATCH

    git init lm-evaluation-harness
    git -C lm-evaluation-harness remote add origin $LM_EVALUATION_HARNESS_URL
    git -C lm-evaluation-harness fetch --depth 1 origin $LM_EVALUATION_HARNESS_SHA
    git -C lm-evaluation-harness checkout FETCH_HEAD
    git -C lm-evaluation-harness apply $LM_EVALUATION_HARNESS_PATCH
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
    uv pip install --python .venv/bin/python --no-config accelerate omegaconf hydra-core datasets tensorboard rich six
    popd

    pushd lm-evaluation-harness
    uv venv
    uv pip install --python .venv/bin/python setuptools wheel
    uv pip install --python .venv/bin/python -e ".[vllm]"
    uv pip install --python .venv/bin/python "vllm<0.11" "transformers>=4.55.2,<5" ray wandb matplotlib
    popd
}

commit_baseline()
{
    git -C Megatron-LM checkout -b main
    git -C Megatron-LM add -A
    git -C Megatron-LM -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"

    git -C Megatron-Bridge checkout -b main
    git -C Megatron-Bridge add -A
    git -C Megatron-Bridge -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"

    git -C lm-evaluation-harness checkout -b main
    git -C lm-evaluation-harness add -A
    git -C lm-evaluation-harness -c user.email=harness@local -c user.name=harness commit -m "setup the workspace"
}

pushd $1
setup_codebase
build_environment
commit_baseline
popd
