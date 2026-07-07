#!/bin/bash
# Prepare a Megatron-LM workspace for install-only benchmarking.

set -euo pipefail
source challenges/exports.sh

setup_codebase()
{
    git init Megatron-LM
    git -C Megatron-LM remote add origin $MEGATRON_LM_URL
    git -C Megatron-LM fetch --depth 1 origin $MEGATRON_LM_SHA
    git -C Megatron-LM checkout FETCH_HEAD
}

commit_baseline()
{
    git -C Megatron-LM checkout -b main
    git -C Megatron-LM add -A
    git -C Megatron-LM -c user.email=harness@local -c user.name=harness commit --allow-empty -m "setup the workspace"
}

pushd $1
setup_codebase
commit_baseline
popd
