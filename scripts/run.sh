#!/bin/bash
# Example job script submitted via the end-of-turn YAML block.
#
# This runs inside the compute container with:
#   - Your workspace mounted at /home/user
#   - GPU(s) available
#   - No internet access
#
# Make sure all data, code, and dependencies are already in the
# workspace or baked into the container before submitting.

set -e

cd /home/user

# Compact CPU smoke run. The full scripts remain externally configurable;
# these defaults keep the first verification inside the test-node limit.
export N_TARGETS="${N_TARGETS:-1}"
export N_OBSERVATIONS="${N_OBSERVATIONS:-300}"
export SEED="${SEED:-0}"

bash scripts/method.sh
