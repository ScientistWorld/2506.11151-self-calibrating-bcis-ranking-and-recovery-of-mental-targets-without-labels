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

# Expanded CPU run. The full scripts remain externally configurable; these
# defaults test whether the smoke-run effect holds across multiple targets.
export N_TARGETS="${N_TARGETS:-5}"
export N_OBSERVATIONS="${N_OBSERVATIONS:-500}"
export SEED="${SEED:-0}"

bash scripts/method.sh
