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

# Paper-style synthetic-random CPU run. The full scripts remain externally
# configurable; these defaults expand the successful five-target setting to
# every target while keeping runtime under the CPU-node limit.
export PROTOCOL="${PROTOCOL:-synthetic_random}"
export N_TARGETS="${N_TARGETS:-17}"
export N_OBSERVATIONS="${N_OBSERVATIONS:-1000}"
export SEED="${SEED:-0}"

bash scripts/method.sh
