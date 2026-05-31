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
# configurable. These defaults expand the number of observations per target
# while reducing CV repeats so the one-core CPU job finishes within the
# scheduler limit; the previous scored all-target run used 10 splits at N=300.
export PROTOCOL="${PROTOCOL:-synthetic_random}"
export N_TARGETS="${N_TARGETS:-17}"
export N_OBSERVATIONS="${N_OBSERVATIONS:-500}"
export N_SPLITS="${N_SPLITS:-5}"
export SEED="${SEED:-0}"

bash scripts/method.sh
