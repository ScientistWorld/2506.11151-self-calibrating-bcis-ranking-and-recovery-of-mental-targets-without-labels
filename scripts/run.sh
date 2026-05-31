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

# Artifact-contract validation run: regenerate the all-target raw-EEG
# synthetic-random ranking artifact with candidate embeddings so eval computes
# ground-truth distances independently of method code.
export PROTOCOL="${PROTOCOL:-synthetic_random}"
export N_TARGETS="${N_TARGETS:-17}"
export N_OBSERVATIONS="${N_OBSERVATIONS:-300}"
export N_SPLITS="${N_SPLITS:-10}"
export SEED="${SEED:-0}"
export EEG="${EEG:-eeg_raw}"
export OUTPUT="${OUTPUT:-outputs/cursor_ranking.json}"
unset CONTINUE_EXISTING

bash scripts/method.sh
