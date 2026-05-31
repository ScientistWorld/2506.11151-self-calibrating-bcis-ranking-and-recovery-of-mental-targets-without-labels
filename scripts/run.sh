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

# Secondary-claim CPU run: repeat the paper-style synthetic-random ranking
# benchmark on the released EEGNet embeddings. The raw-EEG all-target result
# already exists in outputs/cursor_ranking.json; the final evaluation merges
# both artifacts into scoring/scores.json.
export PROTOCOL="${PROTOCOL:-synthetic_random}"
export N_TARGETS="${N_TARGETS:-17}"
export N_OBSERVATIONS="${N_OBSERVATIONS:-300}"
export N_SPLITS="${N_SPLITS:-5}"
export SEED="${SEED:-0}"
export EEG="${EEG:-eeg_net}"
export OUTPUT="${OUTPUT:-outputs/cursor_ranking_eegnet.json}"
export CONTINUE_EXISTING=1

bash scripts/method.sh
bash scripts/evaluate.sh outputs/cursor_ranking.json outputs/cursor_ranking_eegnet.json
