#!/bin/bash
# Download all data and models needed for this environment.
#
# Run on a node with internet access BEFORE building the container
# or submitting compute jobs.
#
# This script should be idempotent — safe to run multiple times.
# It should skip files that already exist.
#
# Update the storage estimate below as you add downloads.
#
# Estimated total download size: 0.03 GB
# Estimated disk usage after extraction: 0.03 GB
#
# This script writes only to workspace-local paths so a fresh clone can
# reproduce the environment without machine-local caches.

set -e

cd "$(dirname "$0")/.."

mkdir -p data/downloads

if [ ! -e data/downloads/all_data_sorted.npz ]; then
  wget -O data/downloads/all_data_sorted.npz \
    https://raw.githubusercontent.com/jgrizou/neurips-self-calibrating-bci/main/src/data/all_data_sorted.npz
fi
