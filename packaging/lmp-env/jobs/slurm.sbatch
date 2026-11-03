#!/usr/bin/env bash
# Generic Slurm template for a single A100 (sm_80) node.
#
#   sbatch --export=ALL,BUNDLE=$PWD/maceles-lmp-env-20261102-cu12.9-sm80 jobs/slurm.sbatch
#
#SBATCH --job-name=maceles-scan
#SBATCH --partition=gpu
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --time=24:00:00
#SBATCH --output=slurm-%j.out

set -euo pipefail
: "${BUNDLE:?set BUNDLE to the unpacked maceles-lmp-env directory}"
export WORKDIR=${WORKDIR:-$PWD}
export MANIFEST=${MANIFEST:-$WORKDIR/manifest.csv}
bash "$BUNDLE/jobs/entrypoint_scan.sh"
