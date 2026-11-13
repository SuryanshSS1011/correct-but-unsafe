#!/bin/bash
# GPU confirmation kit on Penn State Roar. Fill in the account and partition
# for your allocation before submitting. This script has not been run.
#
# Compute nodes may have no internet access, so do the setup on a login node first:
#
#   git clone https://github.com/SuryanshSS1011/correct-but-unsafe && cd correct-but-unsafe
#   python -m venv .venv && source .venv/bin/activate
#   pip install torch datasets
#   python tools/dump_kernels.py --kernels "$(python -c "import json;t=json.load(open('results/triage-census.json'));print(','.join(i.replace('_r',':') for i in sorted(list(t['out_of_bounds_read'])+list(t['uninitialized_output_source_read'])+t['uninitialized_output_pattern_only'])))")" --out data/gpu_kernels.jsonl
#   sbatch gpu/roar.sbatch
#
#SBATCH --account=<your_account>
#SBATCH --partition=<your_gpu_partition>
#SBATCH --gres=gpu:a100:1
#SBATCH --nodes=1
#SBATCH --ntasks=4
#SBATCH --mem=32gb
#SBATCH --time=2:00:00
#SBATCH --job-name=cbu-gpu
#SBATCH --output=cbu-gpu.%j.out

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate
nvidia-smi --query-gpu=name,memory.total --format=csv
# Triton compiles a small launcher stub, so a C compiler must be on PATH.
python gpu/minimal_repro.py --repeats 5 --out gpu_minimal.json
python gpu/confirm.py --data data/gpu_kernels.jsonl --repeats 3
